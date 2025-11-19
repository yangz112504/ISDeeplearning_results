import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


# ============================================================
# DEVICE
# ============================================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print("Using device:", device)


# ============================================================
# MNIST DATASET
# ============================================================
transform_train = transforms.Compose([
    transforms.RandomCrop(28, padding=2),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

batch_size = 64
num_workers = 2
pin_mem = device.type == "cuda"

trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
testset  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=pin_mem)
testloader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                          num_workers=num_workers, pin_memory=pin_mem)


# ============================================================
# CUSTOM ACTIVATIONS
# ============================================================
LARGE = 1000

def arctan(x, sigma=1.0):
    return 0.5 + (1.0 / torch.pi) * torch.atan(sigma * x)

def arctan_approx(x, sigma=1.0):
    z = sigma * x
    return (0.5 + torch.clamp(z, min=0)) / (1.0 + torch.abs(z))

def zailu(x, sigma=1.0):
    return x * arctan(x, sigma)

def zailu_approx(x, sigma=1.0):
    return x * arctan_approx(x, sigma)

try:
    zailu_c = torch.compile(zailu)
    zailu_approx_c = torch.compile(zailu_approx)
except:
    zailu_c = zailu
    zailu_approx_c = zailu_approx


# ============================================================
# RESNET18 (MODIFIED FOR MNIST)
# ============================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, act_fun=F.relu, downsample=None):
        super().__init__()
        self.act_fun = act_fun

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.act_fun(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.act_fun(out + identity)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, act_fun="relu", sigma_param=None, num_classes=10):
        super().__init__()
        self.sigma = sigma_param

        self.act = self._get_activation(act_fun)

        self.in_channels = 64

        # MNIST is 1-channel → adjust input conv
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def _get_activation(self, act):
        if act == "relu": return F.relu
        if act == "silu": return F.silu
        if act == "gelu": return F.gelu
        if act == "zailu": return lambda x: zailu_c(x, 1.0)
        if act == "zailu_approx": return lambda x: zailu_approx_c(x, 1.0)
        return F.relu

    def _make_layer(self, block, out_channels, blocks, stride=1):
        down = None
        if stride != 1 or self.in_channels != out_channels:
            down = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [block(self.in_channels, out_channels, stride, self.act, down)]
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, act_fun=self.act))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        return self.fc(x)


def ResNet18(act_fun="relu", sigma_param=None):
    return ResNet(BasicBlock, [2,2,2,2], act_fun=act_fun, sigma_param=sigma_param)


# ============================================================
# TRAIN CLEAN MODEL
# ============================================================
def train_clean(model, trainloader, epochs=10, lr=1e-3, weight_decay=5e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {running_loss / len(trainloader):.4f}")


# ============================================================
# TEST WITH GAUSSIAN NOISE (MISH PAPER)
# ============================================================
def test_with_noise(model, testloader, sigma):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            if sigma > 0:
                # same thing as torch.randn(x.shape)
                # Add Gaussian noise to inputs
                inputs = inputs + sigma * torch.randn_like(inputs)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(testloader), 100.0 * correct / total


# ============================================================
# MAIN NOISE ROBUSTNESS EXPERIMENT
# ============================================================
ACTIVATIONS = ["relu", "silu", "gelu", "zailu", "zailu_approx"]

def run_noise_experiment(num_trials=3, epochs=100, save_dir="mnist_resnet_noise"):
    os.makedirs(save_dir, exist_ok=True)

    sigmas = list(range(9))   # noise 0..8
    all_results = {}

    for act in ACTIVATIONS:
        print("\n===================================================")
        print(f" Running 3 trials for activation: {act}")
        print("===================================================\n")

        trial_dfs = []

        for trial in range(num_trials):
            print(f"\n--- Trial {trial+1}/{num_trials} for {act} ---")

            # Build fresh model
            model = ResNet18(act_fun=act).to(device)

            # Train clean
            train_clean(model, trainloader, epochs=epochs)

            losses, accs = [], []

            # Noise test
            for sigma in sigmas:
                loss, acc = test_with_noise(model, testloader, sigma)
                losses.append(loss)
                accs.append(acc)
                print(f"σ={sigma} | loss={loss:.4f} | acc={acc:.2f}%")

            df = pd.DataFrame({
                "sigma": sigmas,
                "test_loss": losses,
                "test_acc": accs,
                "trial": trial
            })

            df.to_csv(f"{save_dir}/noise_curve_{act}_trial{trial+1}.csv", index=False)
            trial_dfs.append(df)

        # combine all 3 trials
        full_df = pd.concat(trial_dfs, ignore_index=True)
        full_df.to_csv(f"{save_dir}/noise_curve_{act}_ALL_TRIALS.csv", index=False)

        all_results[act] = full_df

    return all_results

# ============================================================
# PLOTTING
# ============================================================
def plot_noise_curves(results, save_dir="mnist_noise_plots"):
    os.makedirs(save_dir, exist_ok=True)

    for act, df in results.items():

        # Average across trials
        grouped = df.groupby("sigma").agg({
            "test_loss": "mean",
            "test_acc": "mean"
        }).reset_index()

        # === LOSS PLOT ===
        plt.figure(figsize=(6,4))
        plt.plot(grouped["sigma"], grouped["test_loss"], marker="o")
        plt.xlabel("Noise Sigma")
        plt.ylabel("Test Loss")
        plt.title(f"ResNet18 Noise Robustness - {act}")
        plt.grid(True)
        plt.savefig(f"{save_dir}/loss_{act}.png")
        plt.close()

        # === ACCURACY PLOT ===
        plt.figure(figsize=(6,4))
        plt.plot(grouped["sigma"], grouped["test_acc"], marker="o")
        plt.xlabel("Noise Sigma")
        plt.ylabel("Accuracy (%)")
        plt.title(f"ResNet18 Noise Robustness - {act}")
        plt.grid(True)
        plt.savefig(f"{save_dir}/acc_{act}.png")
        plt.close()



# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    results = run_noise_experiment(num_trials=3, epochs=100)
    plot_noise_curves(results)
