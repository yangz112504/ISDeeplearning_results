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

LARGE = 1000  # threshold for switching to ReLU behavior

# ---------------- Device ---------------- #
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print("Using device:", device)

# ---------------- Training ---------------- #
def train_net(model, trainloader, testloader, epochs=10, lr=1e-3, weight_decay=5e-4):
    assert trainloader is not None and testloader is not None, "Pass loaders into train_net"
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epoch_loss_train, epoch_loss_test, epoch_accs = [], [], []

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

        avg_train_loss = running_loss / len(trainloader)
        epoch_loss_train.append(avg_train_loss)

        # --- Eval ---
        model.eval()
        test_running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()

                preds = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        avg_test_loss = test_running_loss / len(testloader)
        acc = 100.0 * correct / total

        epoch_loss_test.append(avg_test_loss)
        epoch_accs.append(acc)

        print(f"[Epoch {epoch+1}/{epochs}] Train {avg_train_loss:.4f} | Test {avg_test_loss:.4f} | Acc {acc:.2f}%")

    return epoch_loss_train, epoch_loss_test, epoch_accs

# ---------------- Dataset (CIFAR-10, 32x32) ---------------- #
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

batch_size = 64
pin_mem = device.type == "cuda"
num_workers = min(4, os.cpu_count() or 2)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=pin_mem)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, pin_memory=pin_mem)

def get_activation(act_name):
    act_name = act_name.lower()
    if act_name == "relu": return F.relu
    if act_name == "gelu": return F.gelu
    if act_name == "silu": return F.silu
    raise ValueError(f"Unknown activation: {act_name}")

# ---------------- ResNet18 (CIFAR-10) ---------------- #
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, act_name="relu", downsample=None):
        super().__init__()
        self.act_name = act_name
        self.act_fun = get_activation(act_name)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_fun(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act_fun(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, act_fun="relu", num_classes=10):
        super().__init__()
        self.act_name = act_fun
        self.act_fun = get_activation(act_fun)

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._init_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [block(self.in_channels, out_channels, stride, self.act_name, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, act_name=self.act_name))
        return nn.Sequential(*layers)

    def _init_weights(self):
        nonlinearity = "relu" if self.act_name == "relu" else "linear"
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nonlinearity)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_fun(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet18(act_fun="relu", num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], act_fun, num_classes)

# ---------------- Experiment Runner ---------------- #
def run_experiments(activations, params=None, num_trials=3, epochs=100, save_dir="results",
                    network_name="ResNet18", dataset_name="CIFAR10", last_n_epochs=10,
                    trainloader=None, testloader=None):
    """
    Runs experiments for a single network (network_name).
    - activations: list of activation names (e.g., "zailu", "zailu_approx", "relu", "gelu_a")
    - params: list of parameter values (used for param sweeps). If None, activations are run without param sweeps.
    """
    os.makedirs(save_dir, exist_ok=True)
    all_results = {}

    for trial in range(num_trials):
        print(f"\n=== Trial {trial+1}/{num_trials} for {network_name} ===")

        for act in activations:
            label = act
            print(f"\nTraining {network_name} with activation={act}...")

            model = ResNet18(act_fun=act).to(device)

            train_losses, test_losses, accs = train_net(
                model, trainloader, testloader, epochs=epochs
            )

            key = f"{network_name}_{act}"
            if key not in all_results:
                all_results[key] = {"train_losses": [], "test_losses": [], "accuracies": []}

            all_results[key]["train_losses"].append(train_losses)
            all_results[key]["test_losses"].append(test_losses)
            all_results[key]["accuracies"].append(accs[-1])

            df = pd.DataFrame({
                "epoch": range(1, epochs+1),
                "train_loss": train_losses,
                "test_loss": test_losses,
                "accuracy": accs
            })
            df.to_csv(f"{save_dir}/{network_name}_{act}_trial{trial+1}.csv", index=False)
            print(f"✅ Saved trial data to {save_dir}/{network_name}_{act}_trial{trial+1}.csv")

    # --- Summary stats ---
    summary_rows = []
    for key, data in all_results.items():
        # data["train_losses"] shape: (num_trials, epochs)
        train_losses = np.array(data["train_losses"])
        test_losses = np.array(data["test_losses"])
        accs = np.array(data["accuracies"])

        # take last_n_epochs across axis 1 (epochs)
        last_n = min(last_n_epochs, train_losses.shape[1])
        mean_train = train_losses[:, -last_n:].mean()
        std_train = train_losses[:, -last_n:].std()
        mean_test = test_losses[:, -last_n:].mean()
        std_test = test_losses[:, -last_n:].std()

        net_name = key.split("_")[0]
        act_part = "_".join(key.split("_")[1:])  # activation name possibly with _param
        act_name = act_part
        param_val = np.nan

        summary_rows.append({
            "network": net_name,
            "dataset": dataset_name,
            "activation": act_name,
            "param": param_val,
            "mean_train_loss": mean_train,
            "std_train_loss": std_train,
            "mean_test_loss": mean_test,
            "std_test_loss": std_test,
            "mean_accuracy": accs.mean(),
            "std_accuracy": accs.std()
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{save_dir}/{network_name}_{dataset_name}_summary.csv", index=False)
    return all_results, summary_df

# ---------------- Plotting ---------------- #
def plot_results(all_results, epochs=100, save_dir="plots", network_name="ResNet18"):
    os.makedirs(save_dir, exist_ok=True)

    def extract_param_val(key):
        # key example: "ResNet18_zailu_param0"
        # return float param if present otherwise inf so base comes last
        if "_param" in key:
            try:
                return float(key.split("_param")[-1])
            except:
                return float("inf")
        return float("inf")

    # activation names are second token in keys split by "_"
    activation_types = sorted(set("_".join(k.split("_")[1:]).split("_param")[0] for k in all_results.keys()))
    for act in activation_types:
        keys = [k for k in all_results if k.endswith(f"_{act}")]
        keys_sorted = sorted(keys, key=extract_param_val)
        if not keys_sorted:
            continue

        # Train loss
        plt.figure(figsize=(6, 3))
        for k in keys_sorted:
            arr = np.array(all_results[k]["train_losses"])  # (trials, epochs)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            accs = np.array(all_results[k]["accuracies"])
            if accs.max() <= 1.5: accs *= 100
            label = k.split("_", 2)[2] if len(k.split("_", 2))>2 else k
            plt.plot(range(1, epochs+1), mean, label=f"{label} (Acc {accs.mean():.2f}%)")
            plt.fill_between(range(1, epochs+1), mean-std, mean+std, alpha=0.2)
        plt.xlabel("Epoch"); plt.ylabel("Train Loss")
        plt.title(f"{network_name} - {act.upper()} Train Loss")
        plt.legend(title="param", bbox_to_anchor=(1.05,1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{network_name}_{act}_train_loss.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Test loss
        plt.figure(figsize=(6, 3))
        for k in keys_sorted:
            arr = np.array(all_results[k]["test_losses"])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            accs = np.array(all_results[k]["accuracies"])
            if accs.max() <= 1.5: accs *= 100
            label = k.split("_", 2)[2] if len(k.split("_", 2))>2 else k
            plt.plot(range(1, epochs+1), mean, label=f"{label} (Acc {accs.mean():.2f}%)")
            plt.fill_between(range(1, epochs+1), mean-std, mean+std, alpha=0.2)
        plt.xlabel("Epoch"); plt.ylabel("Test Loss")
        plt.title(f"{network_name} - {act.upper()} Test Loss")
        plt.legend(title="param", bbox_to_anchor=(1.05,1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{network_name}_{act}_test_loss.png", dpi=300, bbox_inches="tight")
        plt.close()

# ---------------- Main ---------------- #
if __name__ == "__main__":
    # Experiment hyperparams
    num_trials = 3
    epochs = 100
    activations = ["relu", "gelu", "silu"]

    t_start = time.time()
    all_summaries = []

    
    all_results, summary_df = run_experiments(
        activations=activations,
        num_trials=num_trials,
        epochs=epochs,
        save_dir="results",
        network_name="ResNet18",
        dataset_name="CIFAR10",
        last_n_epochs=10,
        trainloader=trainloader,
        testloader=testloader
    )
    plot_results(all_results, epochs=epochs, save_dir=f"plots_ResNet18", network_name="ResNet18")
    all_summaries.append(summary_df)

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_summary.to_csv("results/ALL_NETWORKS_CIFAR10_summary.csv", index=False)

    print("\nCombined summary across all networks:")
    print(combined_summary)
    print(f"\nTotal training time: {time.time() - t_start:.2f} seconds")
