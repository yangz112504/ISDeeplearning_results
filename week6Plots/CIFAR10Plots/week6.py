import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time

LARGE = 1000  # threshold for switching to ReLU behavior

# ---------------- Training Function ---------------- #
def train_net(model, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    epoch_loss_train = []
    epoch_loss_test = []
    epoch_accs = []   # <-- track accuracy for each epoch

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(mps_device), labels.to(mps_device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(trainloader)
        epoch_loss_train.append(avg_train_loss)
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}")

        # --- Validation / Test ---
        model.eval()
        test_running_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(mps_device), labels.to(mps_device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_test_loss = test_running_loss / len(testloader)
        epoch_loss_test.append(avg_test_loss)

        accuracy = 100 * correct / total
        epoch_accs.append(accuracy)   # <-- store accuracy per epoch
        print(f"[Epoch {epoch+1}/{epochs}] Test Loss: {avg_test_loss:.4f} | Accuracy: {accuracy:.2f}%")

    return epoch_loss_train, epoch_loss_test, epoch_accs


# ---------------- Dataset Setup ---------------- #
mps_device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
mps_device = torch.device("cuda")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 64
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

#add Vgg16
class Vgg16(nn.Module):
    def __init__(self, act_fun="relu", num_classes=10, param=None):
        super(Vgg16, self).__init__()
        self.param = param
        self.act_fun = self._get_act_fun(act_fun)

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def _get_act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "gelu":
            return F.gelu
        elif act_fun == "silu":
            return F.silu
        elif act_fun == "gelu_a":
            return lambda x: gelu_a(x, self.param)
        elif act_fun == "silu_a":
            return lambda x: silu_a(x, self.param)
        elif act_fun == "zailu":
            return lambda x: zailu(x, self.param)
        else:
            return F.relu
    
    # Once I get to the ReLU layers, I will apply the custom activation function instead
    def forward(self, x):
        for layer in self.features:
            if isinstance(layer, nn.ReLU):
                x = self.act_fun(x)
            else:
                x = layer(x)
        x = torch.flatten(x, 1)
        # Once I get to the ReLU layers, I will apply the custom activation function instead
        for layer in self.classifier:
            if isinstance(layer, nn.ReLU):
                x = self.act_fun(x)
            else:
                x = layer(x)
        return x

# ---------------- Custom Activations ---------------- #
def gelu_a(x, a=1):
    if a >= LARGE:
        return F.relu(x)
    kAlpha = 0.70710678118654752440
    return x * 0.5 * (1 + torch.erf(a * x * kAlpha))

def silu_a(x, a=1):
    if a >= LARGE:
        return F.relu(x)
    return x * torch.sigmoid(a * x)

# def zailu(x, sigma=1.0):
#     z = sigma * x
#     return x * 0.5 * (1 + 2 * F.relu(z)) / (1 + torch.abs(z))

def zailu(x, s=1):
    if s >= LARGE:
        return F.relu(x)
    return x * (2 * (1/4 + 1/(2 * torch.pi) * torch.arctan(s * x)))


# ---------------- Experiment Runner ---------------- #
def run_experiments(activations, params=None, num_trials=5, epochs=20, save_dir="results",
                    network_name="MLP", dataset_name="CIFAR10", last_n_epochs=10, use_mlp=True):
    os.makedirs(save_dir, exist_ok=True)
    all_results = {}

    for trial in range(num_trials):
        print(f"\n=== Trial {trial+1}/{num_trials} ===")

        for act in activations:
            if act in ["gelu_a", "silu_a", "zailu"] and params is not None:
                for p in params:
                    print(f"\nTraining {act.upper()} model with param={p}...")

                    # Choose model
                    model = Vgg16(act_fun=act, param=p).to(mps_device)
                    net_name = "VGG16"

                    # Xavier init
                    for layer in model.modules():
                        if isinstance(layer, nn.Linear):
                            torch.nn.init.xavier_uniform_(layer.weight)

                    train_losses, test_losses, accs = train_net(model, epochs=epochs)

                    key = f"{act}_param{p}"
                    if key not in all_results:
                        all_results[key] = {"train_losses": [], "test_losses": [], "accuracies": []}

                    all_results[key]["train_losses"].append(train_losses)
                    all_results[key]["test_losses"].append(test_losses)
                    all_results[key]["accuracies"].append(accs[-1])  # final accuracy

                    # Save per-trial results (per epoch)
                    df = pd.DataFrame({
                        "epoch": list(range(1, epochs+1)),
                        "train_loss": train_losses,
                        "test_loss": test_losses,
                        "accuracy": accs   # <-- per-epoch accuracy
                    })
                    df.to_csv(f"{save_dir}/{net_name}_{key}_trial{trial+1}.csv", index=False)

    # Summary stats
    summary_rows = []
    for key, data in all_results.items():
        train_losses = np.array(data["train_losses"])[:, -last_n_epochs:]
        test_losses = np.array(data["test_losses"])[:, -last_n_epochs:]
        accs = np.array(data["accuracies"])

        summary_rows.append({
            "network": net_name,
            "dataset": dataset_name,
            "activation": key.split("_param")[0],
            "param": float(key.split("param")[-1]),
            "mean_train_loss": train_losses.mean(),
            "std_train_loss": train_losses.std(),
            "mean_test_loss": test_losses.mean(),
            "std_test_loss": test_losses.std(),
            "mean_accuracy": accs.mean(),
            "std_accuracy": accs.std()
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{save_dir}/{net_name}_{dataset_name}_summary.csv", index=False)

    return all_results, summary_df


# ---------------- Plotting ---------------- #
def plot_results(all_results, epochs=20, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    def extract_param_val(key):
        return float(key.split("param")[-1])

    activation_types = set(key.split("_")[0] for key in all_results.keys())
    for act in activation_types:
        keys = [key for key in all_results.keys() if key.startswith(act)]
        keys.sort(key=extract_param_val)

        # Train loss plot
        plt.figure(figsize=(6, 3))
        for key in keys:
            avg_train_loss = np.mean(all_results[key]["train_losses"], axis=0)
            std_train_loss = np.std(all_results[key]["train_losses"], axis=0)
            accs = np.array(all_results[key]["accuracies"])
            if accs.max() <= 1.5: accs *= 100
            plt.plot(range(1, epochs + 1), avg_train_loss,
                     label=f"param={extract_param_val(key)} (Acc {accs.mean():.2f}%)")
            plt.fill_between(range(1, epochs + 1),
                             avg_train_loss - std_train_loss,
                             avg_train_loss + std_train_loss, alpha=0.2)
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.title(f"{act.upper()} Train Loss")
        plt.legend(title="param value", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(f"{save_dir}/{act}_train_loss.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Test loss plot
        plt.figure(figsize=(6, 3))
        for key in keys:
            avg_test_loss = np.mean(all_results[key]["test_losses"], axis=0)
            std_test_loss = np.std(all_results[key]["test_losses"], axis=0)
            accs = np.array(all_results[key]["accuracies"])
            if accs.max() <= 1.5: accs *= 100
            plt.plot(range(1, epochs + 1), avg_test_loss,
                     label=f"param={extract_param_val(key)} (Acc {accs.mean():.2f}%)")
            plt.fill_between(range(1, epochs + 1),
                             avg_test_loss - std_test_loss,
                             avg_test_loss + std_test_loss, alpha=0.2)
        plt.xlabel("Epoch")
        plt.ylabel("Test Loss")
        plt.title(f"{act.upper()} Test Loss")
        plt.legend(title="param value", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(f"{save_dir}/{act}_test_loss.png", dpi=300, bbox_inches="tight")
        plt.close()


# ---------------- Main ---------------- #
num_trials = 3
epochs = 100
a_values = [0, 0.25, 0.5, 1, 2, 5, 1000]
s_values = [0, 0.25, 0.5, 1, 2, 5, 1000]

t_start = time.time()

param_activations = ["gelu_a", "silu_a"]
results_param_as, summary_df_as = run_experiments(
    activations=param_activations,
    params=a_values,
    num_trials=num_trials,
    epochs=epochs,
    save_dir="results",
    use_mlp=False
)
plot_results(results_param_as, epochs=epochs)

param_activations = ["zailu"]
results_param_zailu, summary_df_zailu = run_experiments(
    activations=param_activations,
    params=s_values,
    num_trials=num_trials,
    epochs=epochs,
    save_dir="results",
    use_mlp=False
)
plot_results(results_param_zailu, epochs=epochs)

summary_df_all = pd.concat([summary_df_as, summary_df_zailu], ignore_index=True)
summary_df_all.to_csv("results/VGG16_CIFAR10_all_summary.csv", index=False)

print("\nCombined summary across all activations:")
print(summary_df_all)
print(f"Total training time: {time.time() - t_start:.2f} seconds")
