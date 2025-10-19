import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import time

# --- Parameters ---
LARGE = 1000
trials = 10000

# --- Device setup ---
devices = [torch.device("cpu")]

if torch.cuda.is_available():
    devices.append(torch.device("cuda"))
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")

if torch.backends.mps.is_available():
    devices.append(torch.device("mps"))
    print("✅ MPS available (Apple GPU)")

print("Using devices:", devices)

# --- Custom activations ---
_mish = nn.Mish()
mish = _mish if hasattr(F, "mish") else lambda x: x * torch.tanh(F.softplus(x))

def zailuApprox(x):
    return x * 0.5 * (1 + 2 * F.relu(x)) / (1 + torch.abs(x))

def zailuNormal(x):
    return x * (2 * (1/4 + 1/(2 * torch.pi) * torch.arctan(x)))

# --- Activation functions ---
actfun = {
    "elu": F.elu,
    "hardshrink": F.hardshrink,
    "hardsigmoid": F.hardsigmoid,
    "hardtanh": F.hardtanh,
    "hardswish": F.hardswish,
    "leaky_relu": F.leaky_relu,
    "logsigmoid": F.logsigmoid,
    "prelu": lambda x: F.prelu(x, torch.tensor(0.25, device=x.device)),
    "relu": F.relu,
    "relu6": F.relu6,
    "rrelu": F.rrelu,
    "selu": F.selu,
    "celu": F.celu,
    "gelu": F.gelu,
    "sigmoid": torch.sigmoid,
    "silu": F.silu,
    "mish": mish,
    "softplus": F.softplus,
    "softshrink": F.softshrink,
    "softsign": F.softsign,
    "tanh": torch.tanh,
    "tanhshrink": F.tanhshrink,
    "threshold": lambda x, th=0.5, val=0.0: F.threshold(x, th, val),
    "glu": F.glu,
    "identity": lambda x: x,
    # --- custom / experimental --- #
    "zailuApprox": zailuApprox,
    "zailuNormal": zailuNormal,
    "squareplus": lambda x: 0.5 * (x + torch.sqrt(x * x + 4)),
    "swish": lambda x: x * torch.sigmoid(x),
}

# --- Synchronization helper ---
def sync_device(device):
    """Ensure GPU/MPS ops complete before timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

# --- Input tensor ---
x = torch.linspace(-10, 10, 1000)
results = []

# --- Benchmarking loop ---
for device in devices:
    x_device = x.to(device)
    for name, func in actfun.items():
        # Warm-up (stabilizes kernels)
        _ = func(x_device)
        sync_device(device)

        start_time = time.time()
        for _ in range(trials):
            _ = func(x_device)
        sync_device(device)

        elapsed = time.time() - start_time
        print(f"{name} ({device}): {elapsed:.6f}s")

        results.append({
            "activation": name,
            "device": str(device),
            "time": elapsed
        })

# --- Create raw DataFrame ---
df = pd.DataFrame(results)
df.to_csv("activation_benchmarks_raw.csv", index=False)
print("\n✅ Saved raw benchmark data to activation_benchmarks_raw.csv")

# --- Format research-style table ---
df["ms"] = df["time"] * 1000
pivot = df.pivot(index="activation", columns="device", values="ms")

# Rename columns nicely
rename_map = {
    "cpu": "CPU",
    "cuda": "GPU (CUDA)",
    "mps": "GPU (MPS)"
}
pivot = pivot.rename(columns=rename_map)

pivot = pivot.round(3).sort_values("CPU")

# Add " ms" suffix to all numeric entries
pivot = pivot.applymap(lambda x: f"{x:.3f} ms" if pd.notna(x) else "")

print("\n=== Formatted Results (ms per 10k runs) ===")
print(pivot)

pivot.to_csv("activation_benchmarks_formatted.csv")
print("\n✅ Saved formatted table to activation_benchmarks_formatted.csv")
