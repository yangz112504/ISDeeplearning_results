import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import time
import numpy as np

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
    return x * (0.5 + F.relu(x)) / (1 + torch.abs(x))

def zailuNormal(x):
    # return x * (2 * (1/4 + 1/(2 * torch.pi) * torch.arctan(x)))
    return x * (1/2 + 1/torch.pi * torch.arctan(x))

def numpy_softplus(x):
    """Softplus implemented in NumPy, returned as a torch Tensor matching input device/dtype."""
    x_np = x.detach().cpu().numpy()
    # stable softplus: use x for large positive to avoid overflow in exp
    out_np = np.where(x_np > 20, x_np, np.log1p(np.exp(x_np)))
    return torch.from_numpy(out_np).to(device=x.device, dtype=x.dtype)

def numpy_elu(x, alpha=1.0):
    """ELU implemented in NumPy."""
    x_np = x.detach().cpu().numpy()
    out_np = np.where(x_np > 0, x_np, alpha * (np.exp(x_np) - 1.0))
    return torch.from_numpy(out_np).to(device=x.device, dtype=x.dtype)

def numpy_swish(x):
    """Swish (x * sigmoid(x)) implemented in NumPy."""
    x_np = x.detach().cpu().numpy()
    sig = 1.0 / (1.0 + np.exp(-x_np))
    out_np = x_np * sig
    return torch.from_numpy(out_np).to(device=x.device, dtype=x.dtype)

def numpy_relu(x):
    """ReLU implemented in NumPy."""
    x_np = x.detach().cpu().numpy()
    out_np = np.maximum(0.0, x_np)
    return torch.from_numpy(out_np).to(device=x.device, dtype=x.dtype)

def numpy_squareplus(x):
    """Squareplus (0.5 * (x + sqrt(x^2 + 4))) implemented in NumPy."""
    x_np = x.detach().cpu().numpy()
    out_np = 0.5 * (x_np + np.sqrt(x_np * x_np + 4.0))
    return torch.from_numpy(out_np).to(device=x.device, dtype=x.dtype)

# --- Activation functions ---
actfun = {
    "zailuApprox": zailuApprox,
    "zailuNormal": zailuNormal,
    "squareplus": numpy_squareplus,
    "swish": numpy_swish,
    "elu": numpy_elu,
    "hardshrink": F.hardshrink,
    "hardsigmoid": F.hardsigmoid,
    "hardtanh": F.hardtanh,
    "hardswish": F.hardswish,
    "leaky_relu": F.leaky_relu,
    "logsigmoid": F.logsigmoid,
    "prelu": lambda x: F.prelu(x, torch.tensor(0.25, device=x.device)),
    "relu": numpy_relu,
    "relu6": F.relu6,
    "rrelu": F.rrelu,
    "selu": torch.selu,
    "celu": torch.celu,
    "gelu": F.gelu,
    "sigmoid": torch.sigmoid,
    "silu": F.silu,
    "mish": mish,
    "softplus": numpy_softplus,
    "softshrink": F.softshrink,
    "softsign": F.softsign,
    "tanh": torch.tanh,
    "tanhshrink": F.tanhshrink,
    "threshold": lambda x, th=0.5, val=0.0: F.threshold(x, th, val),
    "glu": F.glu,
    "identity": lambda x: x,
    # --- custom / experimental --- #
}

# --- Synchronization helper ---
def sync_device(device):
    """
    Ensures accurate timing by synchronizing GPU/MPS operations.

    PyTorch executes CUDA and MPS operations asynchronously by default, 
    meaning the CPU may continue before the GPU has actually finished 
    computation. Calling this function forces synchronization, ensuring 
    that all queued operations complete before measuring execution time. 
    This makes benchmarks accurate but slightly slower due to blocking.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

# --- Input tensor ---
x = torch.linspace(-10, 10, 1000)
results = []

# --- Benchmarking loop ---
with torch.no_grad(): #Disable gradients for speed
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
