import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import time

# ---------------------------
# Device setup
# ---------------------------

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda"))
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
if torch.backends.mps.is_available():
    devices.append(torch.device("mps"))
    print("✅ MPS available (Apple GPU)")
print("Using devices:", devices)

# ---------------------------
# Pure PyTorch activations (custom)
# ---------------------------

def relu(x):
    return torch.relu(x)

def squareplus(x, b=4.0):
    return 0.5 * (x + torch.sqrt(x * x + b))

def arctan(x, sigma=1.0):
    z = sigma * x
    return 0.5 + (1.0 / torch.pi) * torch.atan(z)

def arctan_approx(x, sigma=1.0):
    z = sigma * x
    return (0.5 + torch.clamp(z, min=0)) / (1.0 + torch.abs(z))

def zailu(x, sigma=1.0):
    return x * arctan(x, sigma)

def zailu_approx(x, sigma=1.0):
    return x * arctan_approx(x, sigma)

# ---------------------------
# Other standard PyTorch activations
# ---------------------------

def swish(x):
    return x * torch.sigmoid(x)  # Pure PyTorch version

_mish = nn.Mish()
mish = _mish if hasattr(F, "mish") else lambda x: x * torch.tanh(F.softplus(x))

actfun = {
    "ReLU": relu,
    "ELU": F.elu,                      # Standard PyTorch ELU
    "Softplus": F.softplus,            # Standard PyTorch Softplus
    "Swish": swish,                    # Pure PyTorch version
    "Squareplus": squareplus,
    "Zailu": zailu,
    "Zailu Approx": zailu_approx,
    "Arctan": arctan,
    "Arctan Approx": arctan_approx,
    "Hardshrink": F.hardshrink,
    "Hardsigmoid": F.hardsigmoid,
    "Hardswish": F.hardswish,
    "Leaky ReLU": F.leaky_relu,
    "ReLU6": F.relu6,
    "SELU": torch.selu,
    "CELU": torch.celu,
    "GELU": F.gelu,
    "Sigmoid": torch.sigmoid,
    "SiLU": F.silu,
    "Mish": mish,
    "Softsign": F.softsign,
    "Tanh": torch.tanh,
    "Tanhshrink": F.tanhshrink,
    "Identity": lambda x: x,
}

# ---------------------------
# Compile with torch.compile
# ---------------------------

compiled_funcs = []
for name, func in actfun.items():
    try:
        compiled_funcs.append((name, torch.compile(func)))
    except Exception as e:
        print(f"⚠️ Skipped {name}: not compilable ({e})")

# ---------------------------
# Benchmark utilities
# ---------------------------

@torch.no_grad()
def bench(f, x, n_iter=50):
    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        y = f(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter

def run_device(device="cpu", N=10_000_000, dtype=torch.float32, iters=50, results=None):
    print(f"\n=== Device: {device} | N={N:,} | dtype={dtype} ===")
    x = torch.randn(N, dtype=dtype, device=device)

    # Warm-up (trigger compile once)
    for _, f in compiled_funcs:
        _ = f(x)

    for name, f in compiled_funcs:
        try:
            t = bench(f, x, n_iter=iters)
            results.append({"activation": name, "device": str(device), "time": t})
            print(f"{name:20s}: {t*1000:8.3f} ms")
        except Exception as e:
            print(f"❌ Failed {name} on {device}: {e}")

# ---------------------------
# Run benchmarks
# ---------------------------

results = []
for device in devices:
    run_device(device, results=results)

# ---------------------------
# Save + format results
# ---------------------------

if not results:
    print("\n❌ No valid results collected.")
    exit()

df = pd.DataFrame(results)
df.to_csv("activation_benchmarks_raw.csv", index=False)
print("\n✅ Saved raw benchmark data to activation_benchmarks_raw.csv")

df["ms"] = df["time"] * 1000
pivot = df.pivot(index="activation", columns="device", values="ms")

rename_map = {
    "cpu": "CPU",
    "cuda": "GPU (CUDA)",
    "mps": "GPU (MPS)"
}
pivot = pivot.rename(columns=rename_map)
pivot = pivot.round(3).sort_values("CPU")
pivot = pivot.applymap(lambda x: f"{x:.3f} ms" if pd.notna(x) else "")

print("\n=== Formatted Results (ms per 50 runs) ===")
print(pivot)

pivot.to_csv("activation_benchmarks_formatted.csv")
print("\n✅ Saved formatted table to activation_benchmarks_formatted.csv")
