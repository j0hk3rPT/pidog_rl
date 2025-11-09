#!/usr/bin/env python3
"""Check GPU/ROCm availability and configuration for RL training."""

import sys

print("=" * 60)
print("GPU/ROCm Configuration Check")
print("=" * 60)

# 1. Check PyTorch
print("\n1. PyTorch Configuration:")
print("-" * 60)
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("  ⚠ No CUDA/ROCm devices detected by PyTorch")
        print("  This could mean:")
        print("    - ROCm drivers not installed on host")
        print("    - GPU not passed to container (need --device=/dev/kfd --device=/dev/dri)")
        print("    - PyTorch not built with ROCm support")
except ImportError:
    print("✗ PyTorch not installed")
    sys.exit(1)

# 2. Check ROCm
print("\n2. ROCm Configuration:")
print("-" * 60)
try:
    # Check if torch was built with ROCm
    print(f"  PyTorch built with ROCm: {torch.version.hip is not None}")
    if torch.version.hip:
        print(f"  ROCm/HIP version: {torch.version.hip}")
except:
    print("  Unable to determine ROCm version")

# 3. Test GPU computation
print("\n3. GPU Computation Test:")
print("-" * 60)
if torch.cuda.is_available():
    try:
        # Create tensor on GPU
        device = torch.device('cuda')
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        print(f"✓ Successfully performed matrix multiplication on GPU")
        print(f"  Device: {z.device}")
    except Exception as e:
        print(f"✗ GPU computation failed: {e}")
else:
    print("  ⚠ Skipping GPU test (no GPU available)")

# 4. Check Stable-Baselines3
print("\n4. Stable-Baselines3 Configuration:")
print("-" * 60)
try:
    import stable_baselines3 as sb3
    print(f"✓ Stable-Baselines3 version: {sb3.__version__}")

    # Check default device
    from stable_baselines3.common.utils import get_device
    device = get_device('auto')
    print(f"  Auto-detected device: {device}")

    if 'cuda' in str(device):
        print("  ✓ Will use GPU for training")
    else:
        print("  ⚠ Will use CPU for training")

except ImportError:
    print("✗ Stable-Baselines3 not installed")

# 5. Recommendations
print("\n5. Recommendations:")
print("-" * 60)

if not torch.cuda.is_available():
    print("GPU NOT AVAILABLE - Training will use CPU only")
    print("\nTo enable GPU with ROCm:")
    print("1. Install ROCm drivers on your host system")
    print("2. Update docker-compose.yml to expose GPU devices:")
    print("   ```yaml")
    print("   services:")
    print("     pidog_rl:")
    print("       devices:")
    print("         - /dev/kfd")
    print("         - /dev/dri")
    print("       group_add:")
    print("         - video")
    print("   ```")
    print("3. Restart the container")
    print("\nAlternatively, for NVIDIA GPUs:")
    print("- Use nvidia/cuda base image instead of rocm/pytorch")
    print("- Add runtime: nvidia to docker-compose.yml")
else:
    print("✓ GPU IS AVAILABLE - Training will use GPU acceleration")
    print(f"\nDetected GPU: {torch.cuda.get_device_name(0)}")
    print("Training should be significantly faster than CPU-only")

print("\n" + "=" * 60)
print("Configuration check complete!")
print("=" * 60)
