#!/usr/bin/env python3
"""
Quick test to verify the environment setup is working.
"""

import sys
from pathlib import Path

print("=" * 70)
print("PiDog RL Setup Test")
print("=" * 70)

# Test 1: Import dependencies
print("\n1. Testing imports...")
try:
    import numpy as np
    print("   ✓ NumPy")
except ImportError as e:
    print(f"   ✗ NumPy: {e}")
    sys.exit(1)

try:
    import mujoco
    print(f"   ✓ MuJoCo (version: {mujoco.__version__})")
except ImportError as e:
    print(f"   ✗ MuJoCo: {e}")
    sys.exit(1)

try:
    import gymnasium as gym
    print(f"   ✓ Gymnasium")
except ImportError as e:
    print(f"   ✗ Gymnasium: {e}")
    sys.exit(1)

try:
    from stable_baselines3 import PPO
    print("   ✓ Stable-Baselines3")
except ImportError as e:
    print(f"   ✗ Stable-Baselines3: {e}")
    sys.exit(1)

try:
    from imitation.algorithms import bc
    print("   ✓ Imitation Learning")
except ImportError as e:
    print("   ⚠ Imitation Learning (optional): not installed")

# Test 2: Check environment
print("\n2. Testing PiDog environment...")
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from pidog_env import PiDogEnv
    print("   ✓ PiDogEnv import successful")

    env = PiDogEnv(use_camera=False, render_mode=None)
    print("   ✓ Environment creation successful")

    obs, info = env.reset()
    print(f"   ✓ Environment reset successful (obs shape: {obs.shape})")

    # Test a few steps
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   ✓ Step successful (reward: {reward:.3f})")

    env.close()
    print("   ✓ Environment close successful")

except Exception as e:
    print(f"   ✗ Environment test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check demonstration dataset
print("\n3. Testing demonstration dataset...")
demos_path = Path(__file__).parent.parent / "datasets" / "sunfounder_demos.pkl"

if demos_path.exists():
    print(f"   ✓ Demonstrations found at {demos_path}")

    import pickle
    with open(demos_path, 'rb') as f:
        dataset = pickle.load(f)

    print(f"   ✓ Dataset loaded successfully")
    print(f"     - Total actions: {dataset['n_actions']}")
    print(f"     - Action shape: {dataset['actions'].shape}")
    print(f"     - Action range: [{dataset['actions'].min():.3f}, {dataset['actions'].max():.3f}]")
else:
    print(f"   ⚠ Demonstrations not found at {demos_path}")
    print("     Run: python3 scripts/extract_sunfounder_demos.py")

# Test 4: Check GPU (if available)
print("\n4. Checking GPU availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"     - PyTorch version: {torch.__version__}")
        print(f"     - CUDA version: {torch.version.cuda}")
    else:
        print("   ⚠ No GPU detected (CPU training will be slower)")
except Exception as e:
    print(f"   ⚠ Could not check GPU: {e}")

print("\n" + "=" * 70)
print("Setup Test Complete!")
print("=" * 70)
print("\nYou're ready to:")
print("  1. Visualize gaits:  python3 scripts/visualize_walk.py --mode sunfounder --gait trot_forward --n-cycles 5")
print("  2. Train a model:    python3 scripts/train_from_sunfounder.py --method bc --gait trot_forward --bc-epochs 50")
print("=" * 70)
