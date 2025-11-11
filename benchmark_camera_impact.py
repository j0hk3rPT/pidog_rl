#!/usr/bin/env python3
"""
Benchmark the performance impact of camera processing.

Tests:
1. Camera enabled (real image processing)
2. Camera disabled (zeros sent through CNN)
3. No camera observation space (if we implement variable obs size)

Measures:
- Steps per second (throughput)
- Memory usage
- CPU usage
"""

import time
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from pidog_env import PiDogEnv

# Try to import psutil for memory tracking (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Note: psutil not available, memory tracking disabled")


def benchmark_env(use_camera, n_steps=1000, warmup_steps=100):
    """Benchmark environment performance."""
    print(f"\n{'='*70}")
    print(f"Benchmarking: Camera {'ENABLED' if use_camera else 'DISABLED (zeros)'}")
    print(f"{'='*70}")

    # Create environment
    env = PiDogEnv(use_camera=use_camera, camera_width=84, camera_height=84)

    # Get baseline memory if available
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 ** 2)  # MB
    else:
        mem_before = 0

    # Warmup
    print(f"Warming up ({warmup_steps} steps)...")
    obs, _ = env.reset()
    for _ in range(warmup_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    # Benchmark
    print(f"Running benchmark ({n_steps} steps)...")
    cpu_samples = []
    mem_samples = []

    obs, _ = env.reset()
    start_time = time.time()

    for i in range(n_steps):
        # Sample CPU/memory every 100 steps (if available)
        if PSUTIL_AVAILABLE and i % 100 == 0:
            cpu_samples.append(process.cpu_percent())
            mem_samples.append(process.memory_info().rss / (1024 ** 2))

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    end_time = time.time()
    elapsed = end_time - start_time

    # Get final memory if available
    if PSUTIL_AVAILABLE:
        mem_after = process.memory_info().rss / (1024 ** 2)  # MB
    else:
        mem_after = 0

    # Calculate metrics
    steps_per_sec = n_steps / elapsed
    avg_cpu = np.mean(cpu_samples) if cpu_samples else 0
    avg_mem = np.mean(mem_samples) if mem_samples else 0
    mem_increase = mem_after - mem_before

    # Print results
    print(f"\nResults:")
    print(f"  Steps per second:   {steps_per_sec:.2f}")
    print(f"  Total time:         {elapsed:.2f}s")
    if PSUTIL_AVAILABLE:
        print(f"  Average CPU:        {avg_cpu:.1f}%")
        print(f"  Average memory:     {avg_mem:.1f} MB")
        print(f"  Memory increase:    {mem_increase:.1f} MB")
    print(f"  Observation shape:  {obs.shape}")
    print(f"  Obs sample (first 5): {obs[:5]}")
    print(f"  Obs sample (last 5):  {obs[-5:]}")

    env.close()

    return {
        'use_camera': use_camera,
        'steps_per_sec': steps_per_sec,
        'elapsed': elapsed,
        'avg_cpu': avg_cpu,
        'avg_mem_mb': avg_mem,
        'mem_increase_mb': mem_increase,
        'obs_shape': obs.shape,
    }


def main():
    """Run benchmarks."""
    print("="*70)
    print("CAMERA PERFORMANCE IMPACT BENCHMARK")
    print("="*70)
    if PSUTIL_AVAILABLE:
        print(f"Process ID: {psutil.Process().pid}")
        print(f"System memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")

    n_steps = 1000
    warmup = 100

    # Test with camera enabled
    result_with_camera = benchmark_env(use_camera=True, n_steps=n_steps, warmup_steps=warmup)

    # Wait a bit
    time.sleep(2)

    # Test with camera disabled (zeros)
    result_no_camera = benchmark_env(use_camera=False, n_steps=n_steps, warmup_steps=warmup)

    # Compare
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")

    speedup = result_no_camera['steps_per_sec'] / result_with_camera['steps_per_sec']

    print(f"Steps/sec speedup (no camera):  {speedup:.2f}x")

    if PSUTIL_AVAILABLE:
        cpu_reduction = result_with_camera['avg_cpu'] - result_no_camera['avg_cpu']
        mem_reduction = result_with_camera['avg_mem_mb'] - result_no_camera['avg_mem_mb']
        print(f"CPU reduction (no camera):      {cpu_reduction:.1f}%")
        print(f"Memory reduction (no camera):   {mem_reduction:.1f} MB")

    if speedup > 1.5:
        print(f"\n⚠️  Camera processing has SIGNIFICANT impact ({speedup:.2f}x slower)")
        print("    Consider optimizing feature extractor to skip CNN when camera disabled")
    elif speedup > 1.1:
        print(f"\n⚠️  Camera processing has MODERATE impact ({speedup:.2f}x slower)")
        print("    Current zero-padding approach is acceptable but could be optimized")
    else:
        print(f"\n✓ Camera processing has MINIMAL impact ({speedup:.2f}x)")
        print("  Current approach is fine")


if __name__ == "__main__":
    main()
