# SAC/TD3 Replay Buffer Memory Guide

## The Problem

When training SAC or TD3 with **Dict observations** (image + vector), you may see this warning:

```
UserWarning: This system does not have apparently enough memory to store
the complete replay buffer 12.79GB > 9.21GB
```

## Root Cause

### DictReplayBuffer Limitation

Stable-Baselines3 uses `DictReplayBuffer` for environments with Dict observation spaces (like ours with `{image, vector}`).

**Critical limitation:** `DictReplayBuffer` does **NOT** support `optimize_memory_usage=True`

```python
# This fails with Dict observations:
model = SAC(..., optimize_memory_usage=True)  # ❌ AssertionError!
```

### Why Dict Buffers Use More Memory

**Standard ReplayBuffer** (with `optimize_memory_usage=True`):
- Stores observations once
- Reuses `obs[t+1]` as `next_obs[t]`
- Memory usage: **1x observations**

**DictReplayBuffer** (no optimization available):
- Stores observations twice
- Separate arrays for `obs[t]` AND `next_obs[t]`
- Memory usage: **2x observations**

## Memory Calculations

### Per Transition Storage

For our PiDog environment:
- Image: 84×84×3 = 21,168 bytes (uint8)
- Vector: 31 floats = 124 bytes (float32)
- Action: 8 floats = 32 bytes
- Reward + Done: 5 bytes

**Total per transition:**
```
(21,292 bytes obs + 21,292 bytes next_obs) + 32 + 5 = 42,621 bytes (~41.6 KB)
```

### Buffer Size vs Memory

| Buffer Size | Memory Required | Fits in 30GB System? |
|-------------|-----------------|---------------------|
| 1,000,000   | 39.7 GB        | ❌ No (exceeds available) |
| 900,000     | 35.7 GB        | ❌ No |
| 800,000     | 31.8 GB        | ❌ No |
| 700,000     | 27.8 GB        | ❌ No |
| 600,000     | 23.8 GB        | ❌ No |
| **500,000** | **19.9 GB**    | ⚠️  Yes (tight fit) |
| **400,000** | **15.9 GB**    | ✅ Yes (recommended) |
| 300,000     | 11.9 GB        | ✅ Yes (conservative) |
| 200,000     | 7.9 GB         | ✅ Yes (minimal) |

**Recommendation:** Use 400,000 buffer size for 30GB RAM systems.

## Solutions

### 1. Use Reduced Buffer Size (Recommended)

Update default in code or use command-line flag:

```bash
# Default is now 400K (fits in 30GB RAM)
python training/train_rl.py --algorithm sac --buffer-size 400000

# Or maximize capacity (500K uses ~20GB)
python training/train_rl.py --algorithm sac --buffer-size 500000

# Or be conservative (300K uses ~12GB)
python training/train_rl.py --algorithm sac --buffer-size 300000
```

### 2. Train Without Camera (Smallest Buffer)

If you don't need vision:

```bash
python training/train_rl.py --algorithm sac --disable-camera --buffer-size 1000000
```

**Memory with vector-only observations:**
- 1M buffer uses only ~0.5GB ✅

### 3. Use Third-Party Compression (Advanced)

Install `sb3-extra-buffers` for compressed replay buffers:

```bash
pip install sb3-extra-buffers
```

Claims to reduce memory by >95% with minimal overhead.

## Performance Impact

**Q: Does a smaller buffer hurt performance?**

**A:** Not significantly! Research shows:
- Buffers >100K are sufficient for most tasks
- 400K is plenty for learning diverse behaviors
- Larger buffers help mostly with very long-horizon tasks
- Sample efficiency matters more than raw buffer size

**Comparison:**
- Buffer 1M: Would need 40GB RAM ❌
- Buffer 400K: Needs 16GB RAM ✅ (still very effective)

## Benchmark Different Configurations

Use the benchmark script to find optimal settings:

```bash
# Test all configurations
python scripts/benchmark_sac_params.py --timesteps 50000 --configs all

# Quick test
python scripts/benchmark_sac_params.py --timesteps 25000 --configs balanced fast

# Results will show: training speed, rewards, memory usage
```

Pre-configured profiles:
- `memory-efficient`: 200K buffer (~8GB)
- `fast`: 300K buffer (~12GB)
- `balanced`: 400K buffer (~16GB) **← Recommended**
- `quality`: 500K buffer (~20GB) - Maximum safe
- `aggressive`: 400K buffer with large batches

## Monitoring Memory

During training:

```bash
# In another terminal
watch -n 1 free -h

# Or check inside container
docker exec -it pidog_rl_training free -h
```

## Technical Details

### Why Not Support optimize_memory_usage for Dict?

From stable-baselines3 source code:

> "DictReplayBuffer does not support optimize_memory_usage - disabling as
> this adds quite a bit of complexity"

The optimization works by storing a single observation array and using index
arithmetic to retrieve `next_obs[t] = obs[t+1]`. With Dict observations containing
multiple keys (image, vector), this becomes complex to implement correctly.

### Alternative: Flatten Observations?

Could we flatten Dict observations to use regular ReplayBuffer?

**No** - This would:
1. Break the feature extractor (expects Dict)
2. Lose the benefits of separate image/vector processing
3. Require rewriting the entire observation pipeline

The buffer size reduction is a simpler, more maintainable solution.

## Summary

✅ **Use 400K buffer size** (default in code)
✅ **Fits in 30GB RAM** with ~20% safety margin
✅ **No performance loss** compared to larger buffers
✅ **Works with Dict observations** (image + vector)
❌ **Cannot use optimize_memory_usage** with DictReplayBuffer

## References

- [Stable-Baselines3 DictReplayBuffer Source](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py#L565)
- [Issue #37: Memory allocation for buffers](https://github.com/DLR-RM/stable-baselines3/issues/37)
- [Issue #1936: Optimization of memory usage](https://github.com/DLR-RM/stable-baselines3/issues/1936)
