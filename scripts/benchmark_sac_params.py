#!/usr/bin/env python3
"""
Benchmark different SAC parameter configurations to find optimal settings.

This script tests various combinations of:
- buffer_size: Replay buffer size
- batch_size: Training batch size
- train_freq: Update frequency
- gradient_steps: Gradient steps per update
- learning_starts: Steps before training starts

Goal: Find fastest training with best sample efficiency and results.
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import sys

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))
from pidog_env import PiDogEnv
from pidog_env.feature_extractors import PiDogCombinedExtractor, PiDogFlattenedExtractor

# Try to import sb3-extra-buffers for compression
try:
    from sb3_extra_buffers.compressed import CompressedReplayBuffer, find_buffer_dtypes
    SB3_EXTRA_BUFFERS_AVAILABLE = True
except ImportError:
    SB3_EXTRA_BUFFERS_AVAILABLE = False


class BenchmarkCallback(BaseCallback):
    """Callback to track training metrics."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_elapsed = 0

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        self.timesteps_elapsed = self.num_timesteps

        # Track episode metrics
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info:
                    self.episode_rewards.append(info['r'])
                if 'l' in info:
                    self.episode_lengths.append(info['l'])

        return True

    def get_metrics(self):
        """Get training metrics."""
        elapsed = time.time() - self.start_time if self.start_time else 0

        return {
            'timesteps': self.timesteps_elapsed,
            'elapsed_time': elapsed,
            'steps_per_second': self.timesteps_elapsed / elapsed if elapsed > 0 else 0,
            'mean_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'mean_ep_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
            'num_episodes': len(self.episode_rewards),
        }


def make_env(rank, use_camera=False, camera_width=84, camera_height=84, use_dict_obs=True, seed=0):
    """Create environment."""
    def _init():
        env = PiDogEnv(
            use_camera=use_camera,
            camera_width=camera_width,
            camera_height=camera_height,
            use_dict_obs=use_dict_obs
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def estimate_memory_usage(buffer_size, obs_shape_image, obs_shape_vector, action_dim):
    """
    Estimate replay buffer memory usage.

    Args:
        buffer_size: Number of transitions
        obs_shape_image: Image observation shape (H, W, C)
        obs_shape_vector: Vector observation shape (dim,)
        action_dim: Action dimension

    Returns:
        Estimated memory in GB
    """
    # Each transition stores: obs, next_obs, action, reward, done
    # NOTE: DictReplayBuffer does NOT support optimize_memory_usage
    # So we store both obs and next_obs separately (full memory usage)

    image_bytes = np.prod(obs_shape_image) * 1  # uint8
    vector_bytes = obs_shape_vector[0] * 4  # float32
    obs_bytes = image_bytes + vector_bytes
    action_bytes = action_dim * 4  # float32
    reward_bytes = 4  # float32
    done_bytes = 1  # bool

    # DictReplayBuffer: stores obs AND next_obs separately (2x memory)
    bytes_per_transition = (obs_bytes * 2) + action_bytes + reward_bytes + done_bytes

    total_bytes = buffer_size * bytes_per_transition
    total_gb = total_bytes / (1024 ** 3)

    return total_gb


def benchmark_config(config, args):
    """Benchmark a single configuration."""
    print(f"\n{'='*70}")
    print(f"TESTING CONFIG: {config['name']}")
    print(f"{'='*70}")
    print(f"Parameters:")
    for key, value in config.items():
        if key != 'name':
            print(f"  {key:20s}: {value}")

    # Memory estimate
    if args.use_camera:
        mem_estimate = estimate_memory_usage(
            config['buffer_size'],
            (args.camera_height, args.camera_width, 3),
            (31,),
            8
        )
        print(f"\nEstimated buffer memory: {mem_estimate:.2f} GB")

        # Check available memory
        available_mem = psutil.virtual_memory().available / (1024 ** 3)
        print(f"Available system memory: {available_mem:.2f} GB")

        if mem_estimate > available_mem * 0.6:
            print(f"⚠️  WARNING: Buffer may be too large!")
            if not args.skip_large:
                print("Skipping this configuration. Use --skip-large=False to test anyway.")
                return None

    # Determine observation format based on compression
    use_dict_obs = not args.use_compression
    use_compression = args.use_compression and SB3_EXTRA_BUFFERS_AVAILABLE

    if args.use_compression and not SB3_EXTRA_BUFFERS_AVAILABLE:
        print("⚠️  Compression requested but sb3-extra-buffers not available!")
        print("Install with: pip install 'sb3-extra-buffers[fast,extra]'")
        print("Falling back to Dict observations (no compression)")
        use_dict_obs = True
        use_compression = False

    # CRITICAL: Initialize buffer dtypes BEFORE creating environments
    # This is required for proper JIT compilation with sb3-extra-buffers
    buffer_dtypes = None
    if use_compression:
        print(f"Initializing compression ({args.compression_method}) - this must happen BEFORE creating environments...")
        obs_size = args.camera_height * args.camera_width * 3 + 31  # 21,199 for 84x84x3 + 31 sensors
        buffer_dtypes = find_buffer_dtypes(
            obs_shape=(obs_size,),
            elem_dtype=np.float32,
            compression_method=args.compression_method
        )
        print(f"✓ Compression initialized (obs_shape={obs_size})")

    # Create vectorized environments AFTER initializing compression
    print(f"Creating {args.n_envs} parallel environment(s)...")
    if args.n_envs > 1:
        env = SubprocVecEnv([
            make_env(i, args.use_camera, args.camera_width, args.camera_height, use_dict_obs, args.seed)
            for i in range(args.n_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(0, args.use_camera, args.camera_width, args.camera_height, use_dict_obs, args.seed)
        ])

    # Policy and feature extractor configuration
    if use_compression:
        policy_type = "CnnPolicy"
        extractor_class = PiDogFlattenedExtractor
        extractor_kwargs = {
            "features_dim": args.features_dim,
            "camera_width": args.camera_width,
            "camera_height": args.camera_height,
        }
        print(f"Using compressed buffer with Box observations")
    else:
        policy_type = "MultiInputPolicy"
        extractor_class = PiDogCombinedExtractor
        extractor_kwargs = {"features_dim": args.features_dim}
        print("Using Dict observations (no compression)")

    policy_kwargs = {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": extractor_kwargs,
    }

    # Adjust gradient_steps for multiple environments
    # With n_envs > 1, gradient_steps=-1 performs as many gradient steps as transitions collected
    gradient_steps = config['gradient_steps']
    if args.n_envs > 1 and gradient_steps > 0:
        # Adjust gradient_steps proportionally to number of environments
        gradient_steps = gradient_steps * args.n_envs
        print(f"Adjusted gradient_steps to {gradient_steps} for {args.n_envs} parallel envs")

    # SAC configuration
    sac_kwargs = {
        "policy": policy_type,
        "env": env,
        "policy_kwargs": policy_kwargs,
        "learning_rate": 3e-4,
        "buffer_size": config['buffer_size'],
        "batch_size": config['batch_size'],
        "train_freq": config['train_freq'],
        "gradient_steps": gradient_steps,
        "learning_starts": config['learning_starts'],
        "gamma": 0.99,
        "tau": 0.005,
        "verbose": 1,
        "device": 'cuda' if args.device == 'cuda' else 'cpu',
        "tensorboard_log": None,  # Disable for benchmarking
    }

    # Add compressed buffer if enabled (buffer_dtypes already initialized earlier)
    if use_compression:
        sac_kwargs["replay_buffer_class"] = CompressedReplayBuffer
        sac_kwargs["replay_buffer_kwargs"] = {
            "dtypes": buffer_dtypes,
            "compression_method": args.compression_method,
        }
        print(f"✓ Configured CompressedReplayBuffer with {config['buffer_size']:,} capacity")

    # Create model
    model = SAC(**sac_kwargs)

    # Create callback
    callback = BenchmarkCallback()

    # Train
    print(f"\nTraining for {args.timesteps:,} timesteps...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        env.close()
        return None

    # Get metrics
    metrics = callback.get_metrics()

    # Add config info
    result = {
        'config': config,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
    }

    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS FOR: {config['name']}")
    print(f"{'='*70}")
    print(f"Training speed:      {metrics['steps_per_second']:.1f} steps/sec")
    print(f"Total time:          {metrics['elapsed_time']:.1f} seconds")
    print(f"Episodes completed:  {metrics['num_episodes']}")
    print(f"Mean episode reward: {metrics['mean_reward']:.2f}")
    print(f"Mean episode length: {metrics['mean_ep_length']:.1f}")

    env.close()

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark SAC parameters")
    parser.add_argument('--timesteps', type=int, default=50_000,
                        help='Timesteps per benchmark (default: 50,000)')
    parser.add_argument('--n-envs', type=int, default=1,
                        help='Number of parallel environments (default: 1)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--use-camera', action='store_true',
                        help='Use camera observations (slower, more memory)')
    parser.add_argument('--camera-width', type=int, default=84)
    parser.add_argument('--camera-height', type=int, default=84)
    parser.add_argument('--features-dim', type=int, default=256)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output JSON file')
    parser.add_argument('--skip-large', type=bool, default=True,
                        help='Skip configs that may exceed memory')
    parser.add_argument('--configs', type=str, nargs='+',
                        choices=['robotics-optimal', 'recommended', 'conservative', 'sb3-default',
                                'fast-training', 'sample-efficient', 'all'],
                        default=['all'],
                        help='Which configs to test')
    parser.add_argument('--use-compression', action='store_true',
                        help='Use compressed replay buffer (requires sb3-extra-buffers)')
    parser.add_argument('--compression-method', type=str, default='zstd-3',
                        help='Compression method (default: zstd-3)')
    args = parser.parse_args()

    print(f"{'='*70}")
    print("SAC PARAMETER BENCHMARK")
    print(f"{'='*70}")
    print(f"Timesteps per test: {args.timesteps:,}")
    print(f"Parallel envs:      {args.n_envs}")
    print(f"Random seed:        {args.seed}")
    print(f"Camera enabled:     {args.use_camera}")
    print(f"Compression:        {args.use_compression}")
    if args.use_compression:
        print(f"Compression method: {args.compression_method}")
        if not SB3_EXTRA_BUFFERS_AVAILABLE:
            print(f"⚠️  WARNING: sb3-extra-buffers not available, compression will be disabled")
    print(f"Device:             {args.device}")
    print(f"Output file:        {args.output}")

    # Available memory
    mem_info = psutil.virtual_memory()
    print(f"\nSystem Memory:")
    print(f"  Total:     {mem_info.total / (1024**3):.1f} GB")
    print(f"  Available: {mem_info.available / (1024**3):.1f} GB")

    # Define configurations to test
    # Note: Buffer sizes based on research for robotics continuous control tasks
    # Research shows smaller buffers (50K-100K) + larger batches (512) work BETTER for robotics
    # than standard 1M buffer defaults. Testing range: 50K-1M to validate.
    # With compression: 50K=~100MB, 100K=~200MB, 200K=~400MB, 500K=~1GB, 1M=~2GB
    #
    # Multiple environments: gradient_steps will be automatically scaled by n_envs
    # e.g., gradient_steps=8 with n_envs=4 becomes 32 gradient steps per update
    all_configs = {
        'robotics-optimal': {
            'name': 'Robotics Optimal',
            'buffer_size': 50_000,  # Research: best for robotics
            'batch_size': 512,       # Research: larger batches for robotics
            'train_freq': 1,
            'gradient_steps': 8,     # Research recommended
            'learning_starts': 1000,
            'description': 'Research-based optimal for robotics (50K buffer, 512 batch)'
        },
        'recommended': {
            'name': 'Recommended',
            'buffer_size': 100_000,  # Research: good balance
            'batch_size': 256,
            'train_freq': 1,
            'gradient_steps': 4,
            'learning_starts': 1000,
            'description': 'Recommended settings (100K buffer, balanced params)'
        },
        'conservative': {
            'name': 'Conservative',
            'buffer_size': 200_000,  # Conservative middle ground
            'batch_size': 256,
            'train_freq': 1,
            'gradient_steps': 1,     # SB3 default
            'learning_starts': 1000,
            'description': 'Conservative middle ground (200K buffer, default gradient_steps)'
        },
        'sb3-default': {
            'name': 'SB3 Default',
            'buffer_size': 1_000_000,  # Official SB3 default
            'batch_size': 256,
            'train_freq': 1,
            'gradient_steps': 1,
            'learning_starts': 100,   # SB3 default
            'description': 'Stable-Baselines3 defaults (1M buffer, baseline comparison)'
        },
        'fast-training': {
            'name': 'Fast Training',
            'buffer_size': 100_000,
            'batch_size': 512,
            'train_freq': 2,         # More frequent updates
            'gradient_steps': 8,
            'learning_starts': 500,  # Start training earlier
            'description': 'Optimized for training speed (frequent updates, large batches)'
        },
        'sample-efficient': {
            'name': 'Sample Efficient',
            'buffer_size': 500_000,  # Larger buffer for diversity
            'batch_size': 256,
            'train_freq': 1,
            'gradient_steps': 4,
            'learning_starts': 5000, # More exploration before training
            'description': 'Optimized for sample efficiency (larger buffer, delayed training)'
        },
    }

    # Select configs
    if 'all' in args.configs:
        configs_to_test = list(all_configs.values())
    else:
        configs_to_test = [all_configs[name] for name in args.configs]

    print(f"\nTesting {len(configs_to_test)} configurations:")
    for config in configs_to_test:
        print(f"  - {config['name']}: {config['description']}")

    # Run benchmarks
    results = []
    for i, config in enumerate(configs_to_test, 1):
        print(f"\n\n{'#'*70}")
        print(f"# CONFIG {i}/{len(configs_to_test)}")
        print(f"{'#'*70}")

        result = benchmark_config(config, args)
        if result is not None:
            results.append(result)

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_path}")

    # Print summary comparison
    if results:
        print(f"\n{'='*70}")
        print("SUMMARY COMPARISON")
        print(f"{'='*70}")
        print(f"{'Config':<25} {'Speed (steps/s)':<18} {'Mean Reward':<15} {'Episodes':<12}")
        print('-' * 70)

        for result in sorted(results, key=lambda r: r['metrics']['steps_per_second'], reverse=True):
            config_name = result['config']['name']
            speed = result['metrics']['steps_per_second']
            reward = result['metrics']['mean_reward']
            episodes = result['metrics']['num_episodes']
            print(f"{config_name:<25} {speed:>15.1f}   {reward:>13.2f}   {episodes:>10}")

        # Best config
        best_speed = max(results, key=lambda r: r['metrics']['steps_per_second'])
        best_reward = max(results, key=lambda r: r['metrics']['mean_reward'])

        print(f"\n🏆 Fastest training:  {best_speed['config']['name']} "
              f"({best_speed['metrics']['steps_per_second']:.1f} steps/s)")
        print(f"🏆 Best performance:  {best_reward['config']['name']} "
              f"({best_reward['metrics']['mean_reward']:.2f} mean reward)")


if __name__ == '__main__':
    main()
