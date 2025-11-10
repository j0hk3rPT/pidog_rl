#!/usr/bin/env python3
"""
Benchmark different PPO hyperparameter configurations.

This script tests various combinations of:
- n_steps: Rollout buffer size per environment
- batch_size: Minibatch size for SGD updates
- n_epochs: Number of epochs per rollout
- gae_lambda: GAE parameter for advantage estimation
- clip_range: PPO clipping parameter
- ent_coef: Entropy coefficient for exploration

Goal: Find optimal settings for quadruped control with camera + sensors.
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import sys

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))
from pidog_env import PiDogEnv
from pidog_env.feature_extractors import PiDogCombinedExtractor, PiDogFlattenedExtractor

# Try to import sb3-extra-buffers for compression
try:
    from sb3_extra_buffers.compressed import CompressedRolloutBuffer, find_buffer_dtypes
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

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        self.timesteps_elapsed += 1

        # Track episodes
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.episode_rewards.append(info['r'])
                self.episode_lengths.append(info['l'])

        return True

    def get_metrics(self):
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        return {
            'total_timesteps': self.timesteps_elapsed,
            'elapsed_time': elapsed_time,
            'fps': self.timesteps_elapsed / elapsed_time if elapsed_time > 0 else 0,
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
            'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
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


def run_benchmark(config, args):
    """Run a single benchmark with given configuration."""
    print(f"\n{'='*70}")
    print(f"Testing: {config['name']}")
    print(f"{'='*70}")
    print(f"Description: {config['description']}")
    print(f"n_steps={config['n_steps']:,}, batch_size={config['batch_size']}, "
          f"n_epochs={config['n_epochs']}")
    print(f"gae_lambda={config['gae_lambda']}, clip_range={config['clip_range']}, "
          f"ent_coef={config['ent_coef']}")

    # Calculate rollout buffer size
    rollout_buffer_size = config['n_steps'] * args.n_envs
    print(f"Rollout buffer size: {rollout_buffer_size:,} ({config['n_steps']}*{args.n_envs})")

    # Validate batch_size is a factor of rollout_buffer_size
    if rollout_buffer_size % config['batch_size'] != 0:
        print(f"⚠️  WARNING: batch_size ({config['batch_size']}) is not a factor of "
              f"rollout_buffer_size ({rollout_buffer_size})")
        print("This may cause issues. Adjusting batch_size...")
        # Find closest factor
        for bs in [64, 128, 256, 512, 1024, 2048]:
            if rollout_buffer_size % bs == 0:
                config['batch_size'] = bs
                print(f"Adjusted batch_size to {bs}")
                break

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

    # PPO configuration
    ppo_kwargs = {
        "policy": policy_type,
        "env": env,
        "policy_kwargs": policy_kwargs,
        "learning_rate": 3e-4,
        "n_steps": config['n_steps'],
        "batch_size": config['batch_size'],
        "n_epochs": config['n_epochs'],
        "gamma": config['gamma'],
        "gae_lambda": config['gae_lambda'],
        "clip_range": config['clip_range'],
        "ent_coef": config['ent_coef'],
        "verbose": 1,
        "device": 'cuda' if args.device == 'cuda' else 'cpu',
        "tensorboard_log": None,  # Disable for benchmarking
    }

    # Add compressed buffer if enabled (buffer_dtypes already initialized earlier)
    if use_compression:
        ppo_kwargs["rollout_buffer_class"] = CompressedRolloutBuffer
        ppo_kwargs["rollout_buffer_kwargs"] = {
            "dtypes": buffer_dtypes,
            "compression_method": args.compression_method,
        }
        print(f"✓ Configured CompressedRolloutBuffer (size={rollout_buffer_size:,})")

    # Create model
    model = PPO(**ppo_kwargs)

    # Setup callback
    callback = BenchmarkCallback()

    # Record memory before training
    mem_before = psutil.virtual_memory()
    process = psutil.Process()
    process_mem_before = process.memory_info().rss / (1024**3)  # GB

    # Train
    print(f"\nStarting training for {args.timesteps:,} timesteps...")
    start_time = time.time()

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=True,
        )
        success = True
        error_msg = None
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        success = False
        error_msg = str(e)

    elapsed_time = time.time() - start_time

    # Record memory after training
    mem_after = psutil.virtual_memory()
    process_mem_after = process.memory_info().rss / (1024**3)  # GB

    # Cleanup
    env.close()

    # Gather results
    metrics = callback.get_metrics() if success else {}

    result = {
        'config': config,
        'success': success,
        'error': error_msg,
        'metrics': metrics,
        'elapsed_time': elapsed_time,
        'fps': args.timesteps / elapsed_time if elapsed_time > 0 else 0,
        'memory': {
            'process_before_gb': process_mem_before,
            'process_after_gb': process_mem_after,
            'process_delta_gb': process_mem_after - process_mem_before,
            'system_available_before_gb': mem_before.available / (1024**3),
            'system_available_after_gb': mem_after.available / (1024**3),
        }
    }

    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    if success:
        print(f"✓ Training completed successfully")
        print(f"Time: {elapsed_time:.1f}s ({result['fps']:.1f} FPS)")
        print(f"Episodes: {metrics['num_episodes']}")
        print(f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Mean length: {metrics['mean_length']:.1f}")
        print(f"Memory: {result['memory']['process_delta_gb']:.2f}GB process delta")
    else:
        print(f"❌ Training failed: {error_msg}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark PPO parameters")
    parser.add_argument('--timesteps', type=int, default=50_000,
                        help='Timesteps per benchmark (default: 50,000)')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Number of parallel environments (default: 4)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--use-camera', action='store_true',
                        help='Use camera observations (slower, more memory)')
    parser.add_argument('--camera-width', type=int, default=84)
    parser.add_argument('--camera-height', type=int, default=84)
    parser.add_argument('--features-dim', type=int, default=256)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--output', type=str, default='benchmark_ppo_results.json',
                        help='Output JSON file')
    parser.add_argument('--configs', type=str, nargs='+',
                        choices=['sb3-default', 'fast', 'quality', 'balanced', 'small-batch', 'large-batch', 'all'],
                        default=['all'],
                        help='Which configs to test')
    parser.add_argument('--use-compression', action='store_true',
                        help='Use compressed rollout buffer (requires sb3-extra-buffers)')
    parser.add_argument('--compression-method', type=str, default='zstd-3',
                        help='Compression method (default: zstd-3)')
    args = parser.parse_args()

    print(f"{'='*70}")
    print("PPO PARAMETER BENCHMARK")
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
    # Based on research: SB3 defaults + variations for robotics
    # Key: n_steps should be 2048 (standard), batch_size should be a factor of (n_steps * n_envs)
    # For n_envs=4: rollout_buffer=8192, good batch_sizes: 64, 128, 256, 512, 1024, 2048
    all_configs = {
        'sb3-default': {
            'name': 'SB3 Default',
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.0,
            'description': 'Stable-Baselines3 default parameters'
        },
        'fast': {
            'name': 'Fast Training',
            'n_steps': 1024,      # Smaller rollout for faster updates
            'batch_size': 128,    # Larger minibatch
            'n_epochs': 4,        # Fewer epochs per rollout
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,     # Small entropy for exploration
            'description': 'Optimized for speed: smaller rollouts, fewer epochs'
        },
        'quality': {
            'name': 'High Quality',
            'n_steps': 2048,      # Standard rollout size
            'batch_size': 64,     # Small minibatch for more updates
            'n_epochs': 20,       # More epochs per rollout
            'gamma': 0.999,       # Higher gamma for long-term planning
            'gae_lambda': 0.98,   # Higher GAE lambda
            'clip_range': 0.2,
            'ent_coef': 0.0,
            'description': 'Optimized for quality: more epochs, higher gamma'
        },
        'balanced': {
            'name': 'Balanced',
            'n_steps': 2048,
            'batch_size': 256,    # Balanced minibatch size
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'description': 'Balanced between speed and quality'
        },
        'small-batch': {
            'name': 'Small Batch',
            'n_steps': 2048,
            'batch_size': 64,     # Smallest batch = more gradient updates
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.0,
            'description': 'Small minibatch for more gradient updates per epoch'
        },
        'large-batch': {
            'name': 'Large Batch',
            'n_steps': 2048,
            'batch_size': 512,    # Larger batch = fewer but bigger updates
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.0,
            'description': 'Large minibatch for faster training'
        },
    }

    # Select configs
    if 'all' in args.configs:
        configs_to_test = list(all_configs.values())
    else:
        configs_to_test = [all_configs[c] for c in args.configs]

    print(f"\nTesting {len(configs_to_test)} configurations...")

    # Run benchmarks
    results = []
    for i, config in enumerate(configs_to_test, 1):
        print(f"\n\n{'#'*70}")
        print(f"# Benchmark {i}/{len(configs_to_test)}")
        print(f"{'#'*70}")

        result = run_benchmark(config, args)
        results.append(result)

        # Save intermediate results
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nIntermediate results saved to {output_path}")

    # Print summary
    print(f"\n\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}\n")

    successful_results = [r for r in results if r['success']]

    if successful_results:
        # Sort by FPS
        by_fps = sorted(successful_results, key=lambda x: x['fps'], reverse=True)
        print("Fastest configurations (FPS):")
        for i, r in enumerate(by_fps[:3], 1):
            print(f"{i}. {r['config']['name']}: {r['fps']:.1f} FPS ({r['elapsed_time']:.1f}s)")

        # Sort by reward
        by_reward = sorted(successful_results, key=lambda x: x['metrics'].get('mean_reward', -np.inf), reverse=True)
        print("\nBest performing configurations (reward):")
        for i, r in enumerate(by_reward[:3], 1):
            reward = r['metrics'].get('mean_reward', 0)
            print(f"{i}. {r['config']['name']}: {reward:.2f} mean reward")

        # Sort by memory
        by_memory = sorted(successful_results, key=lambda x: x['memory']['process_delta_gb'])
        print("\nMost memory-efficient configurations:")
        for i, r in enumerate(by_memory[:3], 1):
            mem = r['memory']['process_delta_gb']
            print(f"{i}. {r['config']['name']}: {mem:.2f}GB")

    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print(f"\n⚠️  {len(failed_results)} configuration(s) failed:")
        for r in failed_results:
            print(f"  - {r['config']['name']}: {r['error']}")

    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
