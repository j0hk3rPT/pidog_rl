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
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))
from pidog_env import PiDogEnv
from pidog_env.feature_extractors import PiDogCombinedExtractor


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


def make_env(use_camera=False, camera_width=84, camera_height=84):
    """Create environment."""
    def _init():
        env = PiDogEnv(
            use_camera=use_camera,
            camera_width=camera_width,
            camera_height=camera_height
        )
        env = Monitor(env)
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

    # Create environment
    env = DummyVecEnv([make_env(args.use_camera, args.camera_width, args.camera_height)])

    # Policy kwargs
    policy_kwargs = {
        "features_extractor_class": PiDogCombinedExtractor,
        "features_extractor_kwargs": {"features_dim": args.features_dim},
    }

    # Create model
    model = SAC(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        train_freq=config['train_freq'],
        gradient_steps=config['gradient_steps'],
        learning_starts=config['learning_starts'],
        gamma=0.99,
        tau=0.005,
        # NOTE: optimize_memory_usage NOT supported with Dict observations
        # DictReplayBuffer does not support this parameter
        verbose=1,
        device='cuda' if args.device == 'cuda' else 'cpu',
        tensorboard_log=None,  # Disable for benchmarking
    )

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
                        choices=['fast', 'balanced', 'quality', 'memory-efficient', 'aggressive', 'all'],
                        default=['all'],
                        help='Which configs to test')
    args = parser.parse_args()

    print(f"{'='*70}")
    print("SAC PARAMETER BENCHMARK")
    print(f"{'='*70}")
    print(f"Timesteps per test: {args.timesteps:,}")
    print(f"Camera enabled:     {args.use_camera}")
    print(f"Device:             {args.device}")
    print(f"Output file:        {args.output}")

    # Available memory
    mem_info = psutil.virtual_memory()
    print(f"\nSystem Memory:")
    print(f"  Total:     {mem_info.total / (1024**3):.1f} GB")
    print(f"  Available: {mem_info.available / (1024**3):.1f} GB")

    # Define configurations to test
    # Note: Buffer sizes optimized for Dict observations (image+vector) with 30GB RAM
    # DictReplayBuffer stores obs+next_obs separately: ~40 bytes/transition
    # 400K buffer = ~16GB, 500K = ~20GB, 300K = ~12GB
    all_configs = {
        'fast': {
            'name': 'Fast Training',
            'buffer_size': 300_000,  # ~12GB RAM
            'batch_size': 256,
            'train_freq': 8,
            'gradient_steps': 8,
            'learning_starts': 1000,
            'description': 'Optimized for training speed'
        },
        'balanced': {
            'name': 'Balanced',
            'buffer_size': 400_000,  # ~16GB RAM (recommended)
            'batch_size': 128,
            'train_freq': 4,
            'gradient_steps': 4,
            'learning_starts': 1000,
            'description': 'Balance between speed and quality'
        },
        'quality': {
            'name': 'High Quality',
            'buffer_size': 500_000,  # ~20GB RAM (max safe)
            'batch_size': 256,
            'train_freq': 1,
            'gradient_steps': 1,
            'learning_starts': 5000,
            'description': 'Optimized for sample efficiency and stability'
        },
        'memory-efficient': {
            'name': 'Memory Efficient',
            'buffer_size': 200_000,  # ~8GB RAM
            'batch_size': 64,
            'train_freq': 4,
            'gradient_steps': 4,
            'learning_starts': 1000,
            'description': 'Minimal memory footprint (~8GB)'
        },
        'aggressive': {
            'name': 'Aggressive Training',
            'buffer_size': 400_000,  # ~16GB RAM
            'batch_size': 512,
            'train_freq': 16,
            'gradient_steps': 16,
            'learning_starts': 500,
            'description': 'Maximum training speed (may be unstable)'
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
