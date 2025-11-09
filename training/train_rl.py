"""Train PiDog using Reinforcement Learning (PPO from Stable-Baselines3)."""

import argparse
import os
from pathlib import Path
from datetime import datetime

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import torch

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pidog_env import PiDogEnv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PiDog with RL")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "sac", "td3"],
        help="RL algorithm to use",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of steps per update (for PPO)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10_000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5_000,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of episodes for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu/cuda/auto)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for logs and checkpoints",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)",
    )
    return parser.parse_args()


def make_env(rank, seed=0):
    """Create a single environment instance."""
    def _init():
        env = PiDogEnv()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def create_algorithm(algorithm_name, env, args):
    """Create RL algorithm instance."""
    common_kwargs = {
        "env": env,
        "learning_rate": args.learning_rate,
        "verbose": 1,
        "device": args.device,
        "tensorboard_log": args.log_dir,
        "seed": args.seed,
    }

    if algorithm_name == "ppo":
        return PPO(
            "MlpPolicy",
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            **common_kwargs,
        )
    elif algorithm_name == "sac":
        return SAC(
            "MlpPolicy",
            batch_size=args.batch_size,
            gamma=0.99,
            tau=0.005,
            learning_starts=1000,
            **common_kwargs,
        )
    elif algorithm_name == "td3":
        return TD3(
            "MlpPolicy",
            batch_size=args.batch_size,
            gamma=0.99,
            tau=0.005,
            learning_starts=1000,
            policy_delay=2,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


def main():
    """Main training function."""
    args = parse_args()

    # Set up experiment directory
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.algorithm}_{timestamp}"

    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    args.log_dir = str(experiment_dir / "logs")
    args.checkpoint_dir = str(experiment_dir / "checkpoints")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 60)
    print(f"Training PiDog with {args.algorithm.upper()}")
    print("=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel envs: {args.n_envs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {experiment_dir}")
    print("=" * 60)

    # Create vectorized environments
    print(f"\nCreating {args.n_envs} parallel environments...")
    if args.n_envs > 1:
        env = SubprocVecEnv([make_env(i, args.seed) for i in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env(0, args.seed)])

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(args.n_envs, args.seed)])

    # Create or load model
    if args.checkpoint is not None:
        print(f"\nLoading checkpoint from {args.checkpoint}...")
        if args.algorithm == "ppo":
            model = PPO.load(args.checkpoint, env=env, device=args.device)
        elif args.algorithm == "sac":
            model = SAC.load(args.checkpoint, env=env, device=args.device)
        elif args.algorithm == "td3":
            model = TD3.load(args.checkpoint, env=env, device=args.device)
    else:
        print(f"\nCreating new {args.algorithm.upper()} model...")
        model = create_algorithm(args.algorithm, env, args)

    # Print model info
    print(f"\nModel architecture:")
    print(model.policy)

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.n_envs,
        save_path=args.checkpoint_dir,
        name_prefix=f"{args.algorithm}_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.checkpoint_dir,
        log_path=args.log_dir,
        eval_freq=args.eval_freq // args.n_envs,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    # Train the model
    print(f"\nStarting training for {args.total_timesteps:,} timesteps...")
    print(f"TensorBoard logs: {args.log_dir}")
    print("Monitor training with: tensorboard --logdir=" + args.log_dir)
    print()

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # Save final model
    final_model_path = experiment_dir / f"{args.algorithm}_final_model"
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Clean up
    env.close()
    eval_env.close()

    print("\nTraining completed!")
    print(f"Results saved in: {experiment_dir}")


if __name__ == "__main__":
    main()
