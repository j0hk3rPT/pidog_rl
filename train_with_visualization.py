#!/usr/bin/env python3
"""
Train PiDog with real-time visualization checkpoints.

This script shows how to train the robot while periodically visualizing
its progress in real-time. You'll see the robot's behavior at regular
intervals (e.g., every 50K timesteps) so you can monitor learning progress.

Usage:
    python train_with_visualization.py --total-timesteps 500000 --visualize-freq 50000
"""

import argparse
from pathlib import Path
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from pidog_env import PiDogEnv
from training.visualization_callback import VisualizeCallback, PeriodicVisualizeCallback


def make_env(rank, seed=0, use_camera=False):
    """Create a single environment instance."""
    def _init():
        env = PiDogEnv(use_camera=use_camera)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PiDog with real-time visualization"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500000,
        help="Total training timesteps (default: 500000)",
    )
    parser.add_argument(
        "--visualize-freq",
        type=int,
        default=50000,
        help="Visualize every N timesteps (default: 50000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10000,
        help="Save checkpoint every N steps (default: 10000)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=2,
        help="Number of episodes to show during visualization (default: 2)",
    )
    parser.add_argument(
        "--render-fps",
        type=int,
        default=30,
        help="Frames per second for visualization (default: 30)",
    )
    parser.add_argument(
        "--use-camera",
        action="store_true",
        help="Use camera observations (MultiInputPolicy)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    return parser.parse_args()


def main():
    """Main training function with visualization."""
    args = parse_args()

    # Set up experiment directory
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"ppo_viz_{timestamp}"

    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    log_dir = experiment_dir / "logs"

    # Print configuration
    print("="*70)
    print(" TRAINING WITH REAL-TIME VISUALIZATION")
    print("="*70)
    print(f"Experiment: {args.experiment_name}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Visualize every: {args.visualize_freq:,} timesteps")
    print(f"Episodes per visualization: {args.n_eval_episodes}")
    print(f"Render FPS: {args.render_fps}")
    print(f"Parallel envs: {args.n_envs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Camera enabled: {args.use_camera}")
    print(f"Output directory: {experiment_dir}")
    print("="*70)
    print("\nThe training will pause periodically to show you the robot's")
    print("behavior in real-time. You can close the viewer window to")
    print("continue training, or press Esc to skip to the next checkpoint.")
    print("="*70 + "\n")

    # Create vectorized environments
    print(f"Creating {args.n_envs} parallel environments...")
    if args.n_envs > 1:
        env = SubprocVecEnv([
            make_env(i, args.seed, args.use_camera)
            for i in range(args.n_envs)
        ])
    else:
        env = DummyVecEnv([make_env(0, args.seed, args.use_camera)])

    # Create model
    policy_type = "MultiInputPolicy" if args.use_camera else "MlpPolicy"
    print(f"\nCreating PPO model with {policy_type}...")

    model = PPO(
        policy_type,
        env,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=args.seed,
    )

    # Set up callbacks
    # 1. Regular checkpoint saving (without visualization)
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.n_envs,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_model",
        save_replay_buffer=False,
    )

    # 2. Visualization callback (shows progress in real-time)
    visualize_callback = VisualizeCallback(
        visualize_freq=args.visualize_freq,
        n_eval_episodes=args.n_eval_episodes,
        render_fps=args.render_fps,
        max_steps_per_episode=500,
        verbose=1,
    )

    # Combine callbacks
    callback = CallbackList([checkpoint_callback, visualize_callback])

    # Train the model
    print(f"\nStarting training for {args.total_timesteps:,} timesteps...")
    print(f"TensorBoard logs: {log_dir}")
    print(f"Monitor with: tensorboard --logdir={log_dir}\n")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    # Save final model
    final_model_path = experiment_dir / "ppo_final_model"
    model.save(final_model_path)
    print(f"\n\nFinal model saved to: {final_model_path}")

    # Clean up
    env.close()

    print("\n" + "="*70)
    print(" TRAINING COMPLETED!")
    print("="*70)
    print(f"Results saved in: {experiment_dir}")
    print(f"\nTo visualize the final model, run:")
    print(f"  python visualize_training.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
