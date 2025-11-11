"""Test a trained RL model and visualize the results."""

import argparse
from pathlib import Path
import numpy as np
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pidog_env import PiDogEnv
from stable_baselines3 import PPO, SAC, TD3


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test trained PiDog model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model (.zip file)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "sac", "td3"],
        help="RL algorithm used",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (no exploration)",
    )
    parser.add_argument(
        "--use-camera",
        action="store_true",
        help="Enable camera observations",
    )
    parser.add_argument(
        "--disable-camera",
        action="store_true",
        help="Disable camera rendering (must match training config!)",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=64,
        help="Camera image width (default: 64, must match training config!)",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=64,
        help="Camera image height (default: 64, must match training config!)",
    )
    return parser.parse_args()


def main():
    """Main testing function."""
    args = parse_args()

    # Handle camera flags
    if args.disable_camera:
        args.use_camera = False

    print("=" * 60)
    print("PiDog RL Model Testing")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Episodes: {args.episodes}")
    print(f"Deterministic: {args.deterministic}")
    if args.use_camera:
        print(f"Camera: Enabled ({args.camera_width}x{args.camera_height})")
    else:
        print(f"Camera: Disabled (zeros)")
    print("=" * 60)

    # Create environment with rendering
    print(f"\nCreating environment with visualization...")

    env = PiDogEnv(
        use_camera=args.use_camera,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )

    # Print actual observation shape
    print(f"Observation format: Box (flattened) - shape {env.observation_space.shape}")

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    if args.algorithm == "ppo":
        model = PPO.load(args.model_path)
    elif args.algorithm == "sac":
        model = SAC.load(args.model_path)
    elif args.algorithm == "td3":
        model = TD3.load(args.model_path)

    print("Model loaded successfully!")

    # Run episodes
    episode_rewards = []
    episode_lengths = []
    episode_velocities = []

    print(f"\nRunning {args.episodes} episodes...\n")

    for episode in range(args.episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        max_velocity = 0

        print(f"Episode {episode + 1}/{args.episodes}")

        # Import viewer for visualization
        import mujoco.viewer

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.distance = 1.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 135

            terminated = False
            truncated = False

            while not (terminated or truncated) and viewer.is_running():
                # Get action from model
                action, _ = model.predict(obs, deterministic=args.deterministic)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                episode_length += 1
                max_velocity = max(max_velocity, info["forward_velocity"])

                # Update viewer
                viewer.sync()
                time.sleep(env.dt * env.frame_skip)

                # Print progress
                if episode_length % 100 == 0:
                    print(f"  Step {episode_length}: "
                          f"vel={info['forward_velocity']:.3f}m/s, "
                          f"height={info['body_height']:.3f}m, "
                          f"reward={reward:.2f}")

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_velocities.append(max_velocity)

        print(f"  Episode finished:")
        print(f"    Total reward: {episode_reward:.2f}")
        print(f"    Steps: {episode_length}")
        print(f"    Max velocity: {max_velocity:.3f} m/s")
        print()

    # Print summary statistics
    print("=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average max velocity: {np.mean(episode_velocities):.3f} ± {np.std(episode_velocities):.3f} m/s")
    print(f"Best velocity: {np.max(episode_velocities):.3f} m/s")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
