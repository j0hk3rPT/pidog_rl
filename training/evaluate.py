"""Evaluate a trained PiDog policy."""

import argparse
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
import imageio
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from pidog_env import PiDogEnv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate PiDog policy")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "sac", "td3"],
        help="Algorithm used for training",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during evaluation",
    )
    parser.add_argument(
        "--record-video",
        type=str,
        default=None,
        help="Path to save video recording",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--use-compression",
        action="store_true",
        help="Model was trained with compression (Box obs). Use this if you trained with --use-compression",
    )
    parser.add_argument(
        "--use-camera",
        action="store_true",
        default=True,
        help="Use camera observations",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=84,
        help="Camera width (default: 84)",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=84,
        help="Camera height (default: 84)",
    )
    return parser.parse_args()


def load_model(model_path, algorithm, env):
    """Load trained model."""
    print(f"Loading {algorithm.upper()} model from {model_path}...")

    if algorithm == "ppo":
        model = PPO.load(model_path, env=env)
    elif algorithm == "sac":
        model = SAC.load(model_path, env=env)
    elif algorithm == "td3":
        model = TD3.load(model_path, env=env)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return model


def evaluate(model, env, n_episodes, deterministic=True, record_video=None):
    """Evaluate model on environment."""
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    frames = [] if record_video else None

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_frames = []

        print(f"\nEpisode {episode + 1}/{n_episodes}")

        while not done:
            # Get action
            action, _ = model.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Record frame if needed
            if record_video and hasattr(env, 'render'):
                frame = env.render()
                if frame is not None:
                    episode_frames.append(frame)

            # Print progress
            if episode_length % 100 == 0:
                print(f"  Step {episode_length}: reward={reward:.3f}, "
                      f"forward_vel={info.get('forward_velocity', 0):.3f}, "
                      f"height={info.get('body_height', 0):.3f}")

        # Episode finished
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Check if successful (didn't fall)
        if not terminated or episode_length >= 1000:
            success_count += 1

        if record_video and episode_frames:
            frames.extend(episode_frames)

        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Episode length: {episode_length}")
        print(f"  Success: {not terminated or episode_length >= 1000}")

    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    success_rate = success_count / n_episodes

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean episode length: {mean_length:.0f}")
    print(f"Success rate: {success_rate * 100:.1f}%")
    print("=" * 60)

    # Save video
    if record_video and frames:
        print(f"\nSaving video to {record_video}...")
        imageio.mimsave(record_video, frames, fps=30)
        print(f"Video saved ({len(frames)} frames)")

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "success_rate": success_rate,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


def main():
    """Main evaluation function."""
    args = parse_args()

    print("=" * 60)
    print("PiDog Policy Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Render: {args.render}")
    print(f"Camera: {args.use_camera} ({args.camera_width}x{args.camera_height})")
    print(f"Compression: {args.use_compression}")
    if args.record_video:
        print(f"Recording to: {args.record_video}")
    print("=" * 60)

    # Create environment
    # Note: If model was trained with compression, use Box observations (flattened)
    use_dict_obs = not args.use_compression
    render_mode = "human" if args.render else ("rgb_array" if args.record_video else None)
    env = PiDogEnv(
        render_mode=render_mode,
        use_camera=args.use_camera,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        use_dict_obs=use_dict_obs
    )

    # Load model
    model = load_model(args.model_path, args.algorithm, env)

    # Evaluate
    results = evaluate(
        model,
        env,
        args.n_episodes,
        deterministic=args.deterministic,
        record_video=args.record_video,
    )

    # Clean up
    env.close()

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
