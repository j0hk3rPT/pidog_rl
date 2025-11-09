"""Collect expert demonstrations for imitation learning."""

import argparse
import pickle
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from imitation.data.types import Transitions
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from pidog_env import PiDogEnv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Collect expert demonstrations")
    parser.add_argument(
        "--policy-path",
        type=str,
        default=None,
        help="Path to expert policy (if None, use hardcoded policy)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=20,
        help="Number of episodes to collect",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="datasets/expert_demos.pkl",
        help="Path to save demonstrations",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during collection",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def hardcoded_policy(obs, step):
    """
    Hardcoded walking policy using realistic trotting gait.

    Based on SunFounder SF006FM servo specs:
    - Range: 0-180° (0 to π radians)
    - Neutral position: 90° (π/2 radians)
    - Movement amplitude: ±30° for hip, ±45° for knee

    Creates a trotting gait where diagonal legs move together.
    """
    t = step * 0.05  # Time parameter
    frequency = 1.5  # Hz

    # Neutral position at 90° (π/2)
    neutral = np.pi / 2

    # Movement amplitudes (in radians)
    hip_amplitude = 0.52  # ±30° converted to radians
    knee_amplitude = 0.79  # ±45° converted to radians

    phase = 2 * np.pi * frequency * t

    # Trotting gait: diagonal pairs move together
    # Back-right with front-left (normalized to [-1, 1])
    back_right_hip = 0.6 * np.sin(phase)
    back_right_knee = 0.8 * np.cos(phase)
    front_left_hip = 0.6 * np.sin(phase)
    front_left_knee = 0.8 * np.cos(phase)

    # Front-right with back-left (opposite phase)
    front_right_hip = 0.6 * np.sin(phase + np.pi)
    front_right_knee = 0.8 * np.cos(phase + np.pi)
    back_left_hip = 0.6 * np.sin(phase + np.pi)
    back_left_knee = 0.8 * np.cos(phase + np.pi)

    # Return normalized actions [-1, 1]
    # Order: back_right, front_right, back_left, front_left (each with hip, knee)
    action = np.array([
        back_right_hip, back_right_knee,
        front_right_hip, front_right_knee,
        back_left_hip, back_left_knee,
        front_left_hip, front_left_knee,
    ])

    return action


def collect_demonstrations(env, policy, n_episodes, use_hardcoded=False):
    """Collect demonstrations from policy."""
    obs_list = []
    acts_list = []
    next_obs_list = []
    dones_list = []
    rewards_list = []

    total_steps = 0

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        episode_reward = 0

        while not done and step < 1000:
            # Get action
            if use_hardcoded:
                action = hardcoded_policy(obs, step)
            else:
                action, _ = policy.predict(obs, deterministic=True)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            obs_list.append(obs)
            acts_list.append(action)
            next_obs_list.append(next_obs)
            dones_list.append(done)
            rewards_list.append(reward)

            obs = next_obs
            step += 1
            episode_reward += reward

        total_steps += step
        print(f"Episode {episode + 1}/{n_episodes}: "
              f"{step} steps, reward={episode_reward:.2f}")

    # Create Transitions object
    transitions = Transitions(
        obs=np.array(obs_list),
        acts=np.array(acts_list),
        infos=np.array([{}] * len(obs_list)),
        next_obs=np.array(next_obs_list),
        dones=np.array(dones_list),
    )

    print(f"\nCollected {len(transitions)} transitions from {n_episodes} episodes")
    print(f"Average steps per episode: {total_steps / n_episodes:.1f}")
    print(f"Average reward per episode: {np.sum(rewards_list) / n_episodes:.2f}")

    return transitions


def main():
    """Main collection function."""
    args = parse_args()

    # Create output directory
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Expert Demonstration Collection")
    print("=" * 60)
    if args.policy_path:
        print(f"Using expert policy: {args.policy_path}")
    else:
        print("Using hardcoded walking policy")
    print(f"Episodes: {args.n_episodes}")
    print(f"Output: {args.output_path}")
    print("=" * 60)

    # Create environment
    render_mode = "human" if args.render else None
    env = PiDogEnv(render_mode=render_mode)
    env.reset(seed=args.seed)

    # Load policy if provided
    policy = None
    use_hardcoded = True

    if args.policy_path:
        print(f"\nLoading expert policy from {args.policy_path}...")
        policy = PPO.load(args.policy_path)
        use_hardcoded = False

    # Collect demonstrations
    print(f"\nCollecting {args.n_episodes} episodes...")
    transitions = collect_demonstrations(
        env,
        policy,
        args.n_episodes,
        use_hardcoded=use_hardcoded
    )

    # Save demonstrations
    print(f"\nSaving demonstrations to {args.output_path}...")
    with open(args.output_path, "wb") as f:
        pickle.dump(transitions, f)

    print(f"Demonstrations saved successfully!")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Clean up
    env.close()


if __name__ == "__main__":
    main()
