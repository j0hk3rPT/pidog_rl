#!/usr/bin/env python3
"""
Visualize PiDog walking in MuJoCo simulation.

This script demonstrates the robot walking using either:
1. Sunfounder extracted demonstrations
2. Hardcoded trotting gait
3. A trained policy
"""

import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pidog_env import PiDogEnv
from stable_baselines3 import PPO


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize PiDog walking")
    parser.add_argument(
        "--mode",
        type=str,
        default="sunfounder",
        choices=["sunfounder", "hardcoded", "policy"],
        help="Visualization mode",
    )
    parser.add_argument(
        "--gait",
        type=str,
        default="trot_forward",
        choices=["trot_forward", "trot_backward", "trot_left", "trot_right",
                 "walk_forward", "walk_backward", "walk_left", "walk_right"],
        help="Gait to visualize (for sunfounder mode)",
    )
    parser.add_argument(
        "--demos-path",
        type=str,
        default="datasets/sunfounder_demos.pkl",
        help="Path to Sunfounder demonstrations",
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default=None,
        help="Path to trained policy (for policy mode)",
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=100,
        help="Number of gait cycles to run",
    )
    parser.add_argument(
        "--render-fps",
        type=int,
        default=30,
        help="Rendering frames per second",
    )
    parser.add_argument(
        "--use-camera",
        action="store_true",
        help="Enable camera observations (slower)",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.01,
        help="Delay in seconds between steps (for visualization)",
    )
    return parser.parse_args()


def hardcoded_trot_policy(step):
    """
    Hardcoded trotting gait using sinusoidal patterns.

    Creates a trotting gait where diagonal legs move together.
    Returns normalized actions in [-1, 1].
    """
    t = step * 0.05  # Time parameter
    frequency = 1.5  # Hz
    phase = 2 * np.pi * frequency * t

    # Trotting gait: diagonal pairs move together
    # Leg order: [back_right_hip, back_right_knee,
    #             front_right_hip, front_right_knee,
    #             back_left_hip, back_left_knee,
    #             front_left_hip, front_left_knee]

    # Diagonal pair 1: back-right with front-left (in phase)
    back_right_hip = 0.6 * np.sin(phase)
    back_right_knee = 0.8 * np.cos(phase)
    front_left_hip = 0.6 * np.sin(phase)
    front_left_knee = 0.8 * np.cos(phase)

    # Diagonal pair 2: front-right with back-left (opposite phase)
    front_right_hip = 0.6 * np.sin(phase + np.pi)
    front_right_knee = 0.8 * np.cos(phase + np.pi)
    back_left_hip = 0.6 * np.sin(phase + np.pi)
    back_left_knee = 0.8 * np.cos(phase + np.pi)

    # Return normalized actions [-1, 1]
    action = np.array([
        back_right_hip, back_right_knee,
        front_right_hip, front_right_knee,
        back_left_hip, back_left_knee,
        front_left_hip, front_left_knee,
    ])

    return action


def load_sunfounder_demos(demos_path, gait_name):
    """
    Load Sunfounder demonstration for a specific gait.

    Returns:
        numpy array of actions for the specified gait
    """
    print(f"Loading Sunfounder demonstrations from {demos_path}...")
    with open(demos_path, 'rb') as f:
        dataset = pickle.load(f)

    # Find indices for the specified gait
    actions = dataset['actions']
    labels = dataset['labels']

    # Extract actions for this gait
    gait_indices = [i for i, label in enumerate(labels) if label == gait_name]

    if not gait_indices:
        raise ValueError(f"Gait '{gait_name}' not found in dataset")

    # Get all actions for this gait
    all_gait_actions = actions[gait_indices]

    # Detect cycle length by finding where pattern repeats
    # For trot: 6 steps, for walk: 49 steps
    # We'll take the first occurrence and find repeats
    cycle_length = 6 if 'trot' in gait_name else 49

    # Get just one cycle
    gait_actions = all_gait_actions[:cycle_length]

    print(f"Loaded {len(gait_actions)} steps (one cycle) for {gait_name}")

    return gait_actions


def visualize_sunfounder(env, gait_actions, n_cycles, step_delay=0.01):
    """Visualize using Sunfounder demonstration actions."""
    print(f"\nVisualizing Sunfounder gait for {n_cycles} cycles...")
    print(f"Steps per cycle: {len(gait_actions)}")
    print(f"Step delay: {step_delay}s")

    obs, _ = env.reset()
    total_steps = 0
    total_reward = 0
    cycle_rewards = []
    max_forward_vel = 0

    for cycle in range(n_cycles):
        cycle_start_reward = total_reward
        cycle_steps = 0

        for step_idx, action in enumerate(gait_actions):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1
            cycle_steps += 1

            # Add delay for visualization
            time.sleep(step_delay)

            # Track max velocity
            fwd_vel = info.get('forward_velocity', 0)
            max_forward_vel = max(max_forward_vel, fwd_vel)

            if terminated or truncated:
                print(f"  Cycle {cycle + 1}: Episode ended at step {step_idx} (fell/terminated)")
                obs, _ = env.reset()
                break

        cycle_reward = total_reward - cycle_start_reward
        cycle_rewards.append(cycle_reward)

        # Print progress for each cycle
        avg_reward = total_reward / total_steps
        print(f"Cycle {cycle + 1}/{n_cycles}: "
              f"reward={cycle_reward:.2f}, "
              f"avg_reward={avg_reward:.3f}, "
              f"height={info.get('body_height', 0):.3f}m, "
              f"vel={info.get('forward_velocity', 0):.3f}m/s")

    print(f"\n{'=' * 70}")
    print(f"Visualization Complete!")
    print(f"{'=' * 70}")
    print(f"Total steps: {total_steps}")
    print(f"Average reward: {total_reward / total_steps:.3f}")
    print(f"Max forward velocity: {max_forward_vel:.3f} m/s")
    print(f"Cycle rewards: min={min(cycle_rewards):.2f}, max={max(cycle_rewards):.2f}, avg={np.mean(cycle_rewards):.2f}")
    print(f"{'=' * 70}")

    # Keep window open
    print("\nKeeping viewer window open. Press Ctrl+C to exit...")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nClosing...")


def visualize_hardcoded(env, n_cycles):
    """Visualize using hardcoded trotting gait."""
    print(f"\nVisualizing hardcoded trotting gait for {n_cycles} cycles...")

    obs, _ = env.reset()
    step = 0
    total_reward = 0
    max_forward_vel = 0

    # Estimate steps per cycle (based on frequency)
    steps_per_cycle = int(1.0 / 0.05 / 1.5)  # ~13 steps per cycle
    max_steps = n_cycles * steps_per_cycle

    print(f"Steps per cycle: ~{steps_per_cycle}")
    print(f"Total steps: {max_steps}\n")

    for step in range(max_steps):
        action = hardcoded_trot_policy(step)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Track max velocity
        fwd_vel = info.get('forward_velocity', 0)
        max_forward_vel = max(max_forward_vel, fwd_vel)

        if terminated or truncated:
            print(f"  Episode ended at step {step} (fell/terminated)")
            obs, _ = env.reset()

        # Print progress
        if (step + 1) % steps_per_cycle == 0:
            cycle = (step + 1) // steps_per_cycle
            avg_reward = total_reward / (step + 1)
            print(f"Cycle {cycle}/{n_cycles}: "
                  f"avg_reward={avg_reward:.3f}, "
                  f"height={info.get('body_height', 0):.3f}m, "
                  f"vel={fwd_vel:.3f}m/s")

    print(f"\n{'=' * 70}")
    print(f"Visualization Complete!")
    print(f"{'=' * 70}")
    print(f"Total steps: {step + 1}")
    print(f"Average reward: {total_reward / (step + 1):.3f}")
    print(f"Max forward velocity: {max_forward_vel:.3f} m/s")
    print(f"{'=' * 70}")


def visualize_policy(env, policy_path, n_cycles):
    """Visualize using a trained policy."""
    print(f"\nLoading trained policy from {policy_path}...")
    policy = PPO.load(policy_path)

    print(f"Visualizing trained policy for {n_cycles} cycles...")

    obs, _ = env.reset()
    step = 0
    total_reward = 0
    max_steps = n_cycles * 100  # Approximate

    for step in range(max_steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            obs, _ = env.reset()

        # Print progress
        if (step + 1) % 50 == 0:
            avg_reward = total_reward / (step + 1)
            print(f"Step {step + 1}/{max_steps}, Avg reward: {avg_reward:.3f}")
            print(f"  Forward velocity: {info.get('forward_velocity', 0):.3f} m/s")
            print(f"  Body height: {info.get('body_height', 0):.3f} m")

    print(f"\nVisualization complete!")
    print(f"Total steps: {step + 1}")
    print(f"Average reward: {total_reward / (step + 1):.3f}")


def main():
    """Main visualization function."""
    args = parse_args()

    print("=" * 70)
    print("PiDog Walking Visualization")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    if args.mode == "sunfounder":
        print(f"Gait: {args.gait}")
    print(f"Cycles: {args.n_cycles}")
    print("=" * 70)

    # Create environment with visual rendering
    print("\nCreating environment...")
    print("Launching MuJoCo viewer window...\n")

    env = PiDogEnv(
        render_mode="human",  # Enable interactive viewer
        use_camera=args.use_camera
    )

    try:
        if args.mode == "sunfounder":
            # Load and visualize Sunfounder demonstrations
            demos_path = Path(__file__).parent.parent / args.demos_path
            gait_actions = load_sunfounder_demos(str(demos_path), args.gait)
            visualize_sunfounder(env, gait_actions, args.n_cycles, args.step_delay)

        elif args.mode == "hardcoded":
            # Use hardcoded trotting gait
            visualize_hardcoded(env, args.n_cycles)

        elif args.mode == "policy":
            # Use trained policy
            if not args.policy_path:
                raise ValueError("--policy-path required for policy mode")
            visualize_policy(env, args.policy_path, args.n_cycles)

    finally:
        env.close()


if __name__ == "__main__":
    main()
