#!/usr/bin/env python3
"""
Visualize expert demonstrations to verify accuracy.

This script loads demonstration files and plays them back in the environment
so you can see if the extracted gaits look correct.

Usage:
    # Visualize Sunfounder demonstrations
    python visualize_demonstrations.py --demo-file demonstrations/sunfounder_demos.pkl

    # Slower playback for inspection
    python visualize_demonstrations.py --demo-file demonstrations/sunfounder_demos.pkl --fps 10

    # Show specific number of cycles
    python visualize_demonstrations.py --demo-file demonstrations/sunfounder_demos.pkl --n-cycles 5
"""

import argparse
import pickle
import numpy as np
import time
from pathlib import Path
import mujoco
import mujoco.viewer
import sys

sys.path.insert(0, str(Path(__file__).parent))
from pidog_env import PiDogEnv

# Try to import imitation
try:
    from imitation.data.types import Transitions
    IMITATION_AVAILABLE = True
except ImportError:
    IMITATION_AVAILABLE = False


def visualize_demonstrations(demo_file, n_cycles=3, fps=30, use_camera=False):
    """
    Visualize demonstration actions in the environment.

    Args:
        demo_file: Path to demonstration file (pkl)
        n_cycles: Number of demonstration cycles to show
        fps: Frames per second for playback
        use_camera: Whether to use camera observations
    """
    print("\n" + "="*70)
    print(" VISUALIZING EXPERT DEMONSTRATIONS")
    print("="*70)
    print(f"Demo file: {demo_file}")
    print(f"Playback speed: {fps} FPS")
    print(f"Cycles to show: {n_cycles}")
    print("="*70 + "\n")

    # Load demonstrations
    print(f"Loading demonstrations from {demo_file}...")
    with open(demo_file, "rb") as f:
        data = pickle.load(f)

    # Extract actions based on data format
    if isinstance(data, dict):
        actions = data.get("acts", data.get("actions"))
        obs = data.get("obs", data.get("observations"))
        print(f"Loaded dict format with {len(actions)} transitions")
    elif IMITATION_AVAILABLE and isinstance(data, Transitions):
        actions = data.acts
        obs = data.obs
        print(f"Loaded Transitions format with {len(actions)} transitions")
    else:
        print(f"Unknown data format: {type(data)}")
        return

    print(f"Actions shape: {actions.shape}")
    print(f"Action range: [{actions.min():.2f}, {actions.max():.2f}]")
    print(f"Observations shape: {obs.shape}\n")

    # Create environment
    env = PiDogEnv(use_camera=use_camera)
    frame_time = 1.0 / fps

    print("Controls:")
    print("  - Left mouse: Rotate view")
    print("  - Right mouse: Zoom")
    print("  - Middle mouse: Pan")
    print("  - Space: Pause/Resume")
    print("  - Esc: Exit")
    print("\n" + "="*70 + "\n")

    # Calculate how many actions per cycle
    actions_per_cycle = len(actions) // n_cycles if n_cycles > 0 else len(actions)
    total_actions = min(len(actions), actions_per_cycle * n_cycles) if n_cycles > 0 else len(actions)

    print(f"Playing {total_actions} actions ({n_cycles} cycles)")
    print(f"Actions per cycle: ~{actions_per_cycle}\n")

    # Launch viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.sitegroup[0] = True  # Show sensor sites

        obs, _ = env.reset()
        action_count = 0
        cycle_count = 0
        steps_in_cycle = 0

        print(f"Cycle 1/{n_cycles}:")

        while viewer.is_running() and action_count < total_actions:
            step_start = time.time()

            # Get next action from demonstrations
            action = actions[action_count]

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            action_count += 1
            steps_in_cycle += 1

            # Sync viewer
            viewer.sync()

            # Print progress
            if steps_in_cycle % 50 == 0:
                forward_vel = env.data.qvel[0]
                height = env.data.qpos[2]
                print(f"  Step {steps_in_cycle}: vel={forward_vel:.2f}m/s, height={height:.3f}m, reward={reward:.2f}")

            # Check if we should start next cycle
            if steps_in_cycle >= actions_per_cycle:
                cycle_count += 1
                steps_in_cycle = 0
                if cycle_count < n_cycles:
                    print(f"\nCycle {cycle_count + 1}/{n_cycles}:")
                    # Reset environment for next cycle
                    obs, _ = env.reset()
                    time.sleep(0.5)  # Brief pause between cycles

            # Handle episode termination
            if terminated or truncated:
                print(f"  Episode terminated at step {steps_in_cycle}, resetting...")
                obs, _ = env.reset()
                time.sleep(0.3)

            # Control frame rate
            elapsed = time.time() - step_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    env.close()

    print("\n" + "="*70)
    print(" VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nTotal actions played: {action_count}")
    print(f"Cycles completed: {cycle_count}\n")


def analyze_demonstrations(demo_file):
    """
    Analyze demonstration statistics without visualization.

    Args:
        demo_file: Path to demonstration file (pkl)
    """
    print("\n" + "="*70)
    print(" ANALYZING DEMONSTRATIONS")
    print("="*70)

    # Load demonstrations
    with open(demo_file, "rb") as f:
        data = pickle.load(f)

    # Extract data based on format
    if isinstance(data, dict):
        actions = data.get("acts", data.get("actions"))
        obs = data.get("obs", data.get("observations"))
        dones = data.get("dones", np.zeros(len(actions), dtype=bool))
    elif IMITATION_AVAILABLE and isinstance(data, Transitions):
        actions = data.acts
        obs = data.obs
        dones = data.dones
    else:
        print(f"Unknown data format: {type(data)}")
        return

    print(f"\nFile: {demo_file}")
    print(f"Format: {type(data).__name__}")
    print(f"\nTransitions: {len(actions)}")
    print(f"Episodes: {dones.sum()}")
    print(f"Avg episode length: {len(actions) / max(dones.sum(), 1):.1f} steps")

    print(f"\nAction Statistics:")
    print(f"  Shape: {actions.shape}")
    print(f"  Range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"  Mean: {actions.mean():.3f}")
    print(f"  Std: {actions.std():.3f}")

    print(f"\nObservation Statistics:")
    print(f"  Shape: {obs.shape}")
    print(f"  Range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  Mean: {obs.mean():.3f}")
    print(f"  Std: {obs.std():.3f}")

    # Joint-wise action statistics
    print(f"\nPer-Joint Action Statistics:")
    joint_names = ["FL_hip", "FL_knee", "FR_hip", "FR_knee",
                   "RL_hip", "RL_knee", "RR_hip", "RR_knee"]
    print(f"{'Joint':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 50)
    for i in range(min(8, actions.shape[1])):
        joint_actions = actions[:, i]
        print(f"{joint_names[i]:<10} {joint_actions.mean():>8.3f} {joint_actions.std():>8.3f} "
              f"{joint_actions.min():>8.3f} {joint_actions.max():>8.3f}")

    print("="*70 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Visualize and analyze expert demonstrations"
    )
    parser.add_argument(
        "--demo-file",
        type=str,
        required=True,
        help="Path to demonstration file (pkl)",
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=3,
        help="Number of demonstration cycles to show (default: 3, 0=all)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for playback (default: 30, lower=slower)",
    )
    parser.add_argument(
        "--use-camera",
        action="store_true",
        help="Use camera observations in environment",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze statistics, don't visualize",
    )

    args = parser.parse_args()

    # Check if file exists
    demo_path = Path(args.demo_file)
    if not demo_path.exists():
        print(f"\nError: Demo file not found: {args.demo_file}")
        print(f"\nTo generate Sunfounder demonstrations, run:")
        print(f"  python extract_sunfounder_demos.py --n-cycles 20")
        print()
        return

    if args.analyze_only:
        # Just analyze, don't visualize
        analyze_demonstrations(args.demo_file)
    else:
        # First analyze
        analyze_demonstrations(args.demo_file)

        # Then visualize
        input("\nPress Enter to start visualization...")
        visualize_demonstrations(
            demo_file=args.demo_file,
            n_cycles=args.n_cycles,
            fps=args.fps,
            use_camera=args.use_camera,
        )


if __name__ == "__main__":
    main()
