#!/usr/bin/env python3
"""
Debug script to compare Sunfounder coordinates with MuJoCo angles.
"""

import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pidog_env import PiDogEnv

# Import the extraction functions
sys.path.insert(0, str(Path(__file__).parent))
from extract_sunfounder_demos import Trot, coord2polar, legs_angle_calculation, degrees_to_normalized


def main():
    print("=" * 80)
    print("DEBUGGING SUNFOUNDER GAIT ANGLES")
    print("=" * 80)

    # Create a trot forward gait
    print("\n1. Creating Sunfounder Trot Forward gait...")
    trot = Trot(fb=Trot.FORWARD, lr=Trot.STRAIGHT)
    coords_sequence = trot.get_coords()
    print(f"   Generated {len(coords_sequence)} coordinate frames")

    # Show first few steps
    print("\n2. First 3 steps - Sunfounder coordinates (Y, Z in mm):")
    print("   Step | LF_y  LF_z  | RF_y  RF_z  | LH_y  LH_z  | RH_y  RH_z")
    print("   " + "-" * 70)
    for i, coords in enumerate(coords_sequence[:3]):
        parts = []
        for coord in coords:
            parts.append(f"{coord[0]:5.1f} {coord[1]:5.1f}")
        print(f"   {i}    | {' | '.join(parts)}")

    # Convert to angles and show the transformation
    print("\n3. First 3 steps - Angle conversion:")
    print("   Step | Joint | Sunfounder | MuJoCo  | Normalized | Scaled")
    print("   " + "-" * 70)

    joint_names = ["RH_hip", "RH_knee", "RF_hip", "RF_knee",
                   "LH_hip", "LH_knee", "LF_hip", "LF_knee"]

    for step_idx in range(3):
        coords = coords_sequence[step_idx]

        # Get Sunfounder angles (before reordering)
        sunfounder_angles = []
        for i, coord in enumerate(coords):
            leg_angle, foot_angle = coord2polar(coord)
            foot_angle = foot_angle - 90
            if i % 2 != 0:  # Right legs
                leg_angle = -leg_angle
                foot_angle = -foot_angle
            sunfounder_angles.extend([leg_angle, foot_angle])

        # Get MuJoCo angles (after reordering)
        mujoco_angles = legs_angle_calculation(coords)

        # Get normalized values
        normalized = [degrees_to_normalized(a) for a in mujoco_angles]
        normalized = [max(-1.0, min(1.0, a)) for a in normalized]

        # Scale to radians (what actually gets sent)
        low, high = -np.pi/2, np.pi
        scaled = [low + (n + 1.0) * 0.5 * (high - low) for n in normalized]

        for joint_idx in range(8):
            sf_idx = [6, 7, 2, 3, 4, 5, 0, 1][joint_idx]  # Reverse mapping
            print(f"   {step_idx}    | {joint_names[joint_idx]:8} | "
                  f"{sunfounder_angles[sf_idx]:6.1f}°    | "
                  f"{mujoco_angles[joint_idx]:6.1f}° | "
                  f"{normalized[joint_idx]:7.3f}    | "
                  f"{scaled[joint_idx]:6.3f}rad")

    # Show neutral standing position
    print("\n4. Expected neutral standing position (from environment):")
    print("   Joint      | Expected Angle | Expected Normalized")
    print("   " + "-" * 50)
    neutral_hip_deg = -30  # -π/6 rad
    neutral_knee_deg = -45  # -π/4 rad

    for i, name in enumerate(joint_names):
        if "hip" in name.lower():
            expected_deg = neutral_hip_deg
            expected_rad = -np.pi/6
        else:
            expected_deg = neutral_knee_deg
            expected_rad = -np.pi/4

        # Convert to normalized
        low, high = -np.pi/2, np.pi
        expected_norm = 2 * (expected_rad - low) / (high - low) - 1

        print(f"   {name:10} | {expected_deg:8.1f}° ({expected_rad:6.3f}rad) | {expected_norm:7.3f}")

    # Load actual demonstrations and test in environment
    print("\n5. Testing first action in environment...")
    demos_path = Path(__file__).parent.parent / "datasets" / "sunfounder_demos.pkl"
    with open(demos_path, 'rb') as f:
        dataset = pickle.load(f)

    # Get trot_forward actions
    actions = dataset['actions']
    labels = dataset['labels']
    gait_indices = [i for i, label in enumerate(labels) if label == 'trot_forward']
    first_action = actions[gait_indices[0]]

    print(f"\n   First trot_forward action (normalized [-1,1]):")
    for i, name in enumerate(joint_names):
        print(f"   {name:10} | {first_action[i]:7.3f}")

    # Create environment and apply action
    env = PiDogEnv(use_camera=False, render_mode=None)
    obs, _ = env.reset()

    print(f"\n   Initial joint positions (rad):")
    initial_qpos = obs[:8]  # First 8 are joint positions
    for i, name in enumerate(joint_names):
        print(f"   {name:10} | {initial_qpos[i]:7.3f}rad ({initial_qpos[i]*180/np.pi:6.1f}°)")

    # Take one step
    obs, reward, terminated, truncated, info = env.step(first_action)

    print(f"\n   After first action - joint positions (rad):")
    after_qpos = obs[:8]
    for i, name in enumerate(joint_names):
        delta = after_qpos[i] - initial_qpos[i]
        print(f"   {name:10} | {after_qpos[i]:7.3f}rad ({after_qpos[i]*180/np.pi:6.1f}°) | delta: {delta:7.3f}rad")

    print(f"\n   Reward: {reward:.3f}")
    print(f"   Height: {info['body_height']:.4f}m (should be ~0.14m)")
    print(f"   Forward vel: {info['forward_velocity']:.4f}m/s")

    env.close()

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
