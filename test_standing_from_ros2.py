#!/usr/bin/env python3
"""
Test neutral standing position from ROS2.

From ROS2 gait_generator_node.py:
    # Standing height: y=0 (neutral), z=80mm
    [0, 80],  # FL: neutral position
    [0, 80],  # FR: neutral position
    [0, 80],  # BL: neutral position
    [0, 80],  # BR: neutral position

This should make the robot stand still in neutral position.
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from math import sqrt, atan2, acos, cos, sin, pi, degrees, radians
import sys

sys.path.insert(0, str(Path(__file__).parent))
from pidog_env import PiDogEnv

try:
    from imitation.data.types import Transitions
    IMITATION_AVAILABLE = True
except ImportError:
    IMITATION_AVAILABLE = False


class PiDogIK:
    """IK from ROS2."""

    LEG = 42.0   # mm
    FOOT = 76.0  # mm

    @classmethod
    def coord2angles(cls, y, z):
        """Convert [y, z] to [hip_angle, knee_angle]."""
        u = sqrt(y**2 + z**2)

        cos_angle1 = (cls.FOOT**2 + cls.LEG**2 - u**2) / (2 * cls.FOOT * cls.LEG)
        cos_angle1 = max(min(cos_angle1, 1.0), -1.0)
        beta = acos(cos_angle1)

        angle1 = atan2(y, z)
        cos_angle2 = (cls.LEG**2 + u**2 - cls.FOOT**2) / (2 * cls.LEG * u)
        cos_angle2 = max(min(cos_angle2, 1.0), -1.0)
        angle2 = acos(cos_angle2)
        alpha = angle2 + angle1

        foot_angle = beta - (pi / 2)

        return alpha, foot_angle

    @classmethod
    def legs_coords_to_angles(cls, leg_coords):
        """
        Convert 4-leg coordinates to 8 joint angles.

        Args:
            leg_coords: [[y1,z1], [y2,z2], [y3,z3], [y4,z4]]
                       Order: [FL, FR, BL, BR]

        Returns:
            8 angles in MuJoCo order [BR, FR, BL, FL]
        """
        gait_order_angles = []

        for i, (y, z) in enumerate(leg_coords):
            hip_angle, knee_angle = cls.coord2angles(y, z)

            # Negate RIGHT legs (FR=1, BR=3)
            if i % 2 != 0:
                hip_angle = -hip_angle
                knee_angle = -knee_angle

            gait_order_angles.extend([hip_angle, knee_angle])

        # Reorder [FL, FR, BL, BR] to [BR, FR, BL, FL]
        FL_hip, FL_knee = gait_order_angles[0], gait_order_angles[1]
        FR_hip, FR_knee = gait_order_angles[2], gait_order_angles[3]
        BL_hip, BL_knee = gait_order_angles[4], gait_order_angles[5]
        BR_hip, BR_knee = gait_order_angles[6], gait_order_angles[7]

        return [BR_hip, BR_knee, FR_hip, FR_knee, BL_hip, BL_knee, FL_hip, FL_knee]


def test_standing_position(n_steps=500, use_camera=False):
    """Test neutral standing position."""
    print("\n" + "="*70)
    print(" TESTING NEUTRAL STANDING POSITION FROM ROS2")
    print("="*70)
    print()
    print("ROS2 neutral position for all 4 legs:")
    print("  y = 0mm  (foot directly below hip)")
    print("  z = 80mm (standing height)")
    print()
    print("="*70)
    print()

    # Neutral position from ROS2: y=0, z=80 for all legs
    neutral_coords = [
        [0, 80],  # FL
        [0, 80],  # FR
        [0, 80],  # BL
        [0, 80],  # BR
    ]

    # Convert to joint angles using ROS2 IK
    angles_rad = PiDogIK.legs_coords_to_angles(neutral_coords)

    print("Calculated joint angles (radians):")
    names = ["BR_hip", "BR_knee", "FR_hip", "FR_knee", "BL_hip", "BL_knee", "FL_hip", "FL_knee"]
    for name, angle in zip(names, angles_rad):
        print(f"  {name:12s}: {angle:7.4f} rad ({degrees(angle):7.2f}°)")
    print()

    # MuJoCo control range
    ctrl_range_low = -np.pi/2
    ctrl_range_high = np.pi

    # Normalize to [-1, 1]
    action = []
    for angle in angles_rad:
        normalized = (angle - ctrl_range_low) / (ctrl_range_high - ctrl_range_low) * 2 - 1
        normalized = np.clip(normalized, -1, 1)
        action.append(normalized)

    action = np.array(action, dtype=np.float32)

    print("Normalized action [-1, 1]:")
    for name, norm in zip(names, action):
        print(f"  {name:12s}: {norm:7.4f}")
    print()

    # Test in environment
    env = PiDogEnv(use_camera=use_camera)
    obs, _ = env.reset()

    obs_list = []
    acts_list = []
    next_obs_list = []
    dones_list = []

    print(f"Holding standing position for {n_steps} steps...")
    print()

    fell = False
    for step in range(n_steps):
        if use_camera and isinstance(obs, dict):
            obs_vec = obs["vector"]
        else:
            obs_vec = obs

        next_obs, reward, terminated, truncated, info = env.step(action)

        if use_camera and isinstance(next_obs, dict):
            next_obs_vec = next_obs["vector"]
        else:
            next_obs_vec = next_obs

        obs_list.append(obs_vec)
        acts_list.append(action)
        next_obs_list.append(next_obs_vec)
        dones_list.append(terminated or truncated)

        obs = next_obs

        if step % 100 == 0:
            print(f"  Step {step}: height={info['body_height']:.3f}m, vel={info['forward_velocity']:.3f}m/s, reward={reward:.2f}")

        if terminated or truncated:
            print(f"\n  ❌ Robot fell at step {step}!")
            fell = True
            break

    env.close()

    if not fell:
        print(f"\n  ✅ Robot stayed standing for all {n_steps} steps!")

    print()
    print("="*70)
    print()

    if not IMITATION_AVAILABLE:
        return {
            "obs": np.array(obs_list),
            "acts": np.array(acts_list),
            "next_obs": np.array(next_obs_list),
            "dones": np.array(dones_list),
        }

    transitions = Transitions(
        obs=np.array(obs_list),
        acts=np.array(acts_list),
        infos=np.array([{}] * len(obs_list)),
        next_obs=np.array(next_obs_list),
        dones=np.array(dones_list),
    )

    return transitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", type=str, default="demonstrations/standing_test.pkl")
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--use-camera", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    demos = test_standing_position(
        n_steps=args.n_steps,
        use_camera=args.use_camera
    )

    with open(output_path, "wb") as f:
        pickle.dump(demos, f)

    print(f"✓ Saved to: {output_path}")
    print(f"\nVisualize with:")
    print(f"  python visualize_demonstrations.py --demo-file {output_path}\n")


if __name__ == "__main__":
    main()
