#!/usr/bin/env python3
"""
Clean implementation based on working ROS2/Gazebo inverse kinematics.

This matches the pidog_ros2 repository exactly:
- LEG = 42mm, FOOT = 76mm
- Neutral position: y=0, z=80
- IK: alpha = angle2 + angle1, foot_angle = beta - π/2
- Mirror right legs
- Order: [FL, FR, BL, BR] → [BR, FR, BL, FL]
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
    """
    Inverse kinematics matching ROS2 implementation exactly.
    From pidog_ros2/pidog_gaits/pidog_gaits/inverse_kinematics.py
    """

    # Physical constants (mm) - from SunFounder
    LEG = 42.0   # Upper leg length
    FOOT = 76.0  # Lower leg length

    @classmethod
    def coord2angles(cls, y, z):
        """
        Convert [y, z] coordinate to [hip_angle, knee_angle].

        Exactly matches ROS2 LegIK.coord2angles()

        Args:
            y: Horizontal distance from hip (mm)
            z: Vertical distance from hip (mm)

        Returns:
            (hip_angle, knee_angle) in radians
        """
        # Calculate distance from hip to foot
        u = sqrt(y**2 + z**2)

        # Law of cosines for knee angle (beta)
        cos_angle1 = (cls.FOOT**2 + cls.LEG**2 - u**2) / (2 * cls.FOOT * cls.LEG)
        cos_angle1 = max(min(cos_angle1, 1.0), -1.0)  # Clamp to [-1, 1]
        beta = acos(cos_angle1)

        # Calculate hip angle (alpha)
        angle1 = atan2(y, z)
        cos_angle2 = (cls.LEG**2 + u**2 - cls.FOOT**2) / (2 * cls.LEG * u)
        cos_angle2 = max(min(cos_angle2, 1.0), -1.0)  # Clamp to [-1, 1]
        angle2 = acos(cos_angle2)
        alpha = angle2 + angle1  # NO subtraction (matches ROS2)

        # Knee adjustment (matches ROS2 line 83)
        foot_angle = beta - (pi / 2)

        return alpha, foot_angle

    @classmethod
    def legs_coords_to_angles(cls, leg_coords):
        """
        Convert 4-leg coordinates to 8 joint angles.

        Exactly matches ROS2 LegIK.legs_coords_to_angles()

        Args:
            leg_coords: List of 4 [y, z] coordinates
                       Order: [FL, FR, BL, BR]

        Returns:
            8 angles in MuJoCo actuator order [BR, FR, BL, FL]
        """
        # Generate angles in gait order: FL, FR, BL, BR
        gait_order_angles = []

        for i, (y, z) in enumerate(leg_coords):
            # NO y-coordinate inversion (ROS2 comment line 76)
            hip_angle, knee_angle = cls.coord2angles(y, z)

            # Negate RIGHT legs (odd indices: FR=1, BR=3)
            # Matches ROS2 lines 89-91
            if i % 2 != 0:  # Right legs
                hip_angle = -hip_angle
                knee_angle = -knee_angle

            gait_order_angles.extend([hip_angle, knee_angle])

        # Reorder from [FL, FR, BL, BR] to [BR, FR, BL, FL]
        # Matches ROS2 lines 100-105
        FL_hip, FL_knee = gait_order_angles[0], gait_order_angles[1]
        FR_hip, FR_knee = gait_order_angles[2], gait_order_angles[3]
        BL_hip, BL_knee = gait_order_angles[4], gait_order_angles[5]
        BR_hip, BR_knee = gait_order_angles[6], gait_order_angles[7]

        controller_order_angles = [
            BR_hip, BR_knee,  # BR (from index 3)
            FR_hip, FR_knee,  # FR (from index 1)
            BL_hip, BL_knee,  # BL (from index 2)
            FL_hip, FL_knee,  # FL (from index 0)
        ]

        return controller_order_angles


class WalkGait:
    """
    Walking gait generator.
    From pidog_ros2/pidog_gaits/pidog_gaits/walk_gait.py
    """

    # Parameters from ROS2 walk_gait.py
    SECTION_COUNT = 8
    STEP_COUNT = 6
    LEG_ORDER = [1, 0, 4, 0, 2, 0, 3, 0]
    LEG_STEP_HEIGHT = 20  # mm
    LEG_STEP_WIDTH = 80   # mm
    CENTER_OF_GRAVITY = -15  # mm
    LEG_POSITION_OFFSETS = [-10, -10, 20, 20]
    Z_ORIGIN = 80  # Standing height in mm

    TURNING_RATE = 0.3
    LEG_STEP_SCALES = [
        [TURNING_RATE, 1, TURNING_RATE, 1],  # Left turn
        [1, 1, 1, 1],                         # Straight
        [1, TURNING_RATE, 1, TURNING_RATE]    # Right turn
    ]
    LEG_ORIGINAL_Y_TABLE = [0, 2, 3, 1]

    def __init__(self, fb=1, lr=0):
        """
        Args:
            fb: Forward (1) or Backward (-1)
            lr: Left (-1), Straight (0), or Right (1)
        """
        self.fb = fb
        self.lr = lr

        # Calculate parameters
        self.y_offset = 0 + self.CENTER_OF_GRAVITY
        self.leg_step_width = [
            self.LEG_STEP_WIDTH * self.LEG_STEP_SCALES[self.lr + 1][i]
            for i in range(4)
        ]
        self.section_length = [
            self.leg_step_width[i] / (self.SECTION_COUNT - 1)
            for i in range(4)
        ]
        self.step_down_length = [
            self.section_length[i] / self.STEP_COUNT
            for i in range(4)
        ]
        self.leg_origin = [
            self.leg_step_width[i] / 2 + self.y_offset +
            (self.LEG_POSITION_OFFSETS[i] * self.LEG_STEP_SCALES[self.lr + 1][i])
            for i in range(4)
        ]

    def get_coords(self):
        """
        Generate walk cycle coordinates.
        Returns list of [[y1,z1], [y2,z2], [y3,z3], [y4,z4]] for each step.
        """
        # Starting position
        origin_leg_coord = [
            [
                self.leg_origin[i] - self.LEG_ORIGINAL_Y_TABLE[i] * 2 * self.section_length[i],
                self.Z_ORIGIN
            ]
            for i in range(4)
        ]

        leg_coord = list(origin_leg_coord)
        all_coords = []

        for section in range(self.SECTION_COUNT):
            for step in range(self.STEP_COUNT):
                # Determine which leg is lifting
                if self.fb == 1:
                    raise_leg = self.LEG_ORDER[section]
                else:
                    raise_leg = self.LEG_ORDER[self.SECTION_COUNT - section - 1]

                # Update all 4 legs
                for i in range(4):
                    if raise_leg != 0 and i == raise_leg - 1:
                        # This leg is lifting
                        theta = step * pi / (self.STEP_COUNT - 1)
                        temp = (self.leg_step_width[i] * (cos(theta) - self.fb) / 2 * self.fb)
                        y = self.leg_origin[i] + temp
                        z = self.Z_ORIGIN - (self.LEG_STEP_HEIGHT * step / (self.STEP_COUNT - 1))
                    else:
                        # Other legs slide on ground
                        y = leg_coord[i][0] + self.step_down_length[i] * self.fb
                        z = self.Z_ORIGIN

                    leg_coord[i] = [y, z]

                all_coords.append([list(coord) for coord in leg_coord])

        return all_coords


def collect_clean_demonstrations(n_cycles=10, use_camera=False):
    """Collect demonstrations using clean ROS2-matched implementation."""
    print("\n" + "="*70)
    print(" CLEAN IMPLEMENTATION FROM ROS2")
    print("="*70)
    print(f"Cycles: {n_cycles}")
    print(f"LEG: {PiDogIK.LEG}mm, FOOT: {PiDogIK.FOOT}mm")
    print(f"Standing height: {WalkGait.Z_ORIGIN}mm")
    print(f"Neutral position: y=0, z={WalkGait.Z_ORIGIN}")
    print("="*70 + "\n")

    env = PiDogEnv(use_camera=use_camera)

    # MuJoCo control range (now fixed in pidog_actuators.xml)
    ctrl_range_low = -np.pi/2
    ctrl_range_high = np.pi

    obs_list = []
    acts_list = []
    next_obs_list = []
    dones_list = []

    total_transitions = 0

    for cycle in range(n_cycles):
        # Generate walk gait
        walk = WalkGait(fb=1, lr=0)  # Forward, straight
        coords_sequence = walk.get_coords()

        obs, _ = env.reset()

        for coords in coords_sequence:
            # Convert coordinates to joint angles
            angles = PiDogIK.legs_coords_to_angles(coords)

            # Normalize to [-1, 1] for MuJoCo
            action = []
            for angle in angles:
                # Map from control range to [-1, 1]
                normalized = (angle - ctrl_range_low) / (ctrl_range_high - ctrl_range_low) * 2 - 1
                normalized = np.clip(normalized, -1, 1)
                action.append(normalized)

            action = np.array(action, dtype=np.float32)

            # Execute action
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
            total_transitions += 1

            if terminated or truncated:
                obs, _ = env.reset()
                break

        if (cycle + 1) % 5 == 0:
            print(f"  Cycle {cycle + 1}/{n_cycles}: {total_transitions} transitions")

    env.close()

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

    print(f"\n✓ Collected {len(transitions)} transitions")
    print("="*70 + "\n")
    return transitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", type=str, default="demonstrations/clean_from_ros2.pkl")
    parser.add_argument("--n-cycles", type=int, default=20)
    parser.add_argument("--use-camera", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    demos = collect_clean_demonstrations(
        n_cycles=args.n_cycles,
        use_camera=args.use_camera
    )

    with open(output_path, "wb") as f:
        pickle.dump(demos, f)

    print(f"✓ Saved to: {output_path}\n")


if __name__ == "__main__":
    main()
