#!/usr/bin/env python3
"""
Extract expert demonstrations from original Sunfounder PiDog gaits.

This script extracts the walking, trotting, and other movements from the
official Sunfounder PiDog repository and converts them to demonstrations
that can be used for imitation learning.

The original PiDog uses coordinate-based control (y, z positions for each leg),
which are converted to joint angles using inverse kinematics. We extract these
angle sequences and convert them to our environment's action space.

Usage:
    python extract_sunfounder_demos.py --output-file demonstrations/sunfounder_demos.pkl
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from math import sqrt, atan2, acos, cos, sin, pi
import sys

sys.path.insert(0, str(Path(__file__).parent))
from pidog_env import PiDogEnv
from stable_baselines3.common.vec_env import DummyVecEnv

# Try to import imitation
try:
    from imitation.data.types import Transitions
    IMITATION_AVAILABLE = True
except ImportError:
    IMITATION_AVAILABLE = False
    print("Warning: 'imitation' package not installed")
    print("Install with: pip install imitation\n")


class SunfounderGaitExtractor:
    """Extract gaits from Sunfounder PiDog motion planning."""

    # Leg dimensions (from Sunfounder pidog.py)
    LEG = 62    # Upper leg length (mm)
    FOOT = 62   # Lower leg length (mm)

    def __init__(self):
        """Initialize gait extractor."""
        pass

    @staticmethod
    def coord2angle(y, z):
        """
        Convert (y, z) coordinates to joint angles using inverse kinematics.

        This is adapted from Sunfounder's coord2polar function.

        Args:
            y: Forward/backward position (mm)
            z: Height position (mm)

        Returns:
            (hip_angle, knee_angle) in radians
        """
        LEG = SunfounderGaitExtractor.LEG
        FOOT = SunfounderGaitExtractor.FOOT

        # Distance from hip to foot
        u = sqrt(y**2 + z**2)

        # Knee angle (beta) using law of cosines
        cos_beta = (FOOT**2 + LEG**2 - u**2) / (2 * FOOT * LEG)
        cos_beta = max(-1, min(1, cos_beta))  # Clamp to [-1, 1]
        beta = acos(cos_beta)

        # Hip angle (alpha)
        angle1 = atan2(y, z)
        cos_angle2 = (LEG**2 + u**2 - FOOT**2) / (2 * LEG * u)
        cos_angle2 = max(-1, min(1, cos_angle2))
        angle2 = acos(cos_angle2)
        alpha = angle2 + angle1

        # Convert to degrees for easier understanding
        alpha_deg = alpha * 180 / pi
        beta_deg = beta * 180 / pi

        # Adjust beta (knee) - subtract 90 as in original code
        beta_deg -= 90

        # Convert back to radians for our environment
        hip_angle = alpha_deg * pi / 180
        knee_angle = beta_deg * pi / 180

        return hip_angle, knee_angle

    @staticmethod
    def get_walk_gait(fb=1, lr=0):
        """
        Generate walking gait coordinates.

        Args:
            fb: Forward (1) or Backward (-1)
            lr: Left (-1), Straight (0), or Right (1)

        Returns:
            List of angle sequences for all 4 legs
        """
        # Walk parameters (from Sunfounder walk.py)
        SECTION_COUNT = 8
        STEP_COUNT = 6
        LEG_ORDER = [1, 0, 4, 0, 2, 0, 3, 0]  # Which leg moves in each section
        LEG_STEP_HEIGHT = 20
        LEG_STEP_WIDTH = 80
        CENTER_OF_GRAVITY = -15
        LEG_POSITION_OFFSETS = [-10, -10, 20, 20]
        Z_ORIGIN = 80

        TURNING_RATE = 0.3
        LEG_STEP_SCALES_LEFT = [TURNING_RATE, 1, TURNING_RATE, 1]
        LEG_STEP_SCALES_MIDDLE = [1, 1, 1, 1]
        LEG_STEP_SCALES_RIGHT = [1, TURNING_RATE, 1, TURNING_RATE]
        LEG_STEP_SCALES = [LEG_STEP_SCALES_LEFT, LEG_STEP_SCALES_MIDDLE, LEG_STEP_SCALES_RIGHT]
        LEG_ORIGINAL_Y_TABLE = [0, 2, 3, 1]

        y_offset = 0 + CENTER_OF_GRAVITY
        leg_step_width = [LEG_STEP_WIDTH * LEG_STEP_SCALES[lr+1][i] for i in range(4)]
        section_length = [leg_step_width[i] / (SECTION_COUNT-1) for i in range(4)]
        step_down_length = [section_length[i] / STEP_COUNT for i in range(4)]
        leg_origin = [leg_step_width[i] / 2 + y_offset + (LEG_POSITION_OFFSETS[i] * LEG_STEP_SCALES[lr+1][i]) for i in range(4)]

        # Generate coordinates for each section and step
        origin_leg_coord = [[leg_origin[i] - LEG_ORIGINAL_Y_TABLE[i] * 2 * section_length[i], Z_ORIGIN] for i in range(4)]
        leg_coord = list.copy(origin_leg_coord)
        all_coords = []

        for section in range(SECTION_COUNT):
            for step in range(STEP_COUNT):
                if fb == 1:
                    raise_leg = LEG_ORDER[section]
                else:
                    raise_leg = LEG_ORDER[SECTION_COUNT - section - 1]

                for i in range(4):
                    if raise_leg != 0 and i == raise_leg-1:
                        # Stepping leg - use cosine for smooth motion
                        theta = step * pi / (STEP_COUNT-1)
                        temp = (leg_step_width[i] * (cos(theta) - fb) / 2 * fb)
                        y = leg_origin[i] + temp
                        z = Z_ORIGIN - (LEG_STEP_HEIGHT * step / (STEP_COUNT-1))
                    else:
                        # Standing leg - moves backward as body moves forward
                        y = leg_coord[i][0] + step_down_length[i] * fb
                        z = Z_ORIGIN

                    leg_coord[i] = [y, z]

                all_coords.append(list.copy(leg_coord))

        return all_coords

    @staticmethod
    def get_trot_gait(fb=1, lr=0):
        """
        Generate trotting gait coordinates (faster than walk).

        Args:
            fb: Forward (1) or Backward (-1)
            lr: Left (-1), Straight (0), or Right (1)

        Returns:
            List of angle sequences for all 4 legs
        """
        # Trot parameters (from Sunfounder trot.py)
        SECTION_COUNT = 2
        STEP_COUNT = 3
        LEG_RAISE_ORDER = [[1, 4], [2, 3]]  # Diagonal pairs
        LEG_STEP_HEIGHT = 20
        LEG_STEP_WIDTH = 100
        CENTER_OF_GRAVITY = -17
        LEG_STAND_OFFSET = 5
        Z_ORIGIN = 80

        TURNING_RATE = 0.5
        LEG_STAND_OFFSET_DIRS = [-1, -1, 1, 1]
        LEG_STEP_SCALES_LEFT = [TURNING_RATE, 1, TURNING_RATE, 1]
        LEG_STEP_SCALES_MIDDLE = [1, 1, 1, 1]
        LEG_STEP_SCALES_RIGHT = [1, TURNING_RATE, 1, TURNING_RATE]
        LEG_STEP_SCALES = [LEG_STEP_SCALES_LEFT, LEG_STEP_SCALES_MIDDLE, LEG_STEP_SCALES_RIGHT]
        LEG_ORIGINAL_Y_TABLE = [0, 1, 1, 0]

        if fb == 1:
            if lr == 0:
                y_offset = 0 + CENTER_OF_GRAVITY
            else:
                y_offset = -2 + CENTER_OF_GRAVITY
        else:
            if lr == 0:
                y_offset = 8 + CENTER_OF_GRAVITY
            else:
                y_offset = 1 + CENTER_OF_GRAVITY

        leg_step_width = [LEG_STEP_WIDTH * LEG_STEP_SCALES[lr+1][i] for i in range(4)]
        section_length = [leg_step_width[i] / (SECTION_COUNT-1) for i in range(4)]
        step_down_length = [section_length[i] / STEP_COUNT for i in range(4)]
        leg_offset = [LEG_STAND_OFFSET * LEG_STAND_OFFSET_DIRS[i] for i in range(4)]
        leg_origin = [leg_step_width[i] / 2 + y_offset + (leg_offset[i] * LEG_STEP_SCALES[lr+1][i]) for i in range(4)]

        # Generate coordinates
        origin_leg_coord = [[leg_origin[i] - LEG_ORIGINAL_Y_TABLE[i] * section_length[i], Z_ORIGIN] for i in range(4)]
        all_coords = []

        for section in range(SECTION_COUNT):
            for step in range(STEP_COUNT):
                if fb == 1:
                    raise_legs = LEG_RAISE_ORDER[section]
                else:
                    raise_legs = LEG_RAISE_ORDER[SECTION_COUNT - section - 1]

                leg_coord = []
                for i in range(4):
                    if i + 1 in raise_legs:
                        # Stepping leg
                        theta = step * pi / (STEP_COUNT-1)
                        temp = (leg_step_width[i] * (cos(theta) - fb) / 2 * fb)
                        y = leg_origin[i] + temp
                        z = Z_ORIGIN - (LEG_STEP_HEIGHT * step / (STEP_COUNT-1))
                    else:
                        # Standing leg
                        y = origin_leg_coord[i][0] + step_down_length[i] * fb
                        z = Z_ORIGIN

                    leg_coord.append([y, z])

                origin_leg_coord = leg_coord
                all_coords.append(leg_coord)

        return all_coords

    @classmethod
    def coords_to_actions(cls, coords_sequence):
        """
        Convert coordinate sequences to action sequences.

        Args:
            coords_sequence: List of [[y0, z0], [y1, z1], [y2, z2], [y3, z3]] for each timestep

        Returns:
            List of action arrays (8 joint angles) normalized to [-1, 1]
        """
        actions = []

        for coords in coords_sequence:
            # Convert each leg's (y, z) to (hip, knee) angles
            leg_angles = []
            for i, (y, z) in enumerate(coords):
                hip_angle, knee_angle = cls.coord2angle(y, z)

                # Mirror angles for right side legs (as in original code)
                if i % 2 != 0:  # Right side legs (1, 3)
                    hip_angle = -hip_angle
                    knee_angle = -knee_angle

                leg_angles.extend([hip_angle, knee_angle])

            # Normalize to [-1, 1] action space
            # Our environment expects actions in [-1, 1] which map to joint limits
            servo_range = (-np.pi/2, np.pi)  # From pidog_env.py
            normalized_action = []
            for angle in leg_angles:
                # Map from servo_range to [-1, 1]
                normalized = (angle - servo_range[0]) / (servo_range[1] - servo_range[0]) * 2 - 1
                normalized = np.clip(normalized, -1, 1)
                normalized_action.append(normalized)

            actions.append(np.array(normalized_action, dtype=np.float32))

        return actions


def collect_sunfounder_demonstrations(n_cycles=10, use_camera=False):
    """
    Collect demonstrations from Sunfounder gaits.

    Args:
        n_cycles: Number of gait cycles to collect
        use_camera: Whether to use camera observations

    Returns:
        Transitions object with demonstrations
    """
    print("\n" + "="*70)
    print(" EXTRACTING SUNFOUNDER PIDOG DEMONSTRATIONS")
    print("="*70)
    print(f"Collecting {n_cycles} cycles of expert gaits...")
    print("Gaits: forward walk, forward trot")
    print("\nThis uses the original Sunfounder PiDog motion planning!\n")

    extractor = SunfounderGaitExtractor()
    env = PiDogEnv(use_camera=use_camera)

    obs_list = []
    acts_list = []
    next_obs_list = []
    dones_list = []

    gaits = [
        ("Forward Walk", lambda: extractor.get_walk_gait(fb=1, lr=0)),
        ("Forward Trot", lambda: extractor.get_trot_gait(fb=1, lr=0)),
    ]

    total_transitions = 0

    for cycle in range(n_cycles):
        for gait_name, gait_func in gaits:
            # Get coordinate sequence for this gait
            coords_sequence = gait_func()

            # Convert to actions
            actions = extractor.coords_to_actions(coords_sequence)

            # Execute in environment and record
            obs, _ = env.reset()
            for action in actions:
                # Extract vector observation if using camera
                if use_camera and isinstance(obs, dict):
                    obs_vec = obs["vector"]
                else:
                    obs_vec = obs

                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)

                if use_camera and isinstance(next_obs, dict):
                    next_obs_vec = next_obs["vector"]
                else:
                    next_obs_vec = next_obs

                # Store transition
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
            print(f"  Completed {cycle + 1}/{n_cycles} cycles ({total_transitions} transitions)")

    env.close()

    if not IMITATION_AVAILABLE:
        print("\nWarning: Cannot create Transitions object (imitation not installed)")
        print("Saving as raw dict instead\n")
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

    print(f"\nCollected {len(transitions)} total transitions from Sunfounder gaits")
    print("="*70 + "\n")

    return transitions


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract Sunfounder PiDog gaits as demonstrations"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="demonstrations/sunfounder_demos.pkl",
        help="Output file path (default: demonstrations/sunfounder_demos.pkl)",
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=20,
        help="Number of gait cycles to collect (default: 20)",
    )
    parser.add_argument(
        "--use-camera",
        action="store_true",
        help="Use camera observations",
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect demonstrations
    demos = collect_sunfounder_demonstrations(
        n_cycles=args.n_cycles,
        use_camera=args.use_camera
    )

    # Save
    with open(output_path, "wb") as f:
        pickle.dump(demos, f)

    print(f"✓ Demonstrations saved to: {output_path}")
    print(f"\nTo use these for imitation learning:")
    print(f"  python train_pretrain_finetune.py \\")
    print(f"      --pretrained-demos {output_path} \\")
    print(f"      --total-bc-epochs 300 \\")
    print(f"      --total-rl-timesteps 500000\n")


if __name__ == "__main__":
    main()
