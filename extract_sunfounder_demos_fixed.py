#!/usr/bin/env python3
"""
FIXED: Extract expert demonstrations from original Sunfounder PiDog gaits.

FIXES:
1. Correct actuator ordering to match MuJoCo (RR, FR, RL, FL not FL, FR, RL, RR)
2. Correct joint range (0 to π, not -π/2 to π)
3. Proper angle mirroring for left vs right legs

Usage:
    python extract_sunfounder_demos_fixed.py --output-file demonstrations/sunfounder_demos.pkl
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from math import sqrt, atan2, acos, cos, sin, pi
import sys

sys.path.insert(0, str(Path(__file__).parent))
from pidog_env import PiDogEnv

# Try to import imitation
try:
    from imitation.data.types import Transitions
    IMITATION_AVAILABLE = True
except ImportError:
    IMITATION_AVAILABLE = False


class SunfounderGaitExtractorFixed:
    """Extract gaits from Sunfounder PiDog motion planning with CORRECT mappings."""

    # Leg dimensions (from Sunfounder pidog.py) in mm
    LEG = 62    # Upper leg length
    FOOT = 62   # Lower leg length

    def __init__(self):
        """Initialize gait extractor."""
        print("\n🔧 Using FIXED gait extractor with correct actuator mapping!")
        print("   - MuJoCo order: RR, FR, RL, FL")
        print("   - Joint range: 0 to π (not negative)")
        print("   - Proper left/right mirroring\n")

    @staticmethod
    def coord2angle(y, z):
        """
        Convert (y, z) coordinates to joint angles using inverse kinematics.

        Args:
            y: Forward/backward position (mm)
            z: Height position (mm)

        Returns:
            (hip_angle, knee_angle) in radians
        """
        LEG = SunfounderGaitExtractorFixed.LEG
        FOOT = SunfounderGaitExtractorFixed.FOOT

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

        # Convert to degrees
        alpha_deg = alpha * 180 / pi
        beta_deg = beta * 180 / pi

        # Adjust beta (knee) - subtract 90 as in original Sunfounder code
        beta_deg -= 90

        # Convert back to radians
        hip_angle = alpha_deg * pi / 180
        knee_angle = beta_deg * pi / 180

        return hip_angle, knee_angle

    @staticmethod
    def get_walk_gait(fb=1, lr=0):
        """Generate walking gait coordinates (same as before)."""
        # Walk parameters (from Sunfounder walk.py)
        SECTION_COUNT = 8
        STEP_COUNT = 6
        LEG_ORDER = [1, 0, 4, 0, 2, 0, 3, 0]
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
                        theta = step * pi / (STEP_COUNT-1)
                        temp = (leg_step_width[i] * (cos(theta) - fb) / 2 * fb)
                        y = leg_origin[i] + temp
                        z = Z_ORIGIN - (LEG_STEP_HEIGHT * step / (STEP_COUNT-1))
                    else:
                        y = leg_coord[i][0] + step_down_length[i] * fb
                        z = Z_ORIGIN

                    leg_coord[i] = [y, z]

                all_coords.append(list.copy(leg_coord))

        return all_coords

    @staticmethod
    def get_trot_gait(fb=1, lr=0):
        """Generate trotting gait coordinates (same as before)."""
        SECTION_COUNT = 2
        STEP_COUNT = 3
        LEG_RAISE_ORDER = [[1, 4], [2, 3]]
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
                        theta = step * pi / (STEP_COUNT-1)
                        temp = (leg_step_width[i] * (cos(theta) - fb) / 2 * fb)
                        y = leg_origin[i] + temp
                        z = Z_ORIGIN - (LEG_STEP_HEIGHT * step / (STEP_COUNT-1))
                    else:
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

        FIXED: Proper actuator ordering and joint range mapping!

        Sunfounder leg order: FL(0), FR(1), RL(2), RR(3)
        MuJoCo actuator order: RR, FR, RL, FL (back-right, front-right, back-left, front-left)

        Mapping:
        - Sunfounder[0] = FL → MuJoCo[6,7]
        - Sunfounder[1] = FR → MuJoCo[2,3]
        - Sunfounder[2] = RL → MuJoCo[4,5]
        - Sunfounder[3] = RR → MuJoCo[0,1]
        """
        actions = []

        for coords in coords_sequence:
            # Convert each leg's (y, z) to (hip, knee) angles
            # Sunfounder order: FL, FR, RL, RR
            sunfounder_angles = []
            for i, (y, z) in enumerate(coords):
                hip_angle, knee_angle = cls.coord2angle(y, z)

                # Mirror for right-side legs (FR=1, RR=3)
                if i % 2 != 0:  # Right side
                    hip_angle = -hip_angle
                    knee_angle = -knee_angle

                sunfounder_angles.append([hip_angle, knee_angle])

            # Sunfounder: [FL_hip, FL_knee, FR_hip, FR_knee, RL_hip, RL_knee, RR_hip, RR_knee]
            FL_hip, FL_knee = sunfounder_angles[0]
            FR_hip, FR_knee = sunfounder_angles[1]
            RL_hip, RL_knee = sunfounder_angles[2]
            RR_hip, RR_knee = sunfounder_angles[3]

            # Reorder to MuJoCo: [RR_hip, RR_knee, FR_hip, FR_knee, RL_hip, RL_knee, FL_hip, FL_knee]
            mujoco_angles = [
                RR_hip, RR_knee,  # Back right
                FR_hip, FR_knee,  # Front right
                RL_hip, RL_knee,  # Back left (REAR LEFT)
                FL_hip, FL_knee,  # Front left
            ]

            # Normalize to [-1, 1] action space
            # MuJoCo servos: 0 to π radians
            servo_min = 0.0
            servo_max = pi
            normalized_action = []

            for angle in mujoco_angles:
                # Map from [0, π] to [-1, 1]
                # But angles from IK can be negative, so we need to handle that
                # The servos are 0 to π, so negative angles need adjustment

                # Add π/2 to shift range (angles are typically around 0)
                shifted_angle = angle + pi/2

                # Normalize to [-1, 1]
                normalized = (shifted_angle - servo_min) / (servo_max - servo_min) * 2 - 1
                normalized = np.clip(normalized, -1, 1)
                normalized_action.append(normalized)

            actions.append(np.array(normalized_action, dtype=np.float32))

        return actions


def collect_sunfounder_demonstrations_fixed(n_cycles=10, use_camera=False):
    """Collect demonstrations with FIXED mappings."""
    print("\n" + "="*70)
    print(" EXTRACTING SUNFOUNDER PIDOG DEMONSTRATIONS (FIXED)")
    print("="*70)
    print(f"Collecting {n_cycles} cycles of expert gaits...")
    print("Gaits: forward walk, forward trot")
    print("\n✓ Correct actuator ordering (RR, FR, RL, FL)")
    print("✓ Correct joint range (0 to π)")
    print("✓ Proper left/right mirroring\n")

    extractor = SunfounderGaitExtractorFixed()
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
            coords_sequence = gait_func()
            actions = extractor.coords_to_actions(coords_sequence)

            obs, _ = env.reset()
            for action in actions:
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
            print(f"  Completed {cycle + 1}/{n_cycles} cycles ({total_transitions} transitions)")

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

    print(f"\n✓ Collected {len(transitions)} total transitions from Sunfounder gaits")
    print("="*70 + "\n")

    return transitions


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract Sunfounder PiDog gaits (FIXED VERSION)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="demonstrations/sunfounder_demos_fixed.pkl",
        help="Output file path",
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

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    demos = collect_sunfounder_demonstrations_fixed(
        n_cycles=args.n_cycles,
        use_camera=args.use_camera
    )

    with open(output_path, "wb") as f:
        pickle.dump(demos, f)

    print(f"✓ Demonstrations saved to: {output_path}")
    print(f"\nTo verify:")
    print(f"  python visualize_demonstrations.py --demo-file {output_path}\n")


if __name__ == "__main__":
    main()
