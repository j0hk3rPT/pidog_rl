#!/usr/bin/env python3
"""
IMPROVED: Sunfounder gait extraction with height tuning.

KEY IMPROVEMENTS:
1. Adjustable standing height (default 100mm instead of 80mm for taller stance)
2. Proper servo angle mapping
3. Debug output to verify angles are in valid range

Usage:
    # Default (taller stance)
    python extract_sunfounder_demos_v2.py --n-cycles 20

    # Custom height
    python extract_sunfounder_demos_v2.py --standing-height 110 --n-cycles 20

    # Match original Sunfounder exactly
    python extract_sunfounder_demos_v2.py --standing-height 80 --n-cycles 20
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from math import sqrt, atan2, acos, cos, sin, pi, degrees
import sys

sys.path.insert(0, str(Path(__file__).parent))
from pidog_env import PiDogEnv

try:
    from imitation.data.types import Transitions
    IMITATION_AVAILABLE = True
except ImportError:
    IMITATION_AVAILABLE = False


class SunfounderGaitExtractorV2:
    """
    Improved gait extractor with height tuning.

    Default standing height increased from 80mm to 100mm for better stability.
    """

    LEG = 62    # Upper leg length (mm)
    FOOT = 62   # Lower leg length (mm)

    def __init__(self, standing_height=100):
        """
        Initialize extractor.

        Args:
            standing_height: Standing height in mm (default: 100, original: 80)
        """
        self.standing_height = standing_height
        print(f"\n🔧 Gait Extractor V2")
        print(f"   Standing height: {standing_height}mm ({standing_height/10:.1f}cm)")
        print(f"   (Original Sunfounder uses 80mm)")

    @staticmethod
    def coord2angle(y, z):
        """
        Convert (y, z) coordinates to joint angles.

        Returns angles in radians relative to vertical.
        """
        LEG = SunfounderGaitExtractorV2.LEG
        FOOT = SunfounderGaitExtractorV2.FOOT

        u = sqrt(y**2 + z**2)

        # Knee angle
        cos_beta = (FOOT**2 + LEG**2 - u**2) / (2 * FOOT * LEG)
        cos_beta = max(-1, min(1, cos_beta))
        beta = acos(cos_beta)

        # Hip angle
        angle1 = atan2(y, z)
        cos_angle2 = (LEG**2 + u**2 - FOOT**2) / (2 * LEG * u)
        cos_angle2 = max(-1, min(1, cos_angle2))
        angle2 = acos(cos_angle2)
        alpha = angle2 + angle1

        # Sunfounder adjustment
        beta_deg = degrees(beta) - 90
        alpha_deg = degrees(alpha)

        # Convert back to radians
        return np.radians(alpha_deg), np.radians(beta_deg)

    def get_walk_gait(self, fb=1, lr=0):
        """Generate walking gait with custom height."""
        SECTION_COUNT = 8
        STEP_COUNT = 6
        LEG_ORDER = [1, 0, 4, 0, 2, 0, 3, 0]
        LEG_STEP_HEIGHT = 20
        LEG_STEP_WIDTH = 80
        CENTER_OF_GRAVITY = -15
        LEG_POSITION_OFFSETS = [-10, -10, 20, 20]
        Z_ORIGIN = self.standing_height  # ← Use custom height

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

    def get_trot_gait(self, fb=1, lr=0):
        """Generate trot gait with custom height."""
        SECTION_COUNT = 2
        STEP_COUNT = 3
        LEG_RAISE_ORDER = [[1, 4], [2, 3]]
        LEG_STEP_HEIGHT = 20
        LEG_STEP_WIDTH = 100
        CENTER_OF_GRAVITY = -17
        LEG_STAND_OFFSET = 5
        Z_ORIGIN = self.standing_height  # ← Use custom height

        TURNING_RATE = 0.5
        LEG_STAND_OFFSET_DIRS = [-1, -1, 1, 1]
        LEG_STEP_SCALES_LEFT = [TURNING_RATE, 1, TURNING_RATE, 1]
        LEG_STEP_SCALES_MIDDLE = [1, 1, 1, 1]
        LEG_STEP_SCALES_RIGHT = [1, TURNING_RATE, 1, TURNING_RATE]
        LEG_STEP_SCALES = [LEG_STEP_SCALES_LEFT, LEG_STEP_SCALES_MIDDLE, LEG_STEP_SCALES_RIGHT]
        LEG_ORIGINAL_Y_TABLE = [0, 1, 1, 0]

        if fb == 1:
            y_offset = 0 + CENTER_OF_GRAVITY if lr == 0 else -2 + CENTER_OF_GRAVITY
        else:
            y_offset = 8 + CENTER_OF_GRAVITY if lr == 0 else 1 + CENTER_OF_GRAVITY

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
    def coords_to_actions(cls, coords_sequence, check_range=True):
        """
        Convert coordinates to actions with proper MuJoCo ordering.

        Args:
            coords_sequence: List of leg coordinates
            check_range: Print warnings if angles out of range
        """
        actions = []
        out_of_range_count = 0

        for coords in coords_sequence:
            # Sunfounder order: FL, FR, RL, RR
            sunfounder_angles = []
            for i, (y, z) in enumerate(coords):
                hip_angle, knee_angle = cls.coord2angle(y, z)

                # Mirror right side
                if i % 2 != 0:
                    hip_angle = -hip_angle
                    knee_angle = -knee_angle

                sunfounder_angles.append([hip_angle, knee_angle])

            FL_hip, FL_knee = sunfounder_angles[0]
            FR_hip, FR_knee = sunfounder_angles[1]
            RL_hip, RL_knee = sunfounder_angles[2]
            RR_hip, RR_knee = sunfounder_angles[3]

            # MuJoCo order: RR, FR, RL, FL
            mujoco_angles = [
                RR_hip, RR_knee,
                FR_hip, FR_knee,
                RL_hip, RL_knee,
                FL_hip, FL_knee,
            ]

            # Map to servo range (0 to π) then normalize
            normalized_action = []
            for angle in mujoco_angles:
                # Shift to positive range
                shifted = angle + pi/2

                # Check if in valid servo range
                if check_range and (shifted < 0 or shifted > pi):
                    out_of_range_count += 1

                # Normalize to [-1, 1]
                normalized = (shifted - 0) / pi * 2 - 1
                normalized = np.clip(normalized, -1, 1)
                normalized_action.append(normalized)

            actions.append(np.array(normalized_action, dtype=np.float32))

        if check_range and out_of_range_count > 0:
            print(f"⚠️  Warning: {out_of_range_count} angles out of servo range (0 to π)")
            print(f"   Consider adjusting standing height")

        return actions


def collect_demonstrations_v2(n_cycles=10, standing_height=100, use_camera=False):
    """Collect demonstrations with custom standing height."""
    print("\n" + "="*70)
    print(" SUNFOUNDER GAIT EXTRACTION V2")
    print("="*70)
    print(f"Collecting {n_cycles} cycles")
    print(f"Standing height: {standing_height}mm ({standing_height/10:.1f}cm)")
    print("Gaits: walk, trot")
    print("="*70 + "\n")

    extractor = SunfounderGaitExtractorV2(standing_height=standing_height)
    env = PiDogEnv(use_camera=use_camera)

    obs_list = []
    acts_list = []
    next_obs_list = []
    dones_list = []

    gaits = [
        ("Walk", lambda: extractor.get_walk_gait(fb=1, lr=0)),
        ("Trot", lambda: extractor.get_trot_gait(fb=1, lr=0)),
    ]

    total_transitions = 0

    for cycle in range(n_cycles):
        for gait_name, gait_func in gaits:
            coords = gait_func()
            actions = extractor.coords_to_actions(coords, check_range=(cycle==0))

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
    print(f"  Standing height: {standing_height}mm")
    print("="*70 + "\n")

    return transitions


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Extract Sunfounder gaits V2")
    parser.add_argument(
        "--output-file",
        type=str,
        default="demonstrations/sunfounder_demos_v2.pkl",
        help="Output file",
    )
    parser.add_argument(
        "--standing-height",
        type=float,
        default=100,
        help="Standing height in mm (default: 100, original: 80, try 90-110)",
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=20,
        help="Number of gait cycles (default: 20)",
    )
    parser.add_argument(
        "--use-camera",
        action="store_true",
        help="Use camera observations",
    )

    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    demos = collect_demonstrations_v2(
        n_cycles=args.n_cycles,
        standing_height=args.standing_height,
        use_camera=args.use_camera
    )

    with open(output_path, "wb") as f:
        pickle.dump(demos, f)

    print(f"✓ Saved to: {output_path}")
    print(f"\nNext: Verify with:")
    print(f"  docker-compose run --rm pidog_rl python visualize_demonstrations.py \\")
    print(f"      --demo-file {output_path}")
    print(f"\nTry different heights if robot is still too low/high:")
    print(f"  --standing-height 90   (lower)")
    print(f"  --standing-height 100  (default)")
    print(f"  --standing-height 110  (higher)\n")


if __name__ == "__main__":
    main()
