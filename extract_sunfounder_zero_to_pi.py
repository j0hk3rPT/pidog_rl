#!/usr/bin/env python3
"""
CORRECT VERSION with 0 to π range.

Based on user verification:
- ctrlrange in MuJoCo is 0 to π (0° to 180°)
- NOT -π/2 to π/2
- Leg naming is correct
- LEG=42mm, FOOT=76mm
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


class SunfounderExtractorZeroToPi:
    """Extractor with 0 to π control range."""

    LEG = 42.0
    FOOT = 76.0

    def __init__(self, standing_height=80):
        self.standing_height = standing_height
        # CORRECT range from pidog_actuators.xml: ctrlrange="0 3.14159"
        self.servo_range_low = 0.0
        self.servo_range_high = np.pi
        print(f"\n✓ CORRECT Control Range: 0 to π")
        print(f"  LEG: {self.LEG}mm, FOOT: {self.FOOT}mm")
        print(f"  Standing height: {standing_height}mm")
        print(f"  Control range: 0 to {self.servo_range_high:.4f} rad (0° to 180°)")
        print(f"  This matches pidog_actuators.xml ctrlrange!\n")

    @classmethod
    def coord2angle(cls, y, z):
        """Convert (y, z) to joint angles."""
        LEG = cls.LEG
        FOOT = cls.FOOT

        u = sqrt(y**2 + z**2)

        cos_beta = (FOOT**2 + LEG**2 - u**2) / (2 * FOOT * LEG)
        cos_beta = max(-1, min(1, cos_beta))
        beta = acos(cos_beta)

        angle1 = atan2(y, z)
        cos_angle2 = (LEG**2 + u**2 - FOOT**2) / (2 * LEG * u)
        cos_angle2 = max(-1, min(1, cos_angle2))
        angle2 = acos(cos_angle2)
        alpha = angle2 + angle1

        leg_angle = alpha
        foot_angle = beta - (pi / 2)

        return leg_angle, foot_angle

    def get_walk_gait(self, fb=1, lr=0):
        """Walking gait."""
        SECTION_COUNT = 8
        STEP_COUNT = 6
        LEG_ORDER = [1, 0, 4, 0, 2, 0, 3, 0]
        LEG_STEP_HEIGHT = 20
        LEG_STEP_WIDTH = 80
        CENTER_OF_GRAVITY = -15
        LEG_POSITION_OFFSETS = [-10, -10, 20, 20]
        Z_ORIGIN = self.standing_height

        TURNING_RATE = 0.3
        LEG_STEP_SCALES = [
            [TURNING_RATE, 1, TURNING_RATE, 1],
            [1, 1, 1, 1],
            [1, TURNING_RATE, 1, TURNING_RATE]
        ]
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
                raise_leg = LEG_ORDER[section] if fb == 1 else LEG_ORDER[SECTION_COUNT - section - 1]

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
        """Trotting gait."""
        SECTION_COUNT = 2
        STEP_COUNT = 3
        LEG_RAISE_ORDER = [[1, 4], [2, 3]]
        LEG_STEP_HEIGHT = 20
        LEG_STEP_WIDTH = 100
        CENTER_OF_GRAVITY = -17
        LEG_STAND_OFFSET = 5
        Z_ORIGIN = self.standing_height

        TURNING_RATE = 0.5
        LEG_STAND_OFFSET_DIRS = [-1, -1, 1, 1]
        LEG_STEP_SCALES = [
            [TURNING_RATE, 1, TURNING_RATE, 1],
            [1, 1, 1, 1],
            [1, TURNING_RATE, 1, TURNING_RATE]
        ]
        LEG_ORIGINAL_Y_TABLE = [0, 1, 1, 0]

        y_offset = (0 if lr == 0 else -2) + CENTER_OF_GRAVITY if fb == 1 else (8 if lr == 0 else 1) + CENTER_OF_GRAVITY

        leg_step_width = [LEG_STEP_WIDTH * LEG_STEP_SCALES[lr+1][i] for i in range(4)]
        section_length = [leg_step_width[i] / (SECTION_COUNT-1) for i in range(4)]
        step_down_length = [section_length[i] / STEP_COUNT for i in range(4)]
        leg_offset = [LEG_STAND_OFFSET * LEG_STAND_OFFSET_DIRS[i] for i in range(4)]
        leg_origin = [leg_step_width[i] / 2 + y_offset + (leg_offset[i] * LEG_STEP_SCALES[lr+1][i]) for i in range(4)]

        origin_leg_coord = [[leg_origin[i] - LEG_ORIGINAL_Y_TABLE[i] * section_length[i], Z_ORIGIN] for i in range(4)]
        all_coords = []

        for section in range(SECTION_COUNT):
            for step in range(STEP_COUNT):
                raise_legs = LEG_RAISE_ORDER[section] if fb == 1 else LEG_RAISE_ORDER[SECTION_COUNT - section - 1]

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

    def coords_to_actions(self, coords_sequence):
        """Convert coordinates to actions with 0 to π range."""
        actions = []

        for coords in coords_sequence:
            gait_order_angles = []

            for i, (y, z) in enumerate(coords):
                leg_angle, foot_angle = self.coord2angle(y, z)

                # Mirror RIGHT legs
                if i % 2 != 0:
                    leg_angle = -leg_angle
                    foot_angle = -foot_angle

                gait_order_angles.extend([leg_angle, foot_angle])

            FL_hip, FL_knee = gait_order_angles[0], gait_order_angles[1]
            FR_hip, FR_knee = gait_order_angles[2], gait_order_angles[3]
            BL_hip, BL_knee = gait_order_angles[4], gait_order_angles[5]
            BR_hip, BR_knee = gait_order_angles[6], gait_order_angles[7]

            # MuJoCo order: BR, FR, BL, FL
            mujoco_angles = [
                BR_hip, BR_knee,
                FR_hip, FR_knee,
                BL_hip, BL_knee,
                FL_hip, FL_knee,
            ]

            # Normalize to [-1, 1] using 0 to π range
            normalized_action = []
            for angle in mujoco_angles:
                # Map [0, π] to [-1, 1]
                normalized = (angle - self.servo_range_low) / (self.servo_range_high - self.servo_range_low) * 2 - 1
                normalized = np.clip(normalized, -1, 1)
                normalized_action.append(normalized)

            actions.append(np.array(normalized_action, dtype=np.float32))

        return actions


def collect_demonstrations(n_cycles=10, standing_height=80, use_camera=False):
    """Collect demonstrations."""
    print("\n" + "="*70)
    print(" SUNFOUNDER EXTRACTION - 0 TO π CONTROL RANGE")
    print("="*70)
    print(f"Cycles: {n_cycles}")
    print("="*70 + "\n")

    extractor = SunfounderExtractorZeroToPi(standing_height=standing_height)
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
            actions = extractor.coords_to_actions(coords)

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
    print("="*70 + "\n")
    return transitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", type=str, default="demonstrations/sunfounder_0_to_pi.pkl")
    parser.add_argument("--standing-height", type=float, default=80)
    parser.add_argument("--n-cycles", type=int, default=20)
    parser.add_argument("--use-camera", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    demos = collect_demonstrations(
        n_cycles=args.n_cycles,
        standing_height=args.standing_height,
        use_camera=args.use_camera
    )

    with open(output_path, "wb") as f:
        pickle.dump(demos, f)

    print(f"✓ Saved to: {output_path}\n")


if __name__ == "__main__":
    main()
