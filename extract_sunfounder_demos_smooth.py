#!/usr/bin/env python3
"""
SMOOTH TRANSITION VERSION: Gradually move from neutral to gait start position.

The key insight: Environment resets to neutral_hip=-30°, neutral_knee=-45°,
but Sunfounder gait expects a different starting position. We need to smoothly
transition from neutral to the gait's first frame before starting the walk cycle.
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


class SunfounderExtractorSmooth:
    """Extractor with smooth transition from neutral to gait start."""

    LEG = 62
    FOOT = 62

    def __init__(self, standing_height=100, transition_steps=10):
        """
        Args:
            standing_height: Height in mm (default 100)
            transition_steps: Number of steps to transition from neutral to gait start
        """
        self.standing_height = standing_height
        self.transition_steps = transition_steps
        self.servo_range_low = -np.pi/2
        self.servo_range_high = np.pi
        print(f"\n✓ SMOOTH TRANSITION Gait Extractor")
        print(f"  Standing height: {standing_height}mm")
        print(f"  Transition steps: {transition_steps}")
        print(f"  NO mirroring + FLIPPED Y\n")

    @staticmethod
    def coord2angle(y, z):
        """Convert (y, z) to joint angles with Y flip."""
        LEG = SunfounderExtractorSmooth.LEG
        FOOT = SunfounderExtractorSmooth.FOOT

        # FLIP Y
        y = -y

        u = sqrt(y**2 + z**2)

        cos_beta = (FOOT**2 + LEG**2 - u**2) / (2 * FOOT * LEG)
        cos_beta = max(-1, min(1, cos_beta))
        beta = acos(cos_beta)

        angle1 = atan2(y, z)
        cos_angle2 = (LEG**2 + u**2 - FOOT**2) / (2 * LEG * u)
        cos_angle2 = max(-1, min(1, cos_angle2))
        angle2 = acos(cos_angle2)
        alpha = angle2 + angle1

        alpha_deg = degrees(alpha)
        beta_deg = degrees(beta) - 90

        return radians(alpha_deg), radians(beta_deg)

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
        """Convert coordinates to actions (NO MIRRORING + FLIPPED Y)."""
        actions = []

        for coords in coords_sequence:
            sunfounder_angles = []
            for i, (y, z) in enumerate(coords):
                hip_angle, knee_angle = self.coord2angle(y, z)
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

            # Normalize
            normalized_action = []
            for angle in mujoco_angles:
                normalized = (angle - self.servo_range_low) / (self.servo_range_high - self.servo_range_low) * 2 - 1
                normalized = np.clip(normalized, -1, 1)
                normalized_action.append(normalized)

            actions.append(np.array(normalized_action, dtype=np.float32))

        return actions

    def create_transition(self, start_action, end_action, steps):
        """Create smooth transition between two actions."""
        transition_actions = []
        for i in range(steps):
            alpha = (i + 1) / steps  # 0 to 1
            interpolated = (1 - alpha) * start_action + alpha * end_action
            transition_actions.append(interpolated)
        return transition_actions


def collect_smooth_demonstrations(n_cycles=10, standing_height=100, transition_steps=10, use_camera=False):
    """Collect demonstrations with smooth transitions from neutral."""
    print("\n" + "="*70)
    print(" SUNFOUNDER EXTRACTION - SMOOTH TRANSITION")
    print("="*70)
    print(f"Cycles: {n_cycles}")
    print(f"Height: {standing_height}mm")
    print(f"Transition steps: {transition_steps}")
    print("="*70 + "\n")

    extractor = SunfounderExtractorSmooth(standing_height=standing_height, transition_steps=transition_steps)
    env = PiDogEnv(use_camera=use_camera)

    obs_list = []
    acts_list = []
    next_obs_list = []
    dones_list = []

    gaits = [
        ("Walk", lambda: extractor.get_walk_gait(fb=1, lr=0)),
        ("Trot", lambda: extractor.get_trot_gait(fb=1, lr=0)),
    ]

    # Get neutral position action (environment's default)
    neutral_hip = -np.pi / 6   # -30°
    neutral_knee = -np.pi / 4  # -45°

    # All legs same in neutral
    neutral_angles = [neutral_hip, neutral_knee] * 4  # RR, FR, RL, FL
    neutral_action = []
    for angle in neutral_angles:
        normalized = (angle - extractor.servo_range_low) / (extractor.servo_range_high - extractor.servo_range_low) * 2 - 1
        normalized = np.clip(normalized, -1, 1)
        neutral_action.append(normalized)
    neutral_action = np.array(neutral_action, dtype=np.float32)

    total_transitions = 0

    for cycle in range(n_cycles):
        for gait_name, gait_func in gaits:
            coords = gait_func()
            gait_actions = extractor.coords_to_actions(coords)

            # Create smooth transition from neutral to gait start
            transition_actions = extractor.create_transition(
                neutral_action,
                gait_actions[0],
                transition_steps
            )

            obs, _ = env.reset()

            # First execute transition
            for action in transition_actions:
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
                    print(f"  WARNING: Robot fell during transition in {gait_name}!")
                    obs, _ = env.reset()
                    break

            # Then execute gait
            if not (terminated or truncated):
                for action in gait_actions:
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
    """Main function."""
    parser = argparse.ArgumentParser(description="Smooth transition gait extraction")
    parser.add_argument(
        "--output-file",
        type=str,
        default="demonstrations/sunfounder_smooth.pkl",
        help="Output file",
    )
    parser.add_argument(
        "--standing-height",
        type=float,
        default=100,
        help="Standing height in mm (default: 100)",
    )
    parser.add_argument(
        "--transition-steps",
        type=int,
        default=10,
        help="Steps to transition from neutral to gait start (default: 10)",
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=20,
        help="Number of cycles (default: 20)",
    )
    parser.add_argument(
        "--use-camera",
        action="store_true",
        help="Use camera",
    )

    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    demos = collect_smooth_demonstrations(
        n_cycles=args.n_cycles,
        standing_height=args.standing_height,
        transition_steps=args.transition_steps,
        use_camera=args.use_camera
    )

    with open(output_path, "wb") as f:
        pickle.dump(demos, f)

    print(f"✓ Saved to: {output_path}")
    print(f"\nVerify with:")
    print(f"  python visualize_demonstrations.py --demo-file {output_path}\n")


if __name__ == "__main__":
    main()
