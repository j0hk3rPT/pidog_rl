#!/usr/bin/env python3
"""
Simple walking gait starting from working neutral position.

We know hip=-30°, knee=-45° works for standing.
Now let's create a simple walking pattern by varying these angles.
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from math import sin, cos, pi
import sys

sys.path.insert(0, str(Path(__file__).parent))
from pidog_env import PiDogEnv

try:
    from imitation.data.types import Transitions
    IMITATION_AVAILABLE = True
except ImportError:
    IMITATION_AVAILABLE = False


def simple_walk_gait(num_steps=48):
    """
    Create a simple walking gait from neutral position.

    Gait pattern: Trot (diagonal legs move together)
    - Phase 0-0.5: FR+BL lift and swing forward
    - Phase 0.5-1.0: FL+BR lift and swing forward
    """
    # Neutral angles that work
    neutral_hip = -np.pi / 6   # -30°
    neutral_knee = -np.pi / 4  # -45°

    # Gait parameters
    hip_swing = 0.3  # ±17° hip swing for stepping
    knee_lift = 0.5  # ±29° knee bend for leg lift

    all_actions = []

    for step in range(num_steps):
        phase = (step / num_steps) * 2 * pi  # 0 to 2π

        # MuJoCo order: BR, FR, BL, FL (hip, knee for each)

        # Diagonal pair 1: FR + BL
        # Lift when sin(phase) > 0, swing forward with cos(phase)
        if sin(phase) > 0:  # Lift phase
            FR_hip = neutral_hip + hip_swing * cos(phase)
            FR_knee = neutral_knee - knee_lift * sin(phase)
            BL_hip = neutral_hip + hip_swing * cos(phase)
            BL_knee = neutral_knee - knee_lift * sin(phase)
        else:  # Ground phase
            FR_hip = neutral_hip
            FR_knee = neutral_knee
            BL_hip = neutral_hip
            BL_knee = neutral_knee

        # Diagonal pair 2: FL + BR (opposite phase)
        if sin(phase + pi) > 0:  # Lift phase (180° out of phase)
            FL_hip = neutral_hip + hip_swing * cos(phase + pi)
            FL_knee = neutral_knee - knee_lift * sin(phase + pi)
            BR_hip = neutral_hip + hip_swing * cos(phase + pi)
            BR_knee = neutral_knee - knee_lift * sin(phase + pi)
        else:  # Ground phase
            FL_hip = neutral_hip
            FL_knee = neutral_knee
            BR_hip = neutral_hip
            BR_knee = neutral_knee

        # Assemble in MuJoCo order
        angles = [
            BR_hip, BR_knee,
            FR_hip, FR_knee,
            BL_hip, BL_knee,
            FL_hip, FL_knee,
        ]

        # Normalize to [-1, 1]
        ctrl_range_low = -np.pi/2
        ctrl_range_high = np.pi

        action = []
        for angle in angles:
            normalized = (angle - ctrl_range_low) / (ctrl_range_high - ctrl_range_low) * 2 - 1
            normalized = np.clip(normalized, -1, 1)
            action.append(normalized)

        all_actions.append(np.array(action, dtype=np.float32))

    return all_actions


def collect_simple_walk(n_cycles=10, use_camera=False):
    """Collect simple walking demonstrations."""
    print("\n" + "="*70)
    print(" SIMPLE WALK FROM WORKING NEUTRAL")
    print("="*70)
    print(f"Cycles: {n_cycles}")
    print("Using working neutral: hip=-30°, knee=-45°")
    print("Trot gait: diagonal legs move together")
    print("="*70 + "\n")

    env = PiDogEnv(use_camera=use_camera)

    obs_list = []
    acts_list = []
    next_obs_list = []
    dones_list = []

    total_transitions = 0

    for cycle in range(n_cycles):
        # Generate walk cycle
        actions = simple_walk_gait(num_steps=48)

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
    parser.add_argument("--output-file", type=str, default="demonstrations/simple_walk.pkl")
    parser.add_argument("--n-cycles", type=int, default=20)
    parser.add_argument("--use-camera", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    demos = collect_simple_walk(
        n_cycles=args.n_cycles,
        use_camera=args.use_camera
    )

    with open(output_path, "wb") as f:
        pickle.dump(demos, f)

    print(f"✓ Saved to: {output_path}")
    print(f"\nVisualize with:")
    print(f"  python visualize_demonstrations.py --demo-file {output_path}\n")


if __name__ == "__main__":
    main()
