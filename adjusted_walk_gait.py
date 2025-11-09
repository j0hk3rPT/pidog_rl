#!/usr/bin/env python3
"""
Simple walking gait with adjusted neutral position.

Changes from basic neutral:
- Move all shoulders back by 10° (more stable stance)
- Space front legs differently from back legs
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


def simple_walk_gait_adjusted(num_steps=48):
    """
    Create walking gait with adjusted neutral position.

    Neutral adjustments:
    - Base angles: hip=-45°, knee=-45°
    - Front legs offset forward, back legs offset back for spacing

    Note: Actions are in [-1, 1] space, scaled by env to [-π/2, π/2]
    To get -45° joint angle: action = -π/4 / (π/2) = -0.5
    """
    # Base neutral (action values for -45°)
    base_hip = -np.pi / 4 / (np.pi / 2)  # -0.5 → -45° joint angle
    base_knee = -np.pi / 4 / (np.pi / 2)  # -0.5 → -45° joint angle

    # Front/back spacing offset (in action space)
    front_offset = 0.3 / (np.pi / 2)  # ~17° forward
    back_offset = -0.3 / (np.pi / 2)  # ~17° back

    # Front legs neutral
    front_hip = base_hip + front_offset
    front_knee = base_knee

    # Back legs neutral
    back_hip = base_hip + back_offset
    back_knee = base_knee

    print(f"Adjusted neutral positions (action values):")
    print(f"  Front legs: hip={front_hip:.3f}, knee={front_knee:.3f}")
    print(f"  Back legs:  hip={back_hip:.3f}, knee={back_knee:.3f}")
    print(f"  (These map to ~{np.degrees(front_hip * np.pi/2):.1f}° and ~{np.degrees(back_hip * np.pi/2):.1f}° joint angles)")
    print()

    # Gait parameters (in action space)
    hip_swing = 0.3 / (np.pi / 2)  # ±17° hip swing
    knee_lift = 0.5 / (np.pi / 2)  # ±29° knee lift

    all_actions = []

    for step in range(num_steps):
        phase = (step / num_steps) * 2 * pi

        # Diagonal pair 1: FR + BL
        if sin(phase) > 0:  # Lift phase
            FR_hip = front_hip + hip_swing * cos(phase)
            FR_knee = front_knee - knee_lift * sin(phase)
            BL_hip = back_hip + hip_swing * cos(phase)
            BL_knee = back_knee - knee_lift * sin(phase)
        else:  # Ground phase
            FR_hip = front_hip
            FR_knee = front_knee
            BL_hip = back_hip
            BL_knee = back_knee

        # Diagonal pair 2: FL + BR (opposite phase)
        if sin(phase + pi) > 0:
            FL_hip = front_hip + hip_swing * cos(phase + pi)
            FL_knee = front_knee - knee_lift * sin(phase + pi)
            BR_hip = back_hip + hip_swing * cos(phase + pi)
            BR_knee = back_knee - knee_lift * sin(phase + pi)
        else:
            FL_hip = front_hip
            FL_knee = front_knee
            BR_hip = back_hip
            BR_knee = back_knee

        # MuJoCo order: BR, FR, BL, FL (already in action space [-1, 1])
        action = [
            BR_hip, BR_knee,
            FR_hip, FR_knee,
            BL_hip, BL_knee,
            FL_hip, FL_knee,
        ]

        # Clip to valid action range
        action = [np.clip(a, -1, 1) for a in action]

        all_actions.append(np.array(action, dtype=np.float32))

    return all_actions


def collect_adjusted_walk(n_cycles=10, use_camera=False):
    """Collect walking demonstrations with adjusted stance."""
    print("\n" + "="*70)
    print(" ADJUSTED WALK GAIT")
    print("="*70)
    print(f"Cycles: {n_cycles}")
    print("Adjustments:")
    print("  - Shoulders moved back 10°")
    print("  - Front/back legs spaced differently")
    print("="*70 + "\n")

    env = PiDogEnv(use_camera=use_camera)

    obs_list = []
    acts_list = []
    next_obs_list = []
    dones_list = []

    total_transitions = 0

    for cycle in range(n_cycles):
        actions = simple_walk_gait_adjusted(num_steps=48)

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
    parser.add_argument("--output-file", type=str, default="demonstrations/adjusted_walk.pkl")
    parser.add_argument("--n-cycles", type=int, default=20)
    parser.add_argument("--use-camera", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    demos = collect_adjusted_walk(
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
