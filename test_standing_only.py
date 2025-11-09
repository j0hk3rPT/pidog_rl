#!/usr/bin/env python3
"""
Test: Just use environment's neutral standing position.

This tests if we can even hold a standing position correctly,
without trying to walk. If this works, we know the angle mapping is correct.
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from pidog_env import PiDogEnv

try:
    from imitation.data.types import Transitions
    IMITATION_AVAILABLE = True
except ImportError:
    IMITATION_AVAILABLE = False


def collect_standing_demonstrations(n_steps=500, use_camera=False):
    """Collect demonstrations of just standing still."""
    print("\n" + "="*70)
    print(" TESTING: NEUTRAL STANDING POSITION ONLY")
    print("="*70)
    print(f"Steps: {n_steps}")
    print("This uses the environment's neutral position (hip=-30°, knee=-45°)")
    print("="*70 + "\n")

    env = PiDogEnv(use_camera=use_camera)

    # Environment's neutral angles
    neutral_hip = -np.pi / 6   # -30°
    neutral_knee = -np.pi / 4  # -45°

    # All 4 legs get same neutral angles
    # MuJoCo order: RR, FR, RL, FL (each has hip, knee)
    neutral_angles = [neutral_hip, neutral_knee] * 4

    # Normalize to [-1, 1]
    servo_range_low = -np.pi/2
    servo_range_high = np.pi

    neutral_action = []
    for angle in neutral_angles:
        normalized = (angle - servo_range_low) / (servo_range_high - servo_range_low) * 2 - 1
        normalized = np.clip(normalized, -1, 1)
        neutral_action.append(normalized)

    neutral_action = np.array(neutral_action, dtype=np.float32)

    print("Neutral action (all legs same):")
    print(f"  Hip:  {neutral_hip:.3f} rad ({np.degrees(neutral_hip):.1f}°) → normalized: {neutral_action[0]:.3f}")
    print(f"  Knee: {neutral_knee:.3f} rad ({np.degrees(neutral_knee):.1f}°) → normalized: {neutral_action[1]:.3f}")
    print()

    obs_list = []
    acts_list = []
    next_obs_list = []
    dones_list = []

    obs, _ = env.reset()

    for step in range(n_steps):
        if use_camera and isinstance(obs, dict):
            obs_vec = obs["vector"]
        else:
            obs_vec = obs

        next_obs, reward, terminated, truncated, info = env.step(neutral_action)

        if use_camera and isinstance(next_obs, dict):
            next_obs_vec = next_obs["vector"]
        else:
            next_obs_vec = next_obs

        obs_list.append(obs_vec)
        acts_list.append(neutral_action)
        next_obs_list.append(next_obs_vec)
        dones_list.append(terminated or truncated)

        obs = next_obs

        if step % 100 == 0:
            print(f"  Step {step}: height={info['body_height']:.3f}m, vel={info['forward_velocity']:.3f}m/s, reward={reward:.2f}")

        if terminated or truncated:
            print(f"\n  Robot fell at step {step}!")
            break

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
    parser = argparse.ArgumentParser(description="Test neutral standing position")
    parser.add_argument(
        "--output-file",
        type=str,
        default="demonstrations/test_standing.pkl",
        help="Output file",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=500,
        help="Number of steps (default: 500)",
    )
    parser.add_argument(
        "--use-camera",
        action="store_true",
        help="Use camera",
    )

    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    demos = collect_standing_demonstrations(
        n_steps=args.n_steps,
        use_camera=args.use_camera
    )

    with open(output_path, "wb") as f:
        pickle.dump(demos, f)

    print(f"✓ Saved to: {output_path}")
    print(f"\nVerify with:")
    print(f"  python visualize_demonstrations.py --demo-file {output_path}\n")


if __name__ == "__main__":
    main()
