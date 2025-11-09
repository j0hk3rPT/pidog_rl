#!/usr/bin/env python3
"""
Find what neutral angles the environment actually uses.

The environment sets neutral_hip=-π/6 and neutral_knee=-π/4.
Let's see what position this creates and if the robot stands.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pidog_env import PiDogEnv

def test_environment_neutral():
    """Test the environment's built-in neutral position."""
    print("\n" + "="*70)
    print(" ENVIRONMENT'S NEUTRAL POSITION")
    print("="*70)
    print()

    env = PiDogEnv(use_camera=False)

    # From pidog_env.py:
    neutral_hip = -np.pi / 6   # -30°
    neutral_knee = -np.pi / 4  # -45°

    print(f"Environment's neutral angles:")
    print(f"  Hip:  {neutral_hip:.4f} rad ({np.degrees(neutral_hip):.1f}°)")
    print(f"  Knee: {neutral_knee:.4f} rad ({np.degrees(neutral_knee):.1f}°)")
    print()

    # All 4 legs get same angles
    # MuJoCo order: BR, FR, BL, FL
    neutral_angles = [neutral_hip, neutral_knee] * 4

    # Normalize to [-1, 1]
    servo_range_low = -np.pi/2
    servo_range_high = np.pi

    action = []
    for angle in neutral_angles:
        normalized = (angle - servo_range_low) / (servo_range_high - servo_range_low) * 2 - 1
        action.append(normalized)

    action = np.array(action, dtype=np.float32)

    print("Normalized action:")
    names = ["BR_hip", "BR_knee", "FR_hip", "FR_knee", "BL_hip", "BL_knee", "FL_hip", "FL_knee"]
    for name, val in zip(names, action):
        print(f"  {name:12s}: {val:7.4f}")
    print()

    # Test it
    obs, _ = env.reset()

    print("Testing environment neutral for 500 steps...")
    print()

    fell = False
    for step in range(500):
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 100 == 0:
            print(f"  Step {step}: height={info['body_height']:.3f}m, vel={info['forward_velocity']:.3f}m/s, reward={reward:.2f}")

        if terminated or truncated:
            print(f"\n  ❌ Robot fell at step {step}!")
            fell = True
            break

    env.close()

    if not fell:
        print(f"\n  ✅ Robot stayed standing for all 500 steps!")

    print()
    print("="*70)
    print()

    return not fell

if __name__ == "__main__":
    success = test_environment_neutral()
    if success:
        print("The environment's neutral position WORKS!")
        print("We should use hip=-30°, knee=-45° for all legs.")
    else:
        print("Even the environment's neutral position fails!")
        print("There may be a deeper issue with the MuJoCo model.")
