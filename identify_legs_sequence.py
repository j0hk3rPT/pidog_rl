#!/usr/bin/env python3
"""
Sequential leg identification tool.

Moves each joint one at a time so you can identify which actuator
controls which physical leg.
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pidog_env import PiDogEnv


def test_joint_sequence():
    """Test each joint sequentially."""
    print("\n" + "="*70)
    print(" SEQUENTIAL LEG IDENTIFICATION")
    print("="*70)
    print()
    print("This will move each joint one at a time.")
    print("Watch which leg moves to identify the mapping!")
    print()
    print("MuJoCo actuator names:")
    print("  [0] back_right_hip")
    print("  [1] back_right_knee")
    print("  [2] front_right_hip")
    print("  [3] front_right_knee")
    print("  [4] back_left_hip")
    print("  [5] back_left_knee")
    print("  [6] front_left_hip")
    print("  [7] front_left_knee")
    print()
    print("="*70)
    print()

    env = PiDogEnv(use_camera=False)

    # Neutral position
    neutral_hip = -np.pi / 6   # -30°
    neutral_knee = -np.pi / 4  # -45°

    servo_range_low = -np.pi/2
    servo_range_high = np.pi

    def angle_to_normalized(angle):
        return (angle - servo_range_low) / (servo_range_high - servo_range_low) * 2 - 1

    neutral_action = np.array([angle_to_normalized(a) for a in [neutral_hip, neutral_knee] * 4], dtype=np.float32)

    joint_names = [
        "back_right_hip (actuator 0)",
        "back_right_knee (actuator 1)",
        "front_right_hip (actuator 2)",
        "front_right_knee (actuator 3)",
        "back_left_hip (actuator 4)",
        "back_left_knee (actuator 5)",
        "front_left_hip (actuator 6)",
        "front_left_knee (actuator 7)",
    ]

    # Test each joint
    for joint_idx in range(8):
        print(f"\n{'='*70}")
        print(f" TESTING: {joint_names[joint_idx]}")
        print(f"{'='*70}")
        print(f"Moving joint {joint_idx} by +0.3...")
        print()

        obs, _ = env.reset()

        # First hold neutral for 1 second
        for _ in range(30):
            obs, reward, terminated, truncated, info = env.step(neutral_action)
            if terminated:
                obs, _ = env.reset()

        # Then move the joint
        test_action = neutral_action.copy()
        test_action[joint_idx] += 0.3
        test_action[joint_idx] = np.clip(test_action[joint_idx], -1.0, 1.0)

        print(f"Holding test position for 3 seconds...")
        for step in range(90):  # 3 seconds at 30 FPS
            obs, reward, terminated, truncated, info = env.step(test_action)

            if step % 30 == 0:
                print(f"  Step {step//30 + 1}/3: height={info['body_height']:.3f}m")

            if terminated:
                print("  Robot fell!")
                obs, _ = env.reset()
                break

        print(f"Returning to neutral...")
        for _ in range(30):
            obs, reward, terminated, truncated, info = env.step(neutral_action)

        print()
        print("WHICH LEG MOVED? (write down for later)")
        print("  - Front Left?")
        print("  - Front Right?")
        print("  - Back Left?")
        print("  - Back Right?")
        print()
        input("Press Enter to test next joint...")

    env.close()

    print("\n" + "="*70)
    print(" ALL JOINTS TESTED")
    print("="*70)
    print()
    print("Please tell me which physical leg each actuator controls:")
    print("  actuator 0 (back_right_hip) → ?")
    print("  actuator 1 (back_right_knee) → ?")
    print("  actuator 2 (front_right_hip) → ?")
    print("  actuator 3 (front_right_knee) → ?")
    print("  actuator 4 (back_left_hip) → ?")
    print("  actuator 5 (back_left_knee) → ?")
    print("  actuator 6 (front_left_hip) → ?")
    print("  actuator 7 (front_left_knee) → ?")
    print()


if __name__ == "__main__":
    test_joint_sequence()
