#!/usr/bin/env python3
"""
Test individual joint movements to understand directions.
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pidog_env import PiDogEnv


def test_joint_individually():
    """Test each joint one at a time to see what direction moves it."""

    env = PiDogEnv(use_camera=False, render_mode="human")

    joint_names = ["RH_hip", "RH_knee", "RF_hip", "RF_knee",
                   "LH_hip", "LH_knee", "LF_hip", "LF_knee"]

    # Neutral position in normalized space
    neutral_hip_norm = -0.556   # -30° → -0.524 rad
    neutral_knee_norm = -0.667  # -45° → -0.785 rad

    neutral_action = np.array([
        neutral_hip_norm, neutral_knee_norm,  # RH
        neutral_hip_norm, neutral_knee_norm,  # RF
        neutral_hip_norm, neutral_knee_norm,  # LH
        neutral_hip_norm, neutral_knee_norm,  # LF
    ])

    print("=" * 80)
    print("TESTING JOINT DIRECTIONS")
    print("=" * 80)
    print("\nStarting at neutral standing position...")
    print("Hip: -30°, Knee: -45°")
    print("\nWe'll test each joint by moving it +20° and -20° from neutral")
    print("=" * 80)

    obs, _ = env.reset()

    # Hold neutral for a moment
    print("\n1. Holding neutral position for 2 seconds...")
    for _ in range(100):
        obs, _, _, _, info = env.step(neutral_action)
        time.sleep(0.02)

    print(f"   Neutral height: {info['body_height']:.4f}m")
    print(f"   Neutral velocity: {info['forward_velocity']:.4f}m/s")

    # Test each joint
    for joint_idx in range(8):
        joint_name = joint_names[joint_idx]
        is_hip = (joint_idx % 2 == 0)

        print(f"\n2. Testing {joint_name} {'(hip)' if is_hip else '(knee)'}...")

        # Create actions for +20° and -20°
        delta_deg = 20
        delta_rad = delta_deg * np.pi / 180
        low, high = -np.pi/2, np.pi
        delta_norm = 2 * delta_rad / (high - low)

        # Test positive direction
        test_action_pos = neutral_action.copy()
        test_action_pos[joint_idx] += delta_norm
        test_action_pos = np.clip(test_action_pos, -1.0, 1.0)

        print(f"   a) Moving {joint_name} +20° (more positive)...")
        for _ in range(50):
            obs, _, _, _, info = env.step(test_action_pos)
            time.sleep(0.02)

        height_pos = info['body_height']
        vel_pos = info['forward_velocity']
        print(f"      Height: {height_pos:.4f}m, Velocity: {vel_pos:.4f}m/s")

        # Return to neutral
        for _ in range(25):
            obs, _, _, _, info = env.step(neutral_action)
            time.sleep(0.02)

        # Test negative direction
        test_action_neg = neutral_action.copy()
        test_action_neg[joint_idx] -= delta_norm
        test_action_neg = np.clip(test_action_neg, -1.0, 1.0)

        print(f"   b) Moving {joint_name} -20° (more negative)...")
        for _ in range(50):
            obs, _, _, _, info = env.step(test_action_neg)
            time.sleep(0.02)

        height_neg = info['body_height']
        vel_neg = info['forward_velocity']
        print(f"      Height: {height_neg:.4f}m, Velocity: {vel_neg:.4f}m/s")

        # Analysis
        height_neutral = 0.14  # Expected
        if height_pos > height_neutral:
            print(f"      → +20° makes leg EXTEND/STAND UP")
        elif height_pos < height_neutral:
            print(f"      → +20° makes leg BEND/SIT DOWN")

        if height_neg > height_neutral:
            print(f"      → -20° makes leg EXTEND/STAND UP")
        elif height_neg < height_neutral:
            print(f"      → -20° makes leg BEND/SIT DOWN")

        # Return to neutral
        for _ in range(25):
            obs, _, _, _, info = env.step(neutral_action)
            time.sleep(0.02)

        print(f"      Returned to neutral")

        time.sleep(1)  # Pause between joints

    print("\n" + "=" * 80)
    print("TEST COMPLETE - Keep window open to observe")
    print("=" * 80)
    print("\nPress Ctrl+C to exit...")

    try:
        while True:
            obs, _, _, _, _ = env.step(neutral_action)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nClosing...")
    finally:
        env.close()


if __name__ == "__main__":
    test_joint_individually()
