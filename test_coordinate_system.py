#!/usr/bin/env python3
"""
Test the MuJoCo coordinate system to understand forward direction.

This will help us figure out if we need to flip the forward direction.
"""

import numpy as np
import sys
from pathlib import Path
import mujoco
import mujoco.viewer
import time

sys.path.insert(0, str(Path(__file__).parent))
from pidog_env import PiDogEnv


def test_neutral_standing():
    """Test what the neutral standing position should be."""
    print("\n" + "="*70)
    print(" TEST 1: NEUTRAL STANDING POSITION")
    print("="*70)

    env = PiDogEnv(use_camera=False)

    # Try all zeros action (should be middle of range)
    action_zeros = np.zeros(8, dtype=np.float32)

    print("\nTesting action = [0, 0, 0, 0, 0, 0, 0, 0] (middle of servo range)")
    print("Expected: Robot standing in neutral position\n")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        obs, _ = env.reset()

        print("Holding neutral position for 5 seconds...")
        print("Watch the robot - is it standing upright?\n")

        for _ in range(150):  # 5 seconds at 30 Hz
            obs, reward, term, trunc, info = env.step(action_zeros)
            viewer.sync()
            time.sleep(1/30)

        height = env.data.qpos[2]
        print(f"Final height: {height:.3f}m ({height*100:.1f}cm)")
        print(f"Body upright? Check the viewer!")

    env.close()


def test_forward_backward():
    """Test which direction is 'forward' in MuJoCo."""
    print("\n" + "="*70)
    print(" TEST 2: FORWARD vs BACKWARD DIRECTION")
    print("="*70)

    env = PiDogEnv(use_camera=False)

    # Create a simple action that should push forward
    # Try extending rear legs, pulling front legs
    # Action order: RR_hip, RR_knee, FR_hip, FR_knee, RL_hip, RL_knee, FL_hip, FL_knee

    print("\nTest A: Rear legs push backward (should move robot forward)")
    action_push = np.array([
        0.3, 0.0,   # RR: hip forward
        0.3, 0.0,   # FR: hip forward
        0.3, 0.0,   # RL: hip forward
        0.3, 0.0,   # FL: hip forward
    ], dtype=np.float32)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        obs, _ = env.reset()
        initial_pos = env.data.qpos[0]  # X position

        print(f"Initial X position: {initial_pos:.3f}m")
        print("Applying forward-pushing action for 3 seconds...")

        for _ in range(90):
            obs, reward, term, trunc, info = env.step(action_push)
            viewer.sync()
            time.sleep(1/30)

        final_pos = env.data.qpos[0]
        delta = final_pos - initial_pos

        print(f"Final X position: {final_pos:.3f}m")
        print(f"Delta: {delta:.3f}m")

        if delta > 0.01:
            print("✓ Robot moved in POSITIVE X direction")
        elif delta < -0.01:
            print("✓ Robot moved in NEGATIVE X direction")
        else:
            print("⚠️  Robot didn't move much")

    env.close()

    # Test opposite
    print("\nTest B: Opposite action (pull forward)")
    action_pull = -action_push

    env = PiDogEnv(use_camera=False)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        obs, _ = env.reset()
        initial_pos = env.data.qpos[0]

        print(f"Initial X position: {initial_pos:.3f}m")
        print("Applying opposite action for 3 seconds...")

        for _ in range(90):
            obs, reward, term, trunc, info = env.step(action_pull)
            viewer.sync()
            time.sleep(1/30)

        final_pos = env.data.qpos[0]
        delta = final_pos - initial_pos

        print(f"Final X position: {final_pos:.3f}m")
        print(f"Delta: {delta:.3f}m")

        if delta > 0.01:
            print("✓ Robot moved in POSITIVE X direction")
        elif delta < -0.01:
            print("✓ Robot moved in NEGATIVE X direction")
        else:
            print("⚠️  Robot didn't move much")

    env.close()


def test_left_right_symmetry():
    """Test if left and right sides are properly mirrored."""
    print("\n" + "="*70)
    print(" TEST 3: LEFT-RIGHT SYMMETRY")
    print("="*70)

    env = PiDogEnv(use_camera=False)

    # Move only left legs
    print("\nTest A: Move left legs only")
    action_left = np.array([
        0.0, 0.0,   # RR (right)
        0.0, 0.0,   # FR (right)
        0.5, 0.3,   # RL (LEFT)
        0.5, 0.3,   # FL (LEFT)
    ], dtype=np.float32)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        obs, _ = env.reset()

        print("Moving left legs...")
        print("Watch: Should tilt to one side")

        for _ in range(90):
            obs, reward, term, trunc, info = env.step(action_left)
            viewer.sync()
            time.sleep(1/30)

        # Check roll (tilt left/right)
        quat = env.data.qpos[3:7]
        w, x, y, z = quat
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))

        print(f"Body roll: {np.degrees(roll):.1f}°")
        if abs(roll) > 0.1:
            if roll > 0:
                print("✓ Tilted to one side (positive roll)")
            else:
                print("✓ Tilted to one side (negative roll)")
        else:
            print("⚠️  Not much tilt")

    env.close()


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" COORDINATE SYSTEM DIAGNOSIS")
    print("="*70)
    print("\nThese tests will help identify:")
    print("  1. Correct neutral standing position")
    print("  2. Which direction is 'forward' in MuJoCo")
    print("  3. If left-right mirroring is correct")
    print("\nPress Ctrl+C to skip a test")
    print("="*70)

    try:
        input("\nPress Enter to start Test 1 (Neutral Standing)...")
        test_neutral_standing()
    except KeyboardInterrupt:
        print("\nSkipped Test 1")

    try:
        input("\nPress Enter to start Test 2 (Forward/Backward)...")
        test_forward_backward()
    except KeyboardInterrupt:
        print("\nSkipped Test 2")

    try:
        input("\nPress Enter to start Test 3 (Left/Right Symmetry)...")
        test_left_right_symmetry()
    except KeyboardInterrupt:
        print("\nSkipped Test 3")

    print("\n" + "="*70)
    print(" DIAGNOSIS COMPLETE")
    print("="*70)
    print("\nBased on the tests:")
    print("  - If robot moved backward when expecting forward → flip forward direction")
    print("  - If robot fell with neutral action → wrong servo range mapping")
    print("  - If left/right not symmetric → check mirroring logic")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
