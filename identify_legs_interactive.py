#!/usr/bin/env python3
"""
Interactive leg identification tool.

Start with all joints at neutral, then move one joint at a time
to identify which MuJoCo actuator controls which physical leg.
"""

import numpy as np
import mujoco
import mujoco.viewer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pidog_env import PiDogEnv


def identify_legs():
    """Interactive leg identification."""
    print("\n" + "="*70)
    print(" INTERACTIVE LEG IDENTIFICATION")
    print("="*70)
    print()
    print("MuJoCo actuator order (8 joints total):")
    print("  [0] actuator 0: back_right_hip")
    print("  [1] actuator 1: back_right_knee")
    print("  [2] actuator 2: front_right_hip")
    print("  [3] actuator 3: front_right_knee")
    print("  [4] actuator 4: back_left_hip")
    print("  [5] actuator 5: back_left_knee")
    print("  [6] actuator 6: front_left_hip")
    print("  [7] actuator 7: front_left_knee")
    print()
    print("="*70)
    print()

    env = PiDogEnv(use_camera=False)
    obs, _ = env.reset()

    # Start with neutral position (environment's default)
    neutral_hip = -np.pi / 6   # -30°
    neutral_knee = -np.pi / 4  # -45°

    servo_range_low = -np.pi/2
    servo_range_high = np.pi

    # All legs neutral
    neutral_angles = [neutral_hip, neutral_knee] * 4
    current_action = []
    for angle in neutral_angles:
        normalized = (angle - servo_range_low) / (servo_range_high - servo_range_low) * 2 - 1
        current_action.append(normalized)

    current_action = np.array(current_action, dtype=np.float32)

    print("Controls:")
    print("  Enter joint index (0-7) and offset (-1.0 to 1.0)")
    print("  Example: '0 0.5' moves actuator 0 by +0.5")
    print("  Enter 'r' to reset all to neutral")
    print("  Enter 'q' to quit")
    print()
    print("Current action (all neutral):")
    for i, val in enumerate(current_action):
        print(f"  [{i}] = {val:6.3f}")
    print()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            # Step with current action
            obs, reward, terminated, truncated, info = env.step(current_action)
            viewer.sync()

            # Non-blocking input check
            print("\nEnter command (joint offset, 'r' reset, 'q' quit, or press Enter to continue): ", end='', flush=True)

            import select
            import sys

            # Wait for input with timeout
            if select.select([sys.stdin], [], [], 0.1)[0]:
                user_input = sys.stdin.readline().strip()

                if user_input == 'q':
                    print("Quitting...")
                    break
                elif user_input == 'r':
                    print("Resetting to neutral...")
                    current_action = np.array([normalized for angle in neutral_angles
                                              for normalized in [(angle - servo_range_low) / (servo_range_high - servo_range_low) * 2 - 1]],
                                             dtype=np.float32)
                    print("Reset complete")
                elif user_input:
                    try:
                        parts = user_input.split()
                        if len(parts) == 2:
                            joint_idx = int(parts[0])
                            offset = float(parts[1])

                            if 0 <= joint_idx <= 7:
                                current_action[joint_idx] += offset
                                current_action[joint_idx] = np.clip(current_action[joint_idx], -1.0, 1.0)

                                print(f"Moved joint {joint_idx} by {offset:+.3f}")
                                print(f"New value: [{joint_idx}] = {current_action[joint_idx]:6.3f}")
                            else:
                                print("Joint index must be 0-7")
                        else:
                            print("Format: 'joint_index offset' (e.g., '0 0.5')")
                    except ValueError:
                        print("Invalid input. Use format: 'joint_index offset'")

            if terminated:
                print("Robot fell! Resetting environment...")
                obs, _ = env.reset()

    env.close()

    print("\n" + "="*70)
    print(" SESSION ENDED")
    print("="*70)


if __name__ == "__main__":
    identify_legs()
