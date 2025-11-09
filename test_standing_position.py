#!/usr/bin/env python3
"""
Test neutral standing position to verify angle calculations are correct.

This will:
1. Calculate neutral standing angles using IK
2. Apply them to the robot
3. Show if the robot stands properly
"""

import numpy as np
from math import sqrt, atan2, acos, cos, sin, pi, degrees, radians
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pidog_env import PiDogEnv
import mujoco


LEG = 62  # mm
FOOT = 62  # mm


def coord2angle(y, z):
    """Original IK calculation."""
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


def test_standing_position(standing_height=80, mirror_right=True):
    """Test standing position."""
    print("\n" + "="*70)
    print(" TESTING NEUTRAL STANDING POSITION")
    print("="*70)
    print(f"Standing height: {standing_height}mm")
    print(f"Mirror right side: {mirror_right}")
    print("="*70 + "\n")

    # Neutral position: feet directly below hips
    y = 0  # Foot directly below hip (no forward/backward offset)
    z = standing_height  # Height in mm

    print(f"Foot position: y={y}mm, z={z}mm")

    hip_angle, knee_angle = coord2angle(y, z)
    print(f"Calculated angles: hip={degrees(hip_angle):.1f}°, knee={degrees(knee_angle):.1f}°")
    print(f"Calculated angles (rad): hip={hip_angle:.3f}, knee={knee_angle:.3f}")
    print()

    # Create actions for all 4 legs
    # Sunfounder order: FL, FR, RL, RR
    sunfounder_angles = []
    for i in range(4):
        h, k = hip_angle, knee_angle

        # Mirror right side (FR=1, RR=3)?
        if mirror_right and i % 2 != 0:
            h, k = -h, -k

        sunfounder_angles.append([h, k])

    print("Sunfounder order angles:")
    for i, (h, k) in enumerate(sunfounder_angles):
        leg_name = ["FL", "FR", "RL", "RR"][i]
        print(f"  {leg_name}: hip={degrees(h):6.1f}°, knee={degrees(k):6.1f}°")
    print()

    # Reorder to MuJoCo: RR, FR, RL, FL
    FL_hip, FL_knee = sunfounder_angles[0]
    FR_hip, FR_knee = sunfounder_angles[1]
    RL_hip, RL_knee = sunfounder_angles[2]
    RR_hip, RR_knee = sunfounder_angles[3]

    mujoco_angles = [
        RR_hip, RR_knee,
        FR_hip, FR_knee,
        RL_hip, RL_knee,
        FL_hip, FL_knee,
    ]

    print("MuJoCo order angles (RR, FR, RL, FL):")
    names = ["RR_hip", "RR_knee", "FR_hip", "FR_knee", "RL_hip", "RL_knee", "FL_hip", "FL_knee"]
    for name, angle in zip(names, mujoco_angles):
        print(f"  {name}: {degrees(angle):6.1f}° ({angle:.3f} rad)")
    print()

    # Normalize to [-1, 1]
    servo_range_low = -np.pi/2
    servo_range_high = np.pi

    normalized_action = []
    for angle in mujoco_angles:
        normalized = (angle - servo_range_low) / (servo_range_high - servo_range_low) * 2 - 1
        normalized = np.clip(normalized, -1, 1)
        normalized_action.append(normalized)

    print("Normalized actions [-1, 1]:")
    for name, norm in zip(names, normalized_action):
        print(f"  {name}: {norm:6.3f}")
    print()

    # Test in environment
    print("Testing in MuJoCo environment...")
    print("This should show robot in neutral standing position.")
    print("Press Ctrl+C to exit.\n")

    env = PiDogEnv(use_camera=False)
    obs, _ = env.reset()

    action = np.array(normalized_action, dtype=np.float32)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        step_count = 0
        while viewer.is_running() and step_count < 300:  # 10 seconds at 30 FPS
            obs, reward, terminated, truncated, info = env.step(action)

            viewer.sync()
            time.sleep(1.0 / 30)  # 30 FPS

            if step_count % 30 == 0:  # Every second
                print(f"  Step {step_count}: height={info['body_height']:.3f}m, velocity={info['forward_velocity']:.3f}m/s")

            step_count += 1

            if terminated:
                print("  Robot fell!")
                break

    env.close()

    print("\n" + "="*70)
    print(" TEST COMPLETE")
    print("="*70)
    print("\nDid the robot stand properly?")
    print("  YES → IK and mirroring are correct")
    print("  NO  → Need to adjust angles or mirroring")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=float, default=80, help="Standing height in mm")
    parser.add_argument("--no-mirror", action="store_true", help="Don't mirror right side")
    args = parser.parse_args()

    test_standing_position(
        standing_height=args.height,
        mirror_right=not args.no_mirror
    )
