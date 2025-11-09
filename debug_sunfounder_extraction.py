#!/usr/bin/env python3
"""
Debug and tune Sunfounder gait extraction.

This tool helps diagnose issues with the extracted gaits by:
1. Showing the actual joint angles being commanded
2. Comparing to expected neutral positions
3. Visualizing coordinate-to-angle conversions
4. Testing different height and position offsets

Usage:
    # Debug current extraction
    python debug_sunfounder_extraction.py

    # Test with height adjustment
    python debug_sunfounder_extraction.py --height-offset 20

    # Test with different neutral position
    python debug_sunfounder_extraction.py --neutral-hip 90 --neutral-knee 90
"""

import argparse
import numpy as np
from math import sqrt, atan2, acos, cos, sin, pi, degrees, radians
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pidog_env import PiDogEnv
import mujoco
import mujoco.viewer
import time


def coord2angle_debug(y, z, show_calc=False):
    """
    Debug version of coordinate to angle conversion.

    Args:
        y: Forward/backward position (mm)
        z: Height position (mm)
        show_calc: Print calculation steps
    """
    LEG = 62  # mm
    FOOT = 62  # mm

    if show_calc:
        print(f"\n  Input: y={y:.1f}mm, z={z:.1f}mm")

    # Distance from hip to foot
    u = sqrt(y**2 + z**2)
    if show_calc:
        print(f"  Distance u = sqrt({y:.1f}² + {z:.1f}²) = {u:.1f}mm")

    # Knee angle (beta)
    cos_beta = (FOOT**2 + LEG**2 - u**2) / (2 * FOOT * LEG)
    cos_beta = max(-1, min(1, cos_beta))
    beta = acos(cos_beta)

    # Hip angle (alpha)
    angle1 = atan2(y, z)
    cos_angle2 = (LEG**2 + u**2 - FOOT**2) / (2 * LEG * u)
    cos_angle2 = max(-1, min(1, cos_angle2))
    angle2 = acos(cos_angle2)
    alpha = angle2 + angle1

    # Convert to degrees
    alpha_deg = degrees(alpha)
    beta_deg = degrees(beta)

    if show_calc:
        print(f"  Hip (alpha) = {alpha_deg:.1f}°")
        print(f"  Knee (beta) = {beta_deg:.1f}°")

    # Adjust beta (subtract 90° as in Sunfounder)
    beta_deg -= 90

    if show_calc:
        print(f"  Adjusted knee = {beta_deg:.1f}° (after -90°)")

    # Convert to radians
    hip_rad = radians(alpha_deg)
    knee_rad = radians(beta_deg)

    # Shift to servo range (0 to π)
    hip_servo = hip_rad + pi/2
    knee_servo = knee_rad + pi/2

    if show_calc:
        print(f"  Final servo positions:")
        print(f"    Hip:  {degrees(hip_servo):.1f}° ({hip_servo:.3f} rad)")
        print(f"    Knee: {degrees(knee_servo):.1f}° ({knee_servo:.3f} rad)")
        print(f"  Valid range: 0° to 180° (0 to {pi:.3f} rad)")
        if hip_servo < 0 or hip_servo > pi:
            print(f"  ⚠️  WARNING: Hip angle out of range!")
        if knee_servo < 0 or knee_servo > pi:
            print(f"  ⚠️  WARNING: Knee angle out of range!")

    return hip_rad, knee_rad


def test_standing_position(height_offset=0, neutral_hip_deg=90, neutral_knee_deg=90):
    """
    Test different standing positions.

    Args:
        height_offset: Offset to add to Z_ORIGIN (mm)
        neutral_hip_deg: Neutral hip angle (degrees)
        neutral_knee_deg: Neutral knee angle (degrees)
    """
    print("\n" + "="*70)
    print(" TESTING STANDING POSITION")
    print("="*70)

    # Sunfounder defaults
    Z_ORIGIN = 80 + height_offset  # Standing height in mm
    Y_NEUTRAL = 0  # Forward/backward neutral

    print(f"\nParameters:")
    print(f"  Height (Z): {Z_ORIGIN}mm")
    print(f"  Forward (Y): {Y_NEUTRAL}mm")
    print(f"  Target hip angle: {neutral_hip_deg}°")
    print(f"  Target knee angle: {neutral_knee_deg}°")

    print(f"\nCalculating joint angles for standing position...")
    hip_rad, knee_rad = coord2angle_debug(Y_NEUTRAL, Z_ORIGIN, show_calc=True)

    # Check against targets
    hip_deg = degrees(hip_rad)
    knee_deg = degrees(knee_rad)

    print(f"\nComparison:")
    print(f"  Hip:  calculated={hip_deg:.1f}°, target={neutral_hip_deg}°, diff={hip_deg-neutral_hip_deg:.1f}°")
    print(f"  Knee: calculated={knee_deg:.1f}°, target={neutral_knee_deg}°, diff={knee_deg-neutral_knee_deg:.1f}°")


def visualize_test_position(height_offset=0, forward_offset=0, render_fps=30):
    """
    Visualize the robot at a test standing position.

    Args:
        height_offset: Height adjustment (mm)
        forward_offset: Forward position adjustment (mm)
        render_fps: Frames per second
    """
    print("\n" + "="*70)
    print(" VISUALIZING TEST POSITION")
    print("="*70)
    print(f"Height offset: {height_offset}mm")
    print(f"Forward offset: {forward_offset}mm")
    print("\nPress Esc to exit")
    print("="*70 + "\n")

    # Create environment
    env = PiDogEnv(use_camera=False)

    # Calculate standing position for all legs
    Z_ORIGIN = 80 + height_offset
    Y_NEUTRAL = 0 + forward_offset

    # Get angles for neutral standing
    hip_rad, knee_rad = coord2angle_debug(Y_NEUTRAL, Z_ORIGIN, show_calc=False)

    print(f"Standing position angles:")
    print(f"  Hip:  {degrees(hip_rad):.1f}° ({hip_rad:.3f} rad)")
    print(f"  Knee: {degrees(knee_rad):.1f}° ({knee_rad:.3f} rad)")

    # Create action for all legs (same position)
    # MuJoCo order: RR, FR, RL, FL
    # All legs same position for standing
    action_angles = []
    for leg_idx in range(4):
        # Alternate left/right mirroring
        is_right = leg_idx in [0, 1]  # RR=0, FR=1 are right

        leg_hip = -hip_rad if is_right else hip_rad
        leg_knee = -knee_rad if is_right else knee_rad

        # Shift to servo range and normalize to [-1, 1]
        hip_servo = leg_hip + pi/2
        knee_servo = leg_knee + pi/2

        hip_norm = (hip_servo - 0) / pi * 2 - 1
        knee_norm = (knee_servo - 0) / pi * 2 - 1

        action_angles.extend([hip_norm, knee_norm])

    action = np.array(action_angles, dtype=np.float32)

    print(f"\nAction (normalized): {action}")
    print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")

    # Visualize
    frame_time = 1.0 / render_fps

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.sitegroup[0] = True

        obs, _ = env.reset()

        # Hold position
        print("\nHolding standing position... (Press Esc to exit)")
        while viewer.is_running():
            step_start = time.time()

            # Keep applying same action
            obs, reward, terminated, truncated, info = env.step(action)

            viewer.sync()

            # Print status occasionally
            height = env.data.qpos[2]
            if int(time.time()) % 2 == 0:  # Every 2 seconds
                print(f"  Height: {height*1000:.1f}mm, Reward: {reward:.2f}")

            # Control frame rate
            elapsed = time.time() - step_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    env.close()


def analyze_gait_heights():
    """Analyze the heights in Sunfounder walk/trot gaits."""
    print("\n" + "="*70)
    print(" ANALYZING SUNFOUNDER GAIT HEIGHTS")
    print("="*70)

    # Walk gait parameters
    Z_ORIGIN = 80  # mm
    LEG_STEP_HEIGHT = 20  # mm
    STEP_COUNT = 6

    print(f"\nWalk Gait:")
    print(f"  Standing height (Z_ORIGIN): {Z_ORIGIN}mm")
    print(f"  Step height: {LEG_STEP_HEIGHT}mm")
    print(f"  Min height during step: {Z_ORIGIN - LEG_STEP_HEIGHT}mm")
    print(f"  Max height: {Z_ORIGIN}mm")

    # Trot gait
    STEP_COUNT_TROT = 3
    print(f"\nTrot Gait:")
    print(f"  Standing height (Z_ORIGIN): {Z_ORIGIN}mm")
    print(f"  Step height: {LEG_STEP_HEIGHT}mm")
    print(f"  Min height during step: {Z_ORIGIN - LEG_STEP_HEIGHT}mm")
    print(f"  Max height: {Z_ORIGIN}mm")

    print(f"\nExpected real-world heights:")
    print(f"  Leg length: 62mm (upper) + 62mm (lower) = 124mm max")
    print(f"  Standing: ~{Z_ORIGIN}mm ({Z_ORIGIN/10:.1f}cm)")
    print(f"  During step: ~{Z_ORIGIN-LEG_STEP_HEIGHT}mm ({(Z_ORIGIN-LEG_STEP_HEIGHT)/10:.1f}cm)")

    # Convert to meters for comparison with MuJoCo
    print(f"\nMuJoCo scale (meters):")
    print(f"  Standing: {Z_ORIGIN/1000:.3f}m")
    print(f"  During step: {(Z_ORIGIN-LEG_STEP_HEIGHT)/1000:.3f}m")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Debug Sunfounder gait extraction"
    )
    parser.add_argument(
        "--height-offset",
        type=float,
        default=0,
        help="Height offset in mm (default: 0, try +20 for higher stance)",
    )
    parser.add_argument(
        "--forward-offset",
        type=float,
        default=0,
        help="Forward position offset in mm (default: 0)",
    )
    parser.add_argument(
        "--neutral-hip",
        type=float,
        default=90,
        help="Target neutral hip angle in degrees (default: 90)",
    )
    parser.add_argument(
        "--neutral-knee",
        type=float,
        default=90,
        help="Target neutral knee angle in degrees (default: 90)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize test position in 3D",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze, don't visualize",
    )

    args = parser.parse_args()

    # Always show analysis
    analyze_gait_heights()
    test_standing_position(
        height_offset=args.height_offset,
        neutral_hip_deg=args.neutral_hip,
        neutral_knee_deg=args.neutral_knee
    )

    if not args.analyze_only:
        # Visualize
        if args.visualize or input("\nVisualize test position? (y/n): ").lower() == 'y':
            visualize_test_position(
                height_offset=args.height_offset,
                forward_offset=args.forward_offset
            )


if __name__ == "__main__":
    main()
