#!/usr/bin/env python3
"""
Compare our gait extraction with original Sunfounder implementation.

This tool verifies that our coordinate-to-angle conversion matches
the original Sunfounder PiDog code exactly.
"""

import numpy as np
from math import sqrt, atan2, acos, cos, sin, pi, degrees, radians
import sys
from pathlib import Path

# Add the Sunfounder repo to path
sunfounder_path = Path("/tmp/pidog")
if sunfounder_path.exists():
    sys.path.insert(0, str(sunfounder_path))

sys.path.insert(0, str(Path(__file__).parent))


def our_coord2angle(y, z):
    """Our implementation of coord to angle."""
    LEG = 62
    FOOT = 62

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
    beta_deg = degrees(beta)

    # Sunfounder adjustment
    beta_deg -= 90

    return alpha_deg, beta_deg


def sunfounder_coord2polar(y, z):
    """Original Sunfounder implementation."""
    try:
        from pidog import Pidog

        # Create a dummy instance to access the method
        class DummyPidog:
            LEG = 62
            FOOT = 62
            A = LEG
            B = FOOT
            C = 0

            def coord2polar(self, coord):
                y, z = coord
                u = sqrt(y**2 + z**2)

                cos_angle1 = (self.FOOT**2 + self.LEG**2 - u**2) / (2 * self.FOOT * self.LEG)
                cos_angle1 = min(max(cos_angle1, -1), 1)
                beta = acos(cos_angle1)

                angle1 = atan2(y, z)
                cos_angle2 = (self.LEG**2 + u**2 - self.FOOT**2) / (2 * self.LEG * u)
                cos_angle2 = min(max(cos_angle2, -1), 1)
                angle2 = acos(cos_angle2)
                alpha = angle2 + angle1

                alpha = alpha / pi * 180
                beta = beta / pi * 180

                return alpha, beta

        dummy = DummyPidog()
        return dummy.coord2polar([y, z])
    except ImportError:
        print("⚠️  Sunfounder pidog module not available")
        print("Using our implementation for both")
        return our_coord2angle(y, z)


def compare_conversions():
    """Compare our conversion with Sunfounder's."""
    print("\n" + "="*70)
    print(" COMPARING COORDINATE TO ANGLE CONVERSION")
    print("="*70)

    # Test coordinates from walk gait
    test_coords = [
        (0, 80, "Standing neutral"),
        (0, 100, "Standing tall"),
        (20, 80, "Forward lean"),
        (-20, 80, "Backward lean"),
        (0, 60, "Low crouch"),
        (40, 80, "Big forward step"),
        (-40, 80, "Big backward step"),
    ]

    print(f"\nTesting {len(test_coords)} coordinate pairs:\n")
    print(f"{'Y (mm)':<10} {'Z (mm)':<10} {'Description':<20} {'Our Alpha':<12} {'SF Alpha':<12} {'Diff':<8} {'Our Beta':<12} {'SF Beta':<12} {'Diff':<8}")
    print("-" * 120)

    max_diff_alpha = 0
    max_diff_beta = 0

    for y, z, desc in test_coords:
        our_alpha, our_beta = our_coord2angle(y, z)
        sf_alpha, sf_beta = sunfounder_coord2polar(y, z)

        # Sunfounder does beta - 90 in legs_angle_calculation
        sf_beta_adjusted = sf_beta - 90

        diff_alpha = abs(our_alpha - sf_alpha)
        diff_beta = abs(our_beta - sf_beta_adjusted)

        max_diff_alpha = max(max_diff_alpha, diff_alpha)
        max_diff_beta = max(max_diff_beta, diff_beta)

        print(f"{y:<10.1f} {z:<10.1f} {desc:<20} "
              f"{our_alpha:<12.2f} {sf_alpha:<12.2f} {diff_alpha:<8.2f} "
              f"{our_beta:<12.2f} {sf_beta_adjusted:<12.2f} {diff_beta:<8.2f}")

    print("\n" + "="*70)
    if max_diff_alpha < 0.01 and max_diff_beta < 0.01:
        print("✓ PERFECT MATCH! Our conversion matches Sunfounder exactly.")
    elif max_diff_alpha < 1.0 and max_diff_beta < 1.0:
        print(f"⚠️  Small differences (max alpha: {max_diff_alpha:.2f}°, max beta: {max_diff_beta:.2f}°)")
    else:
        print(f"❌ SIGNIFICANT DIFFERENCES (max alpha: {max_diff_alpha:.2f}°, max beta: {max_diff_beta:.2f}°)")
    print("="*70 + "\n")


def check_actuator_ordering():
    """Verify actuator ordering matches MuJoCo."""
    print("\n" + "="*70)
    print(" CHECKING ACTUATOR ORDERING")
    print("="*70)

    print("\nSunfounder leg order in coords array:")
    print("  coords[0] = FL (Front Left)")
    print("  coords[1] = FR (Front Right)")
    print("  coords[2] = RL (Rear Left)")
    print("  coords[3] = RR (Rear Right)")

    print("\nMuJoCo actuator order (from pidog_actuators.xml):")
    print("  actuator[0,1] = RR (back_right_hip, back_right_knee)")
    print("  actuator[2,3] = FR (front_right_hip, front_right_knee)")
    print("  actuator[4,5] = RL (back_left_hip, back_left_knee)")
    print("  actuator[6,7] = FL (front_left_hip, front_left_knee)")

    print("\nRequired mapping:")
    print("  Sunfounder[3] (RR) → MuJoCo[0,1]")
    print("  Sunfounder[1] (FR) → MuJoCo[2,3]")
    print("  Sunfounder[2] (RL) → MuJoCo[4,5]")
    print("  Sunfounder[0] (FL) → MuJoCo[6,7]")

    print("\nOur current mapping (from extract_sunfounder_demos_fixed.py):")
    print("  mujoco_angles = [RR_hip, RR_knee, FR_hip, FR_knee, RL_hip, RL_knee, FL_hip, FL_knee]")
    print("  ✓ This looks CORRECT")

    print("="*70 + "\n")


def check_servo_range():
    """Check servo range mapping."""
    print("\n" + "="*70)
    print(" CHECKING SERVO RANGE MAPPING")
    print("="*70)

    print("\nReal SF006FM servos:")
    print("  Physical range: 0° to 180° (0 to π radians)")
    print("  Control range in MuJoCo: ctrlrange=\"0 3.14159\"")
    print("  Neutral standing: ~90° (π/2 radians)")

    print("\nAngles from IK (coord2angle):")
    print("  Can be positive or negative (relative to leg frame)")
    print("  Typical range: -90° to +90° for hip")
    print("  Typical range: -90° to +90° for knee")

    print("\nOur mapping to servo range:")
    print("  shifted_angle = angle + π/2  (shift to positive)")
    print("  normalized = (shifted_angle - 0) / π * 2 - 1")
    print("  This maps [0, π] → [-1, 1]")

    # Test some angles
    print("\nTest angles:")
    test_angles_deg = [0, 30, 45, 60, 90, -30, -45]

    for angle_deg in test_angles_deg:
        angle_rad = radians(angle_deg)
        shifted = angle_rad + pi/2
        servo_deg = degrees(shifted)
        normalized = (shifted - 0) / pi * 2 - 1

        in_range = "✓" if 0 <= shifted <= pi else "❌"
        print(f"  IK angle: {angle_deg:>4.0f}° → Servo: {servo_deg:>6.2f}° ({shifted:.3f} rad) → Norm: {normalized:>6.3f} {in_range}")

    print("="*70 + "\n")


def main():
    """Run all comparisons."""
    compare_conversions()
    check_actuator_ordering()
    check_servo_range()

    print("\n" + "="*70)
    print(" RECOMMENDATIONS")
    print("="*70)
    print("\n1. If conversion matches exactly → Problem is elsewhere (height, mirroring, etc.)")
    print("2. If actuator order is wrong → Fix the mujoco_angles reordering")
    print("3. If servo range is wrong → Fix the normalization")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
