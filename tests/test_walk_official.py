"""
Test PiDog walking with official SunFounder PiDog parameters.

Based on: https://github.com/sunfounder/pidog/blob/master/pidog/trot.py
          https://github.com/sunfounder/pidog/blob/master/pidog/walk.py
          https://github.com/sunfounder/pidog/blob/master/pidog/pidog.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path


# Official PiDog physical dimensions (in meters)
LEG_LENGTH = 0.042  # Upper leg: 42mm
FOOT_LENGTH = 0.076  # Lower leg: 76mm
BODY_HEIGHT = 0.080  # Z_ORIGIN: 80mm

# Trotting gait parameters (official)
TROT_STEP_HEIGHT = 0.020  # 20mm
TROT_STEP_WIDTH = 0.100   # 100mm
TROT_COG_OFFSET = -0.017  # Center of gravity: -17mm

# Walking gait parameters (official)
WALK_STEP_HEIGHT = 0.020  # 20mm
WALK_STEP_WIDTH = 0.080   # 80mm
WALK_COG_OFFSET = -0.015  # Center of gravity: -15mm


def calculate_leg_angles(y_offset, z_height, is_stepping):
    """
    Calculate hip and knee angles for a leg position.

    This is a simplified approach that directly maps to observed working angles.

    Args:
        y_offset: Forward/backward offset from neutral (m)
        z_height: Height of the body (m)
        is_stepping: Whether this leg is currently stepping (lifted)

    Returns:
        (hip_angle, knee_angle) in radians
    """
    # Neutral standing angles (found by systematic search)
    hip_neutral = -np.pi / 6     # -30°
    knee_neutral = -np.pi / 4    # -45°

    # Map y_offset to hip angle change
    # Forward movement rotates hip backward, backward movement rotates hip forward
    hip_amplitude = 0.3  # radians (~17°)
    hip_angle = hip_neutral + (y_offset / TROT_STEP_WIDTH) * hip_amplitude

    # Knee angle changes to lift the leg when stepping
    knee_amplitude = 0.4  # radians (~23°)
    if is_stepping:
        # Lift leg by bending knee more
        step_factor = 1.0  # Full step
        knee_angle = knee_neutral + knee_amplitude * step_factor
    else:
        # Standing leg stays at neutral
        knee_angle = knee_neutral

    return hip_angle, knee_angle


def trotting_gait_official(t, frequency=0.5, forward=True):
    """
    Official PiDog trotting gait pattern.

    Diagonal pairs move together:
    - Pair 1: Back-right (0) + Front-left (3)
    - Pair 2: Front-right (1) + Back-left (2)

    Args:
        t: Time in seconds
        frequency: Gait frequency in Hz (default 0.5 = 2 seconds per cycle)
        forward: True for forward, False for backward

    Returns:
        Array of 8 joint positions [back_right_hip, back_right_knee, ...]
    """
    # Phase for diagonal pair switching (0 to 2π)
    phase = (2 * np.pi * frequency * t) % (2 * np.pi)

    # Direction
    direction = 1 if forward else -1

    # Calculate positions for each leg
    positions = []

    # Leg order in MuJoCo: [back_right, front_right, back_left, front_left]
    for leg_idx in range(4):
        # Determine if this leg is in pair 1 or pair 2
        # Pair 1: back_right (0) and front_left (3)
        # Pair 2: front_right (1) and back_left (2)
        in_pair_1 = (leg_idx == 0) or (leg_idx == 3)

        # Determine if this leg is currently stepping (lifted)
        if in_pair_1:
            is_stepping = phase < np.pi
            leg_phase = phase if is_stepping else 0
        else:
            is_stepping = phase >= np.pi
            leg_phase = (phase - np.pi) if is_stepping else 0

        # Y offset (forward/backward) using cosine curve
        if is_stepping:
            # Stepping leg follows cosine curve
            y_offset = TROT_STEP_WIDTH * (np.cos(leg_phase) - direction) / 2 * direction
        else:
            # Standing leg is at opposite extreme
            y_offset = -TROT_STEP_WIDTH * direction / 2

        # Calculate servo angles
        hip_angle, knee_angle = calculate_leg_angles(y_offset, BODY_HEIGHT, is_stepping)
        positions.extend([hip_angle, knee_angle])

    return np.array(positions)


def main():
    """Run PiDog with official movement parameters."""
    model_path = Path(__file__).parent.parent / "model" / "pidog.xml"

    print("=" * 60)
    print("PiDog Official Gait Test")
    print("=" * 60)
    print(f"Loading model: {model_path}")
    print(f"\nOfficial Parameters:")
    print(f"  Leg length: {LEG_LENGTH*1000:.0f}mm")
    print(f"  Foot length: {FOOT_LENGTH*1000:.0f}mm")
    print(f"  Body height: {BODY_HEIGHT*1000:.0f}mm")
    print(f"  Step height: {TROT_STEP_HEIGHT*1000:.0f}mm")
    print(f"  Step width: {TROT_STEP_WIDTH*1000:.0f}mm")

    # Load model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    print(f"\nModel loaded!")
    print(f"Actuators: {model.nu}, Joints: {model.njnt}")

    # Initialize at standing position
    data.qpos[2] = 0.15  # Initial height

    # Neutral standing angles (found by systematic search)
    hip_neutral = -np.pi / 6   # -30°
    knee_neutral = -np.pi / 4  # -45°

    print(f"\nNeutral angles:")
    print(f"  Hip: {np.degrees(hip_neutral):.1f}°")
    print(f"  Knee: {np.degrees(knee_neutral):.1f}°")

    # Set all legs to neutral
    for i in range(4):
        data.qpos[7 + i*2] = hip_neutral      # Hip
        data.qpos[7 + i*2 + 1] = knee_neutral  # Knee

    mujoco.mj_forward(model, data)

    print("\nStarting simulation with official PiDog trotting gait...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

        start_time = time.time()
        step_count = 0

        try:
            while viewer.is_running():
                current_time = time.time() - start_time

                # Generate official trotting gait
                target_positions = trotting_gait_official(current_time, frequency=0.5)

                # Apply to leg actuators
                data.ctrl[:8] = target_positions

                # Step simulation
                mujoco.mj_step(model, data)
                viewer.sync()

                step_count += 1

                # Print stats
                if step_count % 1000 == 0:
                    height = data.qpos[2]
                    vel = data.qvel[0]
                    print(f"t={current_time:.1f}s | h={height:.3f}m | v={vel:.3f}m/s")

                time.sleep(model.opt.timestep)

        except KeyboardInterrupt:
            print("\nStopped by user")

    print(f"\nTotal: {step_count} steps, {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
