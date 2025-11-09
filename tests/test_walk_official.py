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


def leg_ik(y, z):
    """
    Calculate hip and knee angles from foot position using inverse kinematics.

    Based on official PiDog coordinate-to-angle conversion.

    Args:
        y: Forward/backward position (m)
        z: Vertical position (m)

    Returns:
        (hip_angle, knee_angle) in radians
    """
    # Distance from hip to foot
    r = np.sqrt(y**2 + z**2)

    # Clamp to reachable workspace
    r = np.clip(r, abs(LEG_LENGTH - FOOT_LENGTH), LEG_LENGTH + FOOT_LENGTH)

    # Knee angle using law of cosines
    cos_knee = (LEG_LENGTH**2 + FOOT_LENGTH**2 - r**2) / (2 * LEG_LENGTH * FOOT_LENGTH)
    cos_knee = np.clip(cos_knee, -1.0, 1.0)
    knee_angle = np.arccos(cos_knee)

    # Hip angle
    alpha = np.arctan2(y, -z)  # Angle to target
    cos_beta = (r**2 + LEG_LENGTH**2 - FOOT_LENGTH**2) / (2 * r * LEG_LENGTH)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)  # Angle from upper leg to target

    hip_angle = alpha + beta

    # Convert to servo range (0 to π, with π/2 as neutral)
    # Adjust to match PiDog servo orientation
    hip_servo = np.pi/2 - hip_angle
    knee_servo = np.pi - knee_angle  # Knee servo is reversed

    return hip_servo, knee_servo


def trotting_gait_official(t, frequency=1.0, forward=True):
    """
    Official PiDog trotting gait pattern.

    Diagonal pairs move together:
    - Pair 1: Front-left (leg 3) + Back-right (leg 1)
    - Pair 2: Front-right (leg 2) + Back-left (leg 4)

    Args:
        t: Time in seconds
        frequency: Gait frequency in Hz
        forward: True for forward, False for backward

    Returns:
        Array of 8 joint positions [back_right_hip, back_right_knee, ...]
    """
    # Phase for diagonal pair switching
    phase = (2 * np.pi * frequency * t) % (2 * np.pi)

    # Determine which pair is stepping
    step_phase = phase / (2 * np.pi)  # 0 to 1

    # Direction
    direction = 1 if forward else -1

    # Calculate positions for each leg
    positions = []

    # Leg mapping: 1=back_right, 2=front_right, 3=back_left, 4=front_left
    # But in MuJoCo: [back_right, front_right, back_left, front_left]
    for leg_idx in range(4):
        # Determine if this leg is in pair 1 or pair 2
        # Pair 1 (legs 1,4): back_right (0) and front_left (3)
        # Pair 2 (legs 2,3): front_right (1) and back_left (2)
        in_pair_1 = (leg_idx == 0) or (leg_idx == 3)  # back_right or front_left

        if in_pair_1:
            # Stepping when phase is 0 to π
            theta = phase if phase < np.pi else 0
        else:
            # Stepping when phase is π to 2π
            theta = (phase - np.pi) if phase >= np.pi else 0

        # Y position (forward/backward) using cosine
        if (in_pair_1 and phase < np.pi) or (not in_pair_1 and phase >= np.pi):
            # Stepping leg
            y_offset = TROT_STEP_WIDTH * (np.cos(theta) - direction) / 2 * direction
            z_pos = BODY_HEIGHT - TROT_STEP_HEIGHT * (theta / np.pi)
        else:
            # Standing leg (moves opposite direction)
            y_offset = -TROT_STEP_WIDTH * direction / 2
            z_pos = BODY_HEIGHT

        # Calculate servo angles
        hip_angle, knee_angle = leg_ik(y_offset + TROT_COG_OFFSET, z_pos)
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

    # Calculate neutral standing position
    neutral_y = 0.0
    neutral_z = BODY_HEIGHT
    hip_neutral, knee_neutral = leg_ik(neutral_y, neutral_z)

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
