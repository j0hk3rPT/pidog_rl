"""
Test PiDog walking in MuJoCo viewer.

This script loads the PiDog model and makes it walk using a simple trotting gait.
The gait is based on realistic servo constraints:
- Range: 0-180° (0 to π radians)
- Max speed: 7.0 rad/s
- Max torque: 0.137 Nm
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path


def trotting_gait(t, amplitude=0.6, frequency=1.5):
    """
    Generate trotting gait pattern.

    Trotting: diagonal legs move together
    - Front-left with back-right
    - Front-right with back-left

    Args:
        t: Time in seconds
        amplitude: Movement amplitude in radians
        frequency: Gait frequency in Hz

    Returns:
        Array of 8 joint positions [back_right_hip, back_right_knee, ...]
    """
    phase = 2 * np.pi * frequency * t

    # Neutral positions - adjusted for proper standing
    hip_neutral = np.pi / 2   # 90° for hips
    knee_neutral = np.pi / 4  # 45° for knees (more extended than 90°)

    # Hip and knee offsets for walking
    hip_offset = amplitude * np.sin(phase)
    knee_offset = amplitude * np.cos(phase)

    # Diagonal pairs move together
    # Back-right and front-left (pair 1)
    back_right_hip = hip_neutral + hip_offset
    back_right_knee = knee_neutral + knee_offset

    front_left_hip = hip_neutral + hip_offset
    front_left_knee = knee_neutral + knee_offset

    # Front-right and back-left (pair 2, opposite phase)
    front_right_hip = hip_neutral - hip_offset
    front_right_knee = knee_neutral - knee_offset

    back_left_hip = hip_neutral - hip_offset
    back_left_knee = knee_neutral - knee_offset

    # Return in order: back_right, front_right, back_left, front_left
    # Each with [hip, knee]
    return np.array([
        back_right_hip, back_right_knee,
        front_right_hip, front_right_knee,
        back_left_hip, back_left_knee,
        front_left_hip, front_left_knee,
    ])


def walking_gait(t, amplitude=0.5, frequency=1.0):
    """
    Generate walking gait pattern.

    Walking: one foot at a time
    - More stable but slower than trotting

    Args:
        t: Time in seconds
        amplitude: Movement amplitude in radians
        frequency: Gait frequency in Hz

    Returns:
        Array of 8 joint positions
    """
    phase = 2 * np.pi * frequency * t
    hip_neutral = np.pi / 2   # 90° for hips
    knee_neutral = np.pi / 4  # 45° for knees (more extended)

    # Create phase offsets for each leg (sequential)
    phases = [
        phase,                  # Back right
        phase + np.pi/2,       # Front right
        phase + np.pi,         # Back left
        phase + 3*np.pi/2,     # Front left
    ]

    positions = []
    for leg_phase in phases:
        hip = hip_neutral + amplitude * np.sin(leg_phase)
        knee = knee_neutral + amplitude * np.cos(leg_phase)
        positions.extend([hip, knee])

    return np.array(positions)


def main():
    """Run PiDog walking simulation."""
    # Get model path
    model_path = Path(__file__).parent.parent / "model" / "pidog.xml"

    print("=" * 60)
    print("PiDog Walking Test")
    print("=" * 60)
    print(f"Loading model: {model_path}")

    # Load model
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure the model/pidog_actuators.xml file exists!")
        return

    print(f"Model loaded successfully!")
    print(f"Number of actuators: {model.nu}")
    print(f"Number of joints: {model.njnt}")
    print(f"Timestep: {model.opt.timestep}")

    # Print actuator names
    print("\nActuators:")
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i}: {actuator_name}")

    # Initialize robot position
    # Set body height
    data.qpos[2] = 0.15  # Height of base

    # Set all joints to neutral positions
    hip_neutral = np.pi / 2   # 90° for hips
    knee_neutral = np.pi / 4  # 45° for knees

    # Set hip joints (every other joint starting at 7)
    for i in range(7, 15, 2):  # Hip joints: 7, 9, 11, 13
        data.qpos[i] = hip_neutral

    # Set knee joints (every other joint starting at 8)
    for i in range(8, 15, 2):  # Knee joints: 8, 10, 12, 14
        data.qpos[i] = knee_neutral

    # Forward kinematics to settle
    mujoco.mj_forward(model, data)

    print("\nStarting simulation...")
    print("Choose gait:")
    print("  1) Trotting (diagonal legs together)")
    print("  2) Walking (one foot at a time)")
    print("  3) Standing in place")

    # For this demo, use trotting
    gait_mode = 1

    # Launch viewer
    print("\nLaunching viewer...")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom")
    print("  - Ctrl+Click: Pan")
    print("  - Esc: Exit")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera for better view
        viewer.cam.distance = 1.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

        start_time = time.time()
        step_count = 0

        print("Simulation running... Press Ctrl+C to stop")

        try:
            while viewer.is_running():
                # Calculate current time
                current_time = time.time() - start_time

                # Generate gait pattern
                if gait_mode == 1:
                    target_positions = trotting_gait(current_time)
                elif gait_mode == 2:
                    target_positions = walking_gait(current_time)
                else:
                    target_positions = np.full(8, np.pi / 2)

                # Apply to first 8 actuators (legs)
                data.ctrl[:8] = target_positions

                # Step simulation
                mujoco.mj_step(model, data)

                # Update viewer
                viewer.sync()

                step_count += 1

                # Print stats every 1000 steps
                if step_count % 1000 == 0:
                    body_height = data.qpos[2]
                    forward_vel = data.qvel[0]
                    print(f"Time: {current_time:.1f}s | "
                          f"Height: {body_height:.3f}m | "
                          f"Forward vel: {forward_vel:.3f}m/s")

                # Small delay to match real-time
                time.sleep(model.opt.timestep)

        except KeyboardInterrupt:
            print("\nSimulation stopped by user")

    print(f"\nSimulation completed!")
    print(f"Total steps: {step_count}")
    print(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
