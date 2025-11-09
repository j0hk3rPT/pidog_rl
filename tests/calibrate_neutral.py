"""
Calibrate neutral positions for PiDog legs.

Use keyboard controls to adjust joint positions and find the standing pose.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path


def main():
    """Run calibration."""
    model_path = Path(__file__).parent.parent / "model" / "pidog.xml"

    print("=" * 60)
    print("PiDog Neutral Position Calibration")
    print("=" * 60)
    print(f"Loading model: {model_path}")

    # Load model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    print(f"Model loaded successfully!")
    print(f"Number of actuators: {model.nu}")

    # Initial positions (try different values)
    # Format: [back_right_hip, back_right_knee, front_right_hip, front_right_knee,
    #          back_left_hip, back_left_knee, front_left_hip, front_left_knee]

    # Try different neutral positions
    # Option 1: All at π/2 (90°) - current
    # Option 2: Hips at π/2, knees extended (0 or π)
    # Option 3: Custom values

    hip_angle = np.pi / 2  # 90° - neutral
    knee_angle = 0.3       # Try different values: 0, π/4, π/2, 3π/4, π

    positions = np.array([
        hip_angle, knee_angle,    # back_right
        hip_angle, knee_angle,    # front_right
        hip_angle, knee_angle,    # back_left
        hip_angle, knee_angle,    # front_left
    ])

    # Initialize
    data.qpos[2] = 0.15  # Height
    mujoco.mj_forward(model, data)

    print("\nStarting calibration...")
    print("\nControls:")
    print("  1/2: Adjust hip angle (all legs)")
    print("  3/4: Adjust knee angle (all legs)")
    print("  q/w: Adjust hip angle ±0.1 rad")
    print("  a/s: Adjust knee angle ±0.1 rad")
    print("  SPACE: Print current angles")
    print("  ESC: Exit")
    print()
    print(f"Starting: Hip={hip_angle:.3f} rad ({np.degrees(hip_angle):.1f}°), "
          f"Knee={knee_angle:.3f} rad ({np.degrees(knee_angle):.1f}°)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

        last_print_time = time.time()

        try:
            while viewer.is_running():
                # Apply current positions
                data.ctrl[:8] = positions

                # Step simulation
                mujoco.mj_step(model, data)
                viewer.sync()

                # Print status every 2 seconds
                if time.time() - last_print_time > 2.0:
                    print(f"Current: Hip={positions[0]:.3f} rad ({np.degrees(positions[0]):.1f}°), "
                          f"Knee={positions[1]:.3f} rad ({np.degrees(positions[1]):.1f}°)")
                    last_print_time = time.time()

                time.sleep(model.opt.timestep)

        except KeyboardInterrupt:
            print("\nCalibration stopped")

    print(f"\nFinal values:")
    print(f"  Hip angle: {positions[0]:.4f} rad ({np.degrees(positions[0]):.1f}°)")
    print(f"  Knee angle: {positions[1]:.4f} rad ({np.degrees(positions[1]):.1f}°)")


if __name__ == "__main__":
    main()
