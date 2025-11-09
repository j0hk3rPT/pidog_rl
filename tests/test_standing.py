"""
Test different neutral positions to find the correct standing pose.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path


def test_pose(model, data, viewer, hip_angle, knee_angle, duration=5.0):
    """Test a specific pose."""
    print(f"\nTesting: Hip={np.degrees(hip_angle):.1f}°, Knee={np.degrees(knee_angle):.1f}°")

    positions = np.array([
        hip_angle, knee_angle,    # back_right
        hip_angle, knee_angle,    # front_right
        hip_angle, knee_angle,    # back_left
        hip_angle, knee_angle,    # front_left
    ])

    start_time = time.time()
    while viewer.is_running() and (time.time() - start_time) < duration:
        data.ctrl[:8] = positions
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)


def main():
    """Test different standing poses."""
    model_path = Path(__file__).parent.parent / "model" / "pidog.xml"

    print("=" * 60)
    print("PiDog Standing Pose Test")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Initialize
    data.qpos[2] = 0.15  # Height

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

        hip_neutral = np.pi / 2  # 90° - keep this

        # Test different knee angles
        knee_angles = [
            0.0,           # 0° - fully bent one way
            np.pi / 4,     # 45°
            np.pi / 3,     # 60°
            np.pi / 2,     # 90° - current (elbows closed)
            2 * np.pi / 3, # 120°
            3 * np.pi / 4, # 135°
            np.pi,         # 180° - fully bent other way
        ]

        print("\nTesting different knee angles (hip fixed at 90°)...")
        print("Watch which one gives a natural standing pose")
        print()

        for knee_angle in knee_angles:
            test_pose(model, data, viewer, hip_neutral, knee_angle, duration=3.0)

        print("\n" + "=" * 60)
        print("Test complete! Which angle looked best?")
        print("Update test_walk.py with the best knee angle value")


if __name__ == "__main__":
    main()
