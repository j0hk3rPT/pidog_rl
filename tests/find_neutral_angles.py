"""
Find the correct neutral standing angles by testing different combinations.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path


def test_angles(model, data, viewer, hip_angle, knee_angle, duration=5.0):
    """Test a specific angle combination."""
    print(f"\nTesting Hip={np.degrees(hip_angle):.0f}°, Knee={np.degrees(knee_angle):.0f}°")

    # Set all legs to these angles
    for i in range(4):
        data.qpos[7 + i*2] = hip_angle      # Hip
        data.qpos[7 + i*2 + 1] = knee_angle  # Knee

    # Set body to reasonable height
    data.qpos[2] = 0.10

    mujoco.mj_forward(model, data)

    # Run for duration
    start = time.time()
    step_count = 0

    while viewer.is_running() and (time.time() - start) < duration:
        # Apply same angles to control
        for i in range(4):
            data.ctrl[i*2] = hip_angle
            data.ctrl[i*2 + 1] = knee_angle

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
        step_count += 1

    # Report final state
    final_height = data.qpos[2]
    print(f"  Final height: {final_height:.3f}m (started at 0.10m)")

    if final_height > 0.05:
        print(f"  ✓ STABLE - Good standing pose!")
    else:
        print(f"  ✗ Collapsed to ground")

    return final_height


def main():
    """Find neutral angles."""
    model_path = Path(__file__).parent.parent / "model" / "pidog.xml"

    print("=" * 60)
    print("Finding Neutral Standing Angles")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    print(f"\nTesting different angle combinations...")
    print(f"Looking for stable standing height > 0.05m\n")

    # Test different combinations
    # Hip range: 30° to 150°
    # Knee range: 0° to 90°

    test_cases = [
        # (hip_deg, knee_deg)
        (90, 0),    # Hip 90°, Knee straight
        (90, 30),   # Hip 90°, Knee 30°
        (90, 45),   # Hip 90°, Knee 45° (current)
        (90, 60),   # Hip 90°, Knee 60°
        (90, 90),   # Hip 90°, Knee 90°
        (60, 45),   # Hip 60°, Knee 45°
        (120, 45),  # Hip 120°, Knee 45°
        (45, 45),   # Hip 45°, Knee 45°
        (135, 45),  # Hip 135°, Knee 45°
        (90, 20),   # Hip 90°, Knee 20°
        (90, 10),   # Hip 90°, Knee 10°
    ]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

        results = []

        for hip_deg, knee_deg in test_cases:
            hip_rad = np.radians(hip_deg)
            knee_rad = np.radians(knee_deg)

            height = test_angles(model, data, viewer, hip_rad, knee_rad, duration=3.0)
            results.append((hip_deg, knee_deg, height))

        print("\n" + "=" * 60)
        print("Results Summary:")
        print("=" * 60)

        # Sort by height (best first)
        results.sort(key=lambda x: x[2], reverse=True)

        for hip_deg, knee_deg, height in results:
            status = "✓ GOOD" if height > 0.05 else "✗ BAD"
            print(f"{status} Hip={hip_deg:3d}°, Knee={knee_deg:3d}° → Height={height:.3f}m")

        print("\n" + "=" * 60)
        best = results[0]
        print(f"BEST: Hip={best[0]}°, Knee={best[1]}° → Height={best[2]:.3f}m")
        print("=" * 60)


if __name__ == "__main__":
    main()
