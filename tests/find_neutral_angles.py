"""
Find the correct neutral standing angles by testing different combinations.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path


def test_angles(model, data, viewer, hip_angle, knee_angle, duration=2.0):
    """Test a specific angle combination."""
    print(f"\nTesting Hip={np.degrees(hip_angle):.0f}°, Knee={np.degrees(knee_angle):.0f}°")

    # FULL RESET - respawn the robot
    # Reset all positions and velocities to zero
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    # Set body position (x, y, z)
    data.qpos[0] = 0.0  # X position
    data.qpos[1] = 0.0  # Y position
    data.qpos[2] = 0.10  # Z position (height)

    # Set body orientation (quaternion: w, x, y, z)
    data.qpos[3] = 1.0  # w
    data.qpos[4] = 0.0  # x
    data.qpos[5] = 0.0  # y
    data.qpos[6] = 0.0  # z

    # Set all leg joints to test angles
    for i in range(4):
        data.qpos[7 + i*2] = hip_angle      # Hip
        data.qpos[7 + i*2 + 1] = knee_angle  # Knee

    # Reset control signals
    data.ctrl[:] = 0.0

    # Forward kinematics to update positions
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

    # Only print if potentially good
    if final_height > 0.05:
        print(f"  Final height: {final_height:.3f}m (started at 0.10m)")
        print(f"  ✓ STABLE - Good standing pose!")

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

    # Test different combinations - EXPANDED SEARCH
    # Hip range: 0° to 180°
    # Knee range: 0° to 180°

    test_cases = []

    # Test all combinations systematically
    print("Testing Hip angles: 0° to 180° in 15° steps")
    print("Testing Knee angles: 0° to 180° in 15° steps")
    print()

    for hip_deg in range(0, 181, 15):
        for knee_deg in range(0, 181, 15):
            test_cases.append((hip_deg, knee_deg))

    print(f"Total test cases: {len(test_cases)}")
    print("This will take ~5 minutes...\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

        results = []

        for hip_deg, knee_deg in test_cases:
            hip_rad = np.radians(hip_deg)
            knee_rad = np.radians(knee_deg)

            height = test_angles(model, data, viewer, hip_rad, knee_rad, duration=2.0)
            results.append((hip_deg, knee_deg, height))

        print("\n" + "=" * 60)
        print("Results Summary:")
        print("=" * 60)

        # Sort by height (best first)
        results.sort(key=lambda x: x[2], reverse=True)

        # Show only good results (> 0.05m) and top 20
        good_results = [r for r in results if r[2] > 0.05]

        if good_results:
            print(f"\nGOOD RESULTS (height > 0.05m):")
            for hip_deg, knee_deg, height in good_results[:20]:
                print(f"✓ Hip={hip_deg:3d}°, Knee={knee_deg:3d}° → Height={height:.3f}m")
        else:
            print("\nNO GOOD RESULTS FOUND!")
            print("Showing top 10 best (even though all collapsed):")
            for hip_deg, knee_deg, height in results[:10]:
                print(f"✗ Hip={hip_deg:3d}°, Knee={knee_deg:3d}° → Height={height:.3f}m")

        print("\n" + "=" * 60)
        best = results[0]
        status = "✓ GOOD" if best[2] > 0.05 else "✗ COLLAPSED"
        print(f"{status} BEST: Hip={best[0]}°, Knee={best[1]}° → Height={best[2]:.3f}m")
        print("=" * 60)


if __name__ == "__main__":
    main()
