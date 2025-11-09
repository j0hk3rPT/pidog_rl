"""Visualize PiDog model with all sensors in MuJoCo viewer."""

import mujoco
import mujoco.viewer
from pathlib import Path
import time

def main():
    """Load and visualize the PiDog model."""

    # Load model
    model_path = Path(__file__).parent / "model" / "pidog.xml"
    print(f"Loading model from: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Print model information
    print("\n" + "=" * 60)
    print("PIDOG MODEL INFORMATION")
    print("=" * 60)

    print(f"\nActuators (servos): {model.nu}")
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i}: {actuator_name}")

    print(f"\nSensors: {model.nsensor}")
    for i in range(model.nsensor):
        sensor_id = i
        sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
        sensor_type = model.sensor_type[i]
        sensor_dim = model.sensor_dim[i]
        print(f"  {i}: {sensor_name} (type={sensor_type}, dim={sensor_dim})")

    print(f"\nCameras: {model.ncam}")
    for i in range(model.ncam):
        cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        print(f"  {i}: {cam_name}")

    print(f"\nSites (sensor mount points): {model.nsite}")
    for i in range(model.nsite):
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        print(f"  {i}: {site_name}")

    print(f"\nTotal DOF: {model.nv}")
    print(f"Total bodies: {model.nbody}")
    print(f"Total geoms: {model.ngeom}")

    print("\n" + "=" * 60)
    print("Opening MuJoCo Viewer...")
    print("=" * 60)
    print("\nViewer Controls:")
    print("  - Left mouse: Rotate view")
    print("  - Right mouse: Zoom")
    print("  - Middle mouse: Pan")
    print("  - Double-click: Select body")
    print("  - Ctrl+Right-click: Apply force")
    print("  - Space: Pause/Resume simulation")
    print("  - Page Up/Down: Speed up/down")
    print("  - F5: Show/hide sensor data")
    print("  - Esc: Close viewer")
    print("\n" + "=" * 60)

    # Set initial pose to neutral standing
    data.qpos[2] = 0.14  # Height

    # Set neutral leg angles
    neutral_hip = -3.14159/6  # -30°
    neutral_knee = -3.14159/4  # -45°

    for i in range(4):  # 4 legs
        data.qpos[7 + i*2] = neutral_hip      # Hip
        data.qpos[7 + i*2 + 1] = neutral_knee  # Knee

    # Forward kinematics to update visualization
    mujoco.mj_forward(model, data)

    # Launch passive viewer
    print("\nLaunching viewer... (close window to exit)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Enable site visualization (sensor mounting points)
        viewer.opt.sitegroup[0] = True
        viewer.opt.sitegroup[1] = True
        viewer.opt.sitegroup[2] = True

        # Run simulation
        while viewer.is_running():
            step_start = time.time()

            # Step simulation
            mujoco.mj_step(model, data)

            # Sync viewer
            viewer.sync()

            # Crude time keeping to run at realtime
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("\nViewer closed.")


if __name__ == "__main__":
    main()
