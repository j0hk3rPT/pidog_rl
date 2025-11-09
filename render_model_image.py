"""Render PiDog model as images to visualize sensors."""

import mujoco
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    """Render the PiDog model from different angles."""

    # Load model
    model_path = Path(__file__).parent / "model" / "pidog.xml"
    print(f"Loading model from: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Set initial pose
    data.qpos[2] = 0.14  # Height
    neutral_hip = -np.pi/6
    neutral_knee = -np.pi/4

    for i in range(4):
        data.qpos[7 + i*2] = neutral_hip
        data.qpos[7 + i*2 + 1] = neutral_knee

    mujoco.mj_forward(model, data)

    # Create renderer
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Render from different camera angles
    output_dir = Path(__file__).parent / "outputs" / "model_renders"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nRendering model from different views...")

    # View 1: From fixed camera
    renderer.update_scene(data, camera="fixed")
    pixels = renderer.render()
    img = Image.fromarray(pixels)
    img.save(output_dir / "pidog_view_fixed.png")
    print(f"  ✓ Saved: {output_dir / 'pidog_view_fixed.png'}")

    # View 2: From vil_camera
    renderer.update_scene(data, camera="vil_camera")
    pixels = renderer.render()
    img = Image.fromarray(pixels)
    img.save(output_dir / "pidog_view_vil.png")
    print(f"  ✓ Saved: {output_dir / 'pidog_view_vil.png'}")

    # View 3: From PiDog's own camera (what the robot sees)
    renderer.update_scene(data, camera="pidog_camera")
    pixels = renderer.render()
    img = Image.fromarray(pixels)
    img.save(output_dir / "pidog_camera_view.png")
    print(f"  ✓ Saved: {output_dir / 'pidog_camera_view.png'}")

    print(f"\n✓ All renders saved to: {output_dir}")
    print("\nModel Summary:")
    print(f"  - Actuators: {model.nu}")
    print(f"  - Sensors: {model.nsensor} (ultrasonic, imu_accel, imu_gyro)")
    print(f"  - Cameras: {model.ncam}")
    print(f"  - Sensor sites: {model.nsite}")
    print(f"  - Total DOF: {model.nv}")


if __name__ == "__main__":
    main()
