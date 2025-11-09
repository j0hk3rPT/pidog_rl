#!/usr/bin/env python3
"""
Simple viewer launcher.

Opens MuJoCo viewer with the robot at neutral position.
You can use the UI to manually move each joint and identify the mapping.
"""

import mujoco
import mujoco.viewer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pidog_env import PiDogEnv


def launch_viewer():
    """Launch MuJoCo viewer."""
    print("\n" + "="*70)
    print(" MUJOCO VIEWER - LEG IDENTIFICATION")
    print("="*70)
    print()
    print("MuJoCo actuator order (in the UI):")
    print("  [0] back_right_hip")
    print("  [1] back_right_knee")
    print("  [2] front_right_hip")
    print("  [3] front_right_knee")
    print("  [4] back_left_hip")
    print("  [5] back_left_knee")
    print("  [6] front_left_hip")
    print("  [7] front_left_knee")
    print()
    print("Controls:")
    print("  - Double-click on robot to select actuators")
    print("  - Use sliders in UI to move each joint")
    print("  - Watch which physical leg moves")
    print()
    print("="*70)
    print()

    env = PiDogEnv(use_camera=False)
    obs, _ = env.reset()

    print("Launching viewer...")
    print("Press ESC or close window to exit.")
    print()

    # Launch passive viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            # Just step the simulation to keep it alive
            mujoco.mj_step(env.model, env.data)
            viewer.sync()

    env.close()

    print("\n" + "="*70)
    print(" VIEWER CLOSED")
    print("="*70)
    print()
    print("Please tell me which physical leg each actuator controls:")
    print("  actuator 0 → ?")
    print("  actuator 2 → ?")
    print("  actuator 4 → ?")
    print("  actuator 6 → ?")
    print()


if __name__ == "__main__":
    launch_viewer()
