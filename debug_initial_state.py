"""Debug initial state to see why robot falls immediately."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from pidog_env import PiDogEnv

env = PiDogEnv(use_camera=False)
obs, _ = env.reset()

print("Initial State After Reset:")
print(f"  Body height: {env.data.qpos[2]:.4f} m")
print(f"  Body quat: {env.data.qpos[3:7]}")  # [w, x, y, z]
print(f"  Body quat (w): {env.data.qpos[3]:.4f}")

# Calculate roll and pitch (MuJoCo format: w, x, y, z)
quat = env.data.qpos[3:7]
w, x, y, z = quat
roll = np.arctan2(2.0 * (w * x + y * z),
                  1.0 - 2.0 * (x**2 + y**2))
pitch = np.arcsin(2.0 * (w * y - z * x))

print(f"  Roll: {np.degrees(roll):.2f}°")
print(f"  Pitch: {np.degrees(pitch):.2f}°")

# Check termination
terminated = env._is_terminated()
print(f"\n  Terminated immediately: {terminated}")

if terminated:
    print("\nTermination reasons:")
    if env.data.qpos[2] < 0.05:
        print("  - Body height < 0.05m")
    if quat[0] < 0.5:  # w is quat[0] in MuJoCo format
        print(f"  - Quaternion w < 0.5 ({quat[0]:.4f})")
    if abs(roll) > np.pi/3.6 or abs(pitch) > np.pi/3.6:
        print(f"  - Roll/pitch > 50° (roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°)")

# Try one step with zero action
print("\nTrying one step with zero action...")
action = np.zeros(8)
obs, reward, terminated, truncated, info = env.step(action)

print(f"After 1 step:")
print(f"  Body height: {env.data.qpos[2]:.4f} m")
print(f"  Body quat: {env.data.qpos[3:7]}")
print(f"  Body quat (w): {env.data.qpos[3]:.4f}")

quat = env.data.qpos[3:7]
w, x, y, z = quat
roll = np.arctan2(2.0 * (w * x + y * z),
                  1.0 - 2.0 * (x**2 + y**2))
pitch = np.arcsin(2.0 * (w * y - z * x))

print(f"  Roll: {np.degrees(roll):.2f}°")
print(f"  Pitch: {np.degrees(pitch):.2f}°")
print(f"  Reward: {reward:.2f}")
print(f"  Terminated: {terminated}")

if terminated:
    print("\nTermination reasons after step:")
    if env.data.qpos[2] < 0.05:
        print("  - Body height < 0.05m")
    if quat[0] < 0.5:  # w is quat[0] in MuJoCo format
        print(f"  - Quaternion w < 0.5 ({quat[0]:.4f})")
    if abs(roll) > np.pi/3.6 or abs(pitch) > np.pi/3.6:
        print(f"  - Roll/pitch > 50° (roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°)")

env.close()
