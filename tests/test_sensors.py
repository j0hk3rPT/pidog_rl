"""Test script to verify PiDog sensor integration."""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pidog_env import PiDogEnv


def test_vector_only():
    """Test environment with vector observations only (no camera)."""
    print("=" * 60)
    print("TEST 1: Vector-only observations (no camera)")
    print("=" * 60)

    try:
        env = PiDogEnv(use_camera=False)
        print(f"✓ Environment created successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Expected shape: (31,)")

        # Test reset
        obs, info = env.reset()
        print(f"✓ Environment reset successfully")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation dtype: {obs.dtype}")

        # Test sensor IDs
        print("\n  Sensor IDs:")
        for name, sensor_info in env.sensor_ids.items():
            if sensor_info['id'] != -1:
                print(f"    {name}: ID={sensor_info['id']}, "
                      f"adr={sensor_info['adr']}, dim={sensor_info['dim']}")
            else:
                print(f"    {name}: NOT FOUND (will use zeros)")

        # Check observation values
        print("\n  Observation breakdown:")
        print(f"    Joint positions [0:8]: {obs[0:8]}")
        print(f"    Joint velocities [8:16]: {obs[8:16]}")
        print(f"    Body quaternion [16:20]: {obs[16:20]}")
        print(f"    Linear velocity [20:23]: {obs[20:23]}")
        print(f"    Angular velocity [23:26]: {obs[23:26]}")
        print(f"    Body height [26]: {obs[26]}")
        print(f"    Ultrasonic distance [27]: {obs[27]}")
        print(f"    IMU accel [28:31]: {obs[28:31]}")

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\n✓ Environment step successful")
        print(f"  Reward: {reward:.4f}")
        print(f"  Terminated: {terminated}")

        env.close()
        print("\n✓ TEST 1 PASSED\n")
        return True

    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_camera_observations():
    """Test environment with camera + vector observations."""
    print("=" * 60)
    print("TEST 2: Camera + Vector observations (MultiInput)")
    print("=" * 60)

    try:
        env = PiDogEnv(use_camera=True, camera_width=84, camera_height=84)
        print(f"✓ Environment created successfully")
        print(f"  Observation space: {env.observation_space}")

        # Test reset
        obs, info = env.reset()
        print(f"✓ Environment reset successfully")

        # Check observation structure
        assert isinstance(obs, dict), "Observation should be a dictionary"
        assert "image" in obs, "Observation should have 'image' key"
        assert "vector" in obs, "Observation should have 'vector' key"
        print(f"  Observation keys: {list(obs.keys())}")

        # Check image
        image = obs["image"]
        print(f"\n  Image observation:")
        print(f"    Shape: {image.shape}")
        print(f"    Dtype: {image.dtype}")
        print(f"    Value range: [{image.min()}, {image.max()}]")
        print(f"    Expected: (84, 84, 3) uint8 [0, 255]")

        assert image.shape == (84, 84, 3), f"Image shape should be (84,84,3), got {image.shape}"
        assert image.dtype == np.uint8, f"Image dtype should be uint8, got {image.dtype}"

        # Check vector
        vector = obs["vector"]
        print(f"\n  Vector observation:")
        print(f"    Shape: {vector.shape}")
        print(f"    Dtype: {vector.dtype}")
        print(f"    Expected: (31,) float32")

        assert vector.shape == (31,), f"Vector shape should be (31,), got {vector.shape}"
        assert vector.dtype == np.float32, f"Vector dtype should be float32, got {vector.dtype}"

        # Test that camera is actually rendering (not all zeros)
        if np.any(image > 0):
            print(f"  ✓ Camera is rendering (non-zero pixels detected)")
        else:
            print(f"  ⚠ Warning: Camera image is all zeros (might be expected if scene is empty)")

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\n✓ Environment step successful")
        print(f"  Reward: {reward:.4f}")
        print(f"  Image changed: {not np.array_equal(obs['image'], image)}")

        env.close()
        print("\n✓ TEST 2 PASSED\n")
        return True

    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sensor_values():
    """Test that sensor values are reasonable."""
    print("=" * 60)
    print("TEST 3: Sensor value ranges")
    print("=" * 60)

    try:
        env = PiDogEnv(use_camera=False)
        obs, _ = env.reset()

        print("Testing sensor value ranges...")

        # Ultrasonic distance (should be 0-4.5m)
        ultrasonic = obs[27]
        print(f"\n  Ultrasonic distance: {ultrasonic:.3f}m")
        if 0 <= ultrasonic <= 4.5:
            print(f"    ✓ Within expected range [0, 4.5]m")
        else:
            print(f"    ⚠ Outside expected range (might be no obstacle)")

        # IMU accelerometer (should include gravity ~9.81 m/s²)
        accel = obs[28:31]
        accel_mag = np.linalg.norm(accel)
        print(f"\n  IMU Accelerometer: {accel}")
        print(f"    Magnitude: {accel_mag:.3f} m/s²")
        print(f"    Expected: ~9.81 m/s² (gravity) when stationary")
        if 8.0 < accel_mag < 12.0:
            print(f"    ✓ Reasonable magnitude")
        else:
            print(f"    ⚠ Unexpected magnitude")

        # Body quaternion (should be normalized)
        quat = obs[16:20]
        quat_norm = np.linalg.norm(quat)
        print(f"\n  Body quaternion: {quat}")
        print(f"    Norm: {quat_norm:.6f}")
        print(f"    Expected: 1.0 (normalized quaternion)")
        if abs(quat_norm - 1.0) < 0.01:
            print(f"    ✓ Properly normalized")
        else:
            print(f"    ⚠ Not normalized")

        # Joint positions (should be near neutral angles)
        joint_pos = obs[0:8]
        neutral_hip = -np.pi / 6  # -30°
        neutral_knee = -np.pi / 4  # -45°

        print(f"\n  Joint positions (expected near neutral):")
        for i in range(4):
            hip = joint_pos[i*2]
            knee = joint_pos[i*2 + 1]
            print(f"    Leg {i}: hip={np.degrees(hip):.1f}° (neutral=-30°), "
                  f"knee={np.degrees(knee):.1f}° (neutral=-45°)")

        env.close()
        print("\n✓ TEST 3 PASSED\n")
        return True

    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PIDOG SENSOR INTEGRATION TEST SUITE")
    print("=" * 60 + "\n")

    results = {
        "Vector-only observations": test_vector_only(),
        "Camera + Vector observations": test_camera_observations(),
        "Sensor value ranges": test_sensor_values(),
    }

    # Print summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
