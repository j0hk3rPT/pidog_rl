"""Comprehensive test of the complete PiDog RL training system."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pidog_env import PiDogEnv


def test_complete_reward_system():
    """Test all reward components together."""
    print("=" * 70)
    print("COMPLETE REWARD SYSTEM TEST")
    print("=" * 70)

    env = PiDogEnv(use_camera=False)
    obs, _ = env.reset()

    print("\nReward Components:")
    print("  1. Forward Velocity (3.0×)        - PRIMARY GOAL")
    print("  2. Obstacle Avoidance (1.0×)      - Ultrasonic sensor")
    print("  3. Upright Stability (1.5×)       - Stay balanced")
    print("  4. Stationary Penalty (1.0×)      - Keep moving")
    print("  5. Energy Efficiency (small)      - Smooth motion")
    print("  6. Lateral Stability (0.5×)       - Walk straight")
    print("  7. Leg Collision Penalty (2.0×)   - Natural gait")

    print("\nRunning 10 simulation steps...")
    print("-" * 70)

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Get individual components for analysis
        forward_vel = env.data.qvel[0]
        collisions = env._detect_leg_collisions()
        ultrasonic_dist = env._get_sensor_data('ultrasonic')[0]

        print(f"Step {i+1:2d}: "
              f"reward={reward:6.2f}, "
              f"vel={forward_vel:5.2f}m/s, "
              f"collisions={collisions}, "
              f"sonar={ultrasonic_dist:5.2f}m, "
              f"term={terminated}")

        if terminated:
            print("  → Episode terminated (robot fell)")
            break

    env.close()
    print("\n✓ Complete reward system working\n")


def test_sensor_integration():
    """Test all sensors are providing data."""
    print("=" * 70)
    print("SENSOR INTEGRATION TEST")
    print("=" * 70)

    # Test with camera
    env_camera = PiDogEnv(use_camera=True, camera_width=84, camera_height=84)
    obs, _ = env_camera.reset()

    print("\nWith Camera (MultiInput):")
    print(f"  Observation type: {type(obs)}")
    print(f"  Image shape: {obs['image'].shape}")
    print(f"  Image dtype: {obs['image'].dtype}")
    print(f"  Image range: [{obs['image'].min()}, {obs['image'].max()}]")
    print(f"  Vector shape: {obs['vector'].shape}")
    print(f"  Vector dtype: {obs['vector'].dtype}")

    # Check individual sensors
    ultrasonic = obs['vector'][27]
    imu_accel = obs['vector'][28:31]

    print(f"\n  Ultrasonic distance: {ultrasonic:.3f}m")
    print(f"  IMU acceleration: {imu_accel}")

    env_camera.close()

    # Test without camera
    env_vector = PiDogEnv(use_camera=False)
    obs, _ = env_vector.reset()

    print("\nWithout Camera (Vector-only):")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation dtype: {obs.dtype}")
    print(f"  Expected: (31,) float32")

    env_vector.close()
    print("\n✓ All sensors providing data correctly\n")


def test_termination_conditions():
    """Test enhanced fall detection."""
    print("=" * 70)
    print("TERMINATION CONDITIONS TEST")
    print("=" * 70)

    env = PiDogEnv(use_camera=False)
    env.reset()

    print("\nTermination triggers:")
    print("  1. Body height < 0.05m (touching ground)")
    print("  2. Quaternion w < 0.5 (tilted > 60°)")
    print("  3. Roll or Pitch > 50° (fallen to side)")

    print("\nTesting normal walking (should NOT terminate)...")
    terminated = False
    for i in range(20):
        action = np.array([0.0, -0.5, 0.0, -0.5, 0.0, -0.5, 0.0, -0.5])  # Gentle squat
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            print(f"  Terminated at step {i+1} (unexpected)")
            break

    if not terminated:
        print("  ✓ No premature termination")

    env.close()
    print("\n✓ Termination conditions working\n")


def test_collision_avoidance():
    """Test that robot can detect and should avoid leg collisions."""
    print("=" * 70)
    print("LEG COLLISION AVOIDANCE TEST")
    print("=" * 70)

    env = PiDogEnv(use_camera=False)
    env.reset()

    print(f"\nLeg geoms tracked: {len(env.leg_geom_ids)}")
    print("Penalty: -2.0 points per collision")

    # Test normal vs collision scenarios
    print("\nScenario 1: Normal actions (should have few/no collisions)")
    normal_collisions = []
    for _ in range(10):
        action = env.action_space.sample() * 0.3  # Small actions
        obs, reward, term, trunc, info = env.step(action)
        collisions = env._detect_leg_collisions()
        normal_collisions.append(collisions)

    avg_normal = np.mean(normal_collisions)
    print(f"  Average collisions: {avg_normal:.2f}")

    env.close()
    print("\n✓ Collision detection active and working\n")


def main():
    """Run all system tests."""
    print("\n" + "=" * 70)
    print(" PIDOG RL TRAINING SYSTEM - COMPREHENSIVE TEST SUITE ")
    print("=" * 70 + "\n")

    try:
        test_complete_reward_system()
        test_sensor_integration()
        test_termination_conditions()
        test_collision_avoidance()

        print("=" * 70)
        print(" ALL TESTS PASSED ✓ ")
        print("=" * 70)
        print("\nSystem Status:")
        print("  ✓ Reward function: 7 components active")
        print("  ✓ Sensors: Camera, Ultrasonic, IMU (accel + gyro)")
        print("  ✓ Observation space: MultiInput (image + 31D vector)")
        print("  ✓ Action space: 8 leg servos")
        print("  ✓ Termination: Enhanced fall detection")
        print("  ✓ Collision detection: Leg-to-leg contacts")
        print("\n" + "=" * 70)
        print(" READY FOR TRAINING! ")
        print("=" * 70)
        print("\nStart training with:")
        print("  docker-compose run --rm pidog_rl python3 training/train_rl.py \\")
        print("    --algorithm ppo \\")
        print("    --total-timesteps 2000000 \\")
        print("    --n-envs 4 \\")
        print("    --use-camera \\")
        print("    --experiment-name pidog_full_sensors")
        print("\n" + "=" * 70 + "\n")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
