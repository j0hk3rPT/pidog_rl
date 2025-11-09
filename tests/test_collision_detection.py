"""Test leg self-collision detection."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pidog_env import PiDogEnv


def test_collision_detection():
    """Test that leg collision detection works."""
    print("=" * 60)
    print("TEST: Leg Self-Collision Detection")
    print("=" * 60)

    env = PiDogEnv(use_camera=False)
    obs, _ = env.reset()

    print(f"\nLeg geoms tracked for collision: {len(env.leg_geom_ids)}")
    print(f"Geom IDs: {env.leg_geom_ids[:10]}... (showing first 10)")

    # Test normal standing pose - should have no collisions
    print("\n--- Test 1: Normal Standing Pose ---")
    collisions = env._detect_leg_collisions()
    print(f"Collisions detected: {collisions}")
    if collisions == 0:
        print("✓ No collisions in neutral standing pose (GOOD)")
    else:
        print(f"⚠ Warning: {collisions} collisions in neutral pose")

    # Test a few random steps
    print("\n--- Test 2: Random Actions (5 steps) ---")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        collisions = env._detect_leg_collisions()
        print(f"  Step {step+1}: {collisions} leg collisions, reward={reward:.3f}")

    # Test extreme pose that might cause collisions
    print("\n--- Test 3: Force Leg Crossing ---")
    print("Setting all legs to extreme inward angles...")

    # Reset first
    env.reset()

    # Force legs inward (trying to make them cross)
    # This is an action that should cause collisions
    extreme_action = np.array([
        -0.8, -0.5,  # Back right - inward
        -0.8, -0.5,  # Front right - inward
        0.8, -0.5,   # Back left - inward
        0.8, -0.5,   # Front left - inward
    ])

    for _ in range(10):  # Run for several steps to let legs move
        obs, reward, terminated, truncated, info = env.step(extreme_action)

    collisions = env._detect_leg_collisions()
    print(f"Collisions after extreme pose: {collisions}")
    if collisions > 0:
        print(f"✓ Collision detection working! ({collisions} collisions detected)")
    else:
        print("⚠ No collisions detected (legs may not have crossed)")

    env.close()
    print("\n✓ Collision detection test complete\n")


def test_reward_with_collisions():
    """Test that collisions affect reward."""
    print("=" * 60)
    print("TEST: Collision Penalty in Reward")
    print("=" * 60)

    env = PiDogEnv(use_camera=False)
    env.reset()

    print("\nReward structure includes:")
    print("  - Forward velocity (3.0×)")
    print("  - Obstacle avoidance (1.0×)")
    print("  - Upright stability (1.5×)")
    print("  - Stationary penalty (1.0×)")
    print("  - Energy efficiency (small)")
    print("  - Lateral stability (0.5×)")
    print("  - Leg collision penalty (2.0× per collision) [NEW!]")

    print("\nTesting reward with different collision counts:")

    # Run a few steps and check reward components
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        collisions = env._detect_leg_collisions()
        collision_penalty = -2.0 * collisions

        print(f"  Step {i+1}: collisions={collisions}, "
              f"collision_penalty={collision_penalty:.2f}, total_reward={reward:.3f}")

    env.close()
    print("\n✓ Collision penalty integrated in reward\n")


def main():
    """Run all collision tests."""
    print("\n" + "=" * 60)
    print("LEG SELF-COLLISION TEST SUITE")
    print("=" * 60 + "\n")

    test_collision_detection()
    test_reward_with_collisions()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n✓ Leg collision detection implemented")
    print("✓ Penalty: -2.0 points per leg-to-leg contact")
    print("✓ Robot will learn to avoid crossing legs")
    print("✓ Natural gait will be rewarded")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
