#!/usr/bin/env python3
"""
Test a simple trotting gait using direct angle control.
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pidog_env import PiDogEnv


def create_simple_trot_cycle():
    """
    Create a simple trot cycle using direct joint angles (in degrees).

    Trot gait: diagonal pairs move together
    - Pair 1: RH + LF (right hind + left front)
    - Pair 2: RF + LH (right front + left hind)

    Joint order: [RH_hip, RH_knee, RF_hip, RF_knee, LH_hip, LH_knee, LF_hip, LF_knee]
    """
    # Neutral standing position
    neutral_hip = -30    # degrees
    neutral_knee = -45   # degrees

    # Movement amplitude
    hip_swing = 25       # degrees of forward/backward swing
    knee_lift = 30       # degrees of knee bend when lifting

    # Create 6-step trot cycle
    steps = []

    # Step 0: Pair 1 (RH+LF) start lifting, Pair 2 (RF+LH) pushing
    steps.append([
        neutral_hip - hip_swing,  # RH hip - swing back
        neutral_knee + knee_lift,  # RH knee - lift
        neutral_hip + hip_swing,  # RF hip - swing forward
        neutral_knee,             # RF knee - on ground
        neutral_hip + hip_swing,  # LH hip - swing forward
        neutral_knee,             # LH knee - on ground
        neutral_hip - hip_swing,  # LF hip - swing back
        neutral_knee + knee_lift,  # LF knee - lift
    ])

    # Step 1: Pair 1 (RH+LF) lifted, moving forward
    steps.append([
        neutral_hip,              # RH hip - neutral
        neutral_knee + knee_lift,  # RH knee - still lifted
        neutral_hip + hip_swing,  # RF hip - swing forward
        neutral_knee,             # RF knee - on ground
        neutral_hip + hip_swing,  # LH hip - swing forward
        neutral_knee,             # LH knee - on ground
        neutral_hip,              # LF hip - neutral
        neutral_knee + knee_lift,  # LF knee - still lifted
    ])

    # Step 2: Pair 1 (RH+LF) landing forward, Pair 2 (RF+LH) preparing to lift
    steps.append([
        neutral_hip + hip_swing,  # RH hip - swing forward
        neutral_knee,             # RH knee - land
        neutral_hip - hip_swing,  # RF hip - swing back
        neutral_knee + knee_lift,  # RF knee - start lift
        neutral_hip - hip_swing,  # LH hip - swing back
        neutral_knee + knee_lift,  # LH knee - start lift
        neutral_hip + hip_swing,  # LF hip - swing forward
        neutral_knee,             # LF knee - land
    ])

    # Step 3: Pair 1 (RH+LF) pushing, Pair 2 (RF+LH) lifted
    steps.append([
        neutral_hip + hip_swing,  # RH hip - on ground
        neutral_knee,             # RH knee - on ground
        neutral_hip,              # RF hip - neutral
        neutral_knee + knee_lift,  # RF knee - lifted
        neutral_hip,              # LH hip - neutral
        neutral_knee + knee_lift,  # LH knee - lifted
        neutral_hip + hip_swing,  # LF hip - on ground
        neutral_knee,             # LF knee - on ground
    ])

    # Step 4: Pair 2 (RF+LH) moving forward
    steps.append([
        neutral_hip + hip_swing,  # RH hip - on ground
        neutral_knee,             # RH knee - on ground
        neutral_hip + hip_swing,  # RF hip - swing forward
        neutral_knee,             # RF knee - landing
        neutral_hip + hip_swing,  # LH hip - swing forward
        neutral_knee,             # LH knee - landing
        neutral_hip + hip_swing,  # LF hip - on ground
        neutral_knee,             # LF knee - on ground
    ])

    # Step 5: Return to start position
    steps.append([
        neutral_hip - hip_swing,  # RH hip - swing back (ready for cycle)
        neutral_knee,             # RH knee - on ground
        neutral_hip + hip_swing,  # RF hip - on ground
        neutral_knee,             # RF knee - on ground
        neutral_hip + hip_swing,  # LH hip - on ground
        neutral_knee,             # LH knee - on ground
        neutral_hip - hip_swing,  # LF hip - swing back (ready for cycle)
        neutral_knee,             # LF knee - on ground
    ])

    return np.array(steps)


def degrees_to_normalized(angles_deg):
    """Convert angles in degrees to normalized action space [-1, 1]."""
    # MuJoCo range: -π/2 to π radians (-90° to 180°)
    angles_rad = np.array(angles_deg) * np.pi / 180
    low = -np.pi / 2
    high = np.pi
    normalized = 2 * (angles_rad - low) / (high - low) - 1
    return np.clip(normalized, -1.0, 1.0)


def main():
    print("=" * 80)
    print("TESTING SIMPLE TROT GAIT")
    print("=" * 80)

    # Create trot cycle
    print("\n1. Creating trot cycle from angles...")
    angles_deg = create_simple_trot_cycle()
    print(f"   Generated {len(angles_deg)} steps")

    # Show the angles
    print("\n2. Trot cycle angles (degrees):")
    joint_names = ["RH_hip", "RH_knee", "RF_hip", "RF_knee",
                   "LH_hip", "LH_knee", "LF_hip", "LF_knee"]
    print("   Step | " + " | ".join([f"{name:7}" for name in joint_names]))
    print("   " + "-" * 75)
    for i, angles in enumerate(angles_deg):
        angle_strs = [f"{a:6.1f}°" for a in angles]
        print(f"   {i}    | " + " | ".join(angle_strs))

    # Convert to normalized actions
    actions = np.array([degrees_to_normalized(angles) for angles in angles_deg])

    print("\n3. Testing in environment...")
    env = PiDogEnv(use_camera=False, render_mode="human")

    try:
        obs, _ = env.reset()
        print(f"   Initial height: {obs[26]:.4f}m")

        n_cycles = 100
        print(f"\n4. Running {n_cycles} cycles...")

        for cycle in range(n_cycles):
            cycle_reward = 0

            for step_idx, action in enumerate(actions):
                obs, reward, terminated, truncated, info = env.step(action)
                cycle_reward += reward
                time.sleep(0.02)  # 20ms delay for visualization

                if terminated or truncated:
                    print(f"   Cycle {cycle + 1}: Terminated at step {step_idx}")
                    obs, _ = env.reset()
                    break

            # Print progress every 10 cycles
            if (cycle + 1) % 10 == 0:
                print(f"   Cycle {cycle + 1}/{n_cycles}: "
                      f"reward={cycle_reward:.2f}, "
                      f"height={info['body_height']:.3f}m, "
                      f"vel={info['forward_velocity']:.3f}m/s")

        print("\n   Keeping window open. Press Ctrl+C to exit...")
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n   Closing...")
    finally:
        env.close()

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
