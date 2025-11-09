"""Quick test training run with visualization and statistics."""

import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent))

from pidog_env import PiDogEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


def test_environment_manual():
    """Test environment with manual control to visualize sensors."""
    print("\n" + "=" * 70)
    print(" PHASE 1: ENVIRONMENT TEST - RANDOM ACTIONS ")
    print("=" * 70 + "\n")

    # Test with camera
    env = PiDogEnv(use_camera=True, camera_width=84, camera_height=84)
    obs, _ = env.reset()

    print("Environment Configuration:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Camera enabled: True")
    print(f"  Image shape: {obs['image'].shape}")
    print(f"  Vector shape: {obs['vector'].shape}")

    print("\nReward Components Active:")
    print("  1. Forward Velocity (3.0×)")
    print("  2. Obstacle Avoidance (1.0×) - Ultrasonic")
    print("  3. Upright Stability (1.5×)")
    print("  4. Stationary Penalty (1.0×)")
    print("  5. Energy Efficiency")
    print("  6. Lateral Stability (0.5×)")
    print("  7. Leg Collision Penalty (2.0×)")

    print("\n" + "-" * 70)
    print("Running 50 steps with random actions...")
    print("-" * 70)

    total_reward = 0
    collision_count = 0

    for step in range(50):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Get sensor data
        forward_vel = env.data.qvel[0]
        collisions = env._detect_leg_collisions()
        ultrasonic = obs['vector'][27]
        height = obs['vector'][26]

        total_reward += reward
        collision_count += collisions

        # Print every 10 steps
        if (step + 1) % 10 == 0:
            print(f"Step {step+1:3d}: "
                  f"reward={reward:6.2f}, "
                  f"vel={forward_vel:5.2f}m/s, "
                  f"height={height:.3f}m, "
                  f"collisions={collisions}, "
                  f"sonar={ultrasonic:5.2f}m")

        if terminated:
            print(f"  → Episode ended at step {step+1} (robot fell)")
            break

    print(f"\nRandom Action Statistics:")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Avg reward per step: {total_reward/(step+1):.2f}")
    print(f"  Total leg collisions: {collision_count}")
    print(f"  Steps survived: {step+1}/50")

    env.close()


def test_short_training():
    """Run a short training session to see learning in action."""
    print("\n" + "=" * 70)
    print(" PHASE 2: SHORT TRAINING RUN (10,000 steps) ")
    print("=" * 70 + "\n")

    # Create environment
    def make_env():
        env = PiDogEnv(use_camera=True, camera_width=84, camera_height=84)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    print("Training Configuration:")
    print("  Algorithm: PPO")
    print("  Total steps: 10,000")
    print("  Policy: MultiInputPolicy (CNN + MLP)")
    print("  Learning rate: 3e-4")
    print("  Batch size: 64")
    print("  N-steps: 512")

    # Import feature extractor
    from pidog_env.feature_extractors import PiDogCombinedExtractor

    # Create model
    print("\nInitializing PPO model...")
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        verbose=1,
        policy_kwargs={
            "features_extractor_class": PiDogCombinedExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
        }
    )

    print("\n" + "-" * 70)
    print("Starting training...")
    print("-" * 70)

    # Train
    model.learn(total_timesteps=10000, progress_bar=True)

    print("\n✓ Short training completed!")

    # Save the model
    output_dir = Path("outputs/test_run")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "test_model.zip"
    model.save(model_path)
    print(f"  Model saved to: {model_path}")

    return model, env


def evaluate_trained_model(model, env):
    """Evaluate the trained model."""
    print("\n" + "=" * 70)
    print(" PHASE 3: EVALUATION - TESTING LEARNED POLICY ")
    print("=" * 70 + "\n")

    print("Running 5 episodes with trained policy...")
    print("-" * 70)

    episode_rewards = []
    episode_lengths = []

    for episode in range(5):
        obs = env.reset()
        episode_reward = 0
        step_count = 0

        while step_count < 200:  # Max 200 steps per episode
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            episode_reward += reward[0]
            step_count += 1

            if done[0]:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)

        print(f"Episode {episode+1}: reward={episode_reward:7.2f}, steps={step_count:3d}")

    print("\nEvaluation Results:")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Best episode: {np.max(episode_rewards):.2f}")


def test_vector_only_training():
    """Quick test with vector-only (faster training)."""
    print("\n" + "=" * 70)
    print(" ALTERNATIVE: VECTOR-ONLY TRAINING (FASTER) ")
    print("=" * 70 + "\n")

    # Create vector-only environment
    def make_env():
        env = PiDogEnv(use_camera=False)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    print("Training Configuration:")
    print("  Algorithm: PPO")
    print("  Total steps: 20,000")
    print("  Policy: MlpPolicy (vector-only, faster)")
    print("  Learning rate: 3e-4")

    # Create model
    print("\nInitializing PPO model (MLP)...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        verbose=1,
    )

    print("\n" + "-" * 70)
    print("Starting vector-only training...")
    print("-" * 70)

    # Train
    model.learn(total_timesteps=20000, progress_bar=True)

    print("\n✓ Vector-only training completed!")

    # Save
    output_dir = Path("outputs/test_run")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "test_model_vector.zip"
    model.save(model_path)
    print(f"  Model saved to: {model_path}")

    # Quick evaluation
    print("\nQuick evaluation (3 episodes)...")
    episode_rewards = []

    for episode in range(3):
        obs = env.reset()
        episode_reward = 0
        step_count = 0

        while step_count < 200:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            step_count += 1
            if done[0]:
                break

        episode_rewards.append(episode_reward)
        print(f"  Episode {episode+1}: reward={episode_reward:7.2f}, steps={step_count}")

    print(f"\n  Mean reward: {np.mean(episode_rewards):.2f}")


def main():
    """Run all test phases."""
    print("\n" + "=" * 70)
    print(" PIDOG RL TRAINING - TEST RUN ")
    print("=" * 70)

    print("\nThis test will:")
    print("  1. Test environment with random actions")
    print("  2. Run short training with camera (10k steps)")
    print("  3. Evaluate learned policy")
    print("  4. Alternative: Vector-only training (20k steps, faster)")

    print("\n" + "=" * 70)

    try:
        # Phase 1: Test environment
        test_environment_manual()

        # Ask user which training mode
        print("\n" + "=" * 70)
        print("Choose training mode:")
        print("  1. Full sensors (camera + vector) - Slower, more realistic")
        print("  2. Vector-only - Faster, good for testing")
        print("=" * 70)

        # For automated testing, let's do vector-only (faster)
        print("\nRunning vector-only test (faster)...")

        test_vector_only_training()

        print("\n" + "=" * 70)
        print(" TEST RUN COMPLETED SUCCESSFULLY! ✓ ")
        print("=" * 70)

        print("\nNext steps:")
        print("  1. Review the test results above")
        print("  2. Check saved models in: outputs/test_run/")
        print("  3. Start full training when ready:")
        print("\n     docker-compose run --rm pidog_rl python3 training/train_rl.py \\")
        print("       --algorithm ppo \\")
        print("       --total-timesteps 2000000 \\")
        print("       --n-envs 4 \\")
        print("       --use-camera \\")
        print("       --experiment-name pidog_production")

        print("\n" + "=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
