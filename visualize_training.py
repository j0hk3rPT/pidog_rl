"""Visualize current training progress with 3D rendering."""

import sys
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import time

sys.path.insert(0, str(Path(__file__).parent))

from pidog_env import PiDogEnv
from stable_baselines3 import PPO

def find_latest_checkpoint(experiment_dir):
    """Find the most recent model checkpoint."""
    checkpoint_dir = Path(experiment_dir)
    if not checkpoint_dir.exists():
        return None

    # Look for saved models
    checkpoints = list(checkpoint_dir.glob("*.zip"))
    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return checkpoints[0]

def visualize_policy(model_path, num_episodes=5, render_fps=30):
    """Visualize the trained policy in 3D with real-time rendering."""
    print(f"\nLoading model from: {model_path}")

    # Load the trained model
    model = PPO.load(model_path)

    # Create environment with rendering
    env = PiDogEnv(use_camera=True, camera_width=84, camera_height=84)

    print("\n" + "="*70)
    print(" VISUALIZING TRAINED POLICY ")
    print("="*70)
    print("\nControls:")
    print("  - Left mouse: Rotate view")
    print("  - Right mouse: Zoom")
    print("  - Middle mouse: Pan")
    print("  - Space: Pause/Resume")
    print("  - Esc: Close viewer")
    print(f"\nRendering at {render_fps} FPS (real-time)")
    print("\n" + "="*70 + "\n")

    # Calculate frame time for real-time rendering
    frame_time = 1.0 / render_fps

    # Launch viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.sitegroup[0] = True  # Show sensor sites

        for episode in range(num_episodes):
            if not viewer.is_running():
                break

            print(f"\nEpisode {episode + 1}/{num_episodes}")
            obs, _ = env.reset()
            episode_reward = 0
            step_count = 0

            while viewer.is_running():
                step_start = time.time()

                # Get action from policy
                if isinstance(obs, dict):
                    action, _states = model.predict(obs, deterministic=True)
                else:
                    action, _states = model.predict(obs, deterministic=True)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1

                # Sync viewer
                viewer.sync()

                # Print info every 50 steps
                if step_count % 50 == 0:
                    forward_vel = env.data.qvel[0]
                    height = env.data.qpos[2]
                    print(f"  Step {step_count}: vel={forward_vel:.2f}m/s, height={height:.3f}m, reward={reward:.2f}")

                if terminated or truncated:
                    print(f"  Episode ended: {step_count} steps, total reward: {episode_reward:.2f}")
                    time.sleep(0.5)  # Brief pause before resetting
                    break

                # Control frame rate for real-time visualization
                elapsed = time.time() - step_start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

    env.close()

def visualize_random_policy(num_steps=500, render_fps=30):
    """Visualize with random actions (if no checkpoint available) in real-time."""
    print("\nNo checkpoint found. Showing random policy...")

    env = PiDogEnv(use_camera=False)

    print("\n" + "="*70)
    print(" VISUALIZING RANDOM POLICY ")
    print("="*70)
    print("\nControls:")
    print("  - Left mouse: Rotate view")
    print("  - Right mouse: Zoom")
    print("  - Middle mouse: Pan")
    print("  - Space: Pause/Resume")
    print("  - Esc: Close viewer")
    print(f"\nRendering at {render_fps} FPS (real-time)")
    print("\n" + "="*70 + "\n")

    # Calculate frame time for real-time rendering
    frame_time = 1.0 / render_fps

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.sitegroup[0] = True

        obs, _ = env.reset()
        step_count = 0

        while viewer.is_running() and step_count < num_steps:
            step_start = time.time()

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            viewer.sync()

            if step_count % 50 == 0:
                forward_vel = env.data.qvel[0]
                height = env.data.qpos[2]
                print(f"  Step {step_count}: vel={forward_vel:.2f}m/s, height={height:.3f}m")

            if terminated or truncated:
                print(f"  Episode ended at step {step_count}, resetting...")
                time.sleep(0.5)  # Brief pause before resetting
                obs, _ = env.reset()

            # Control frame rate for real-time visualization
            elapsed = time.time() - step_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    env.close()

def main():
    experiment_dir = Path("outputs/pidog_production")

    # Try to find latest checkpoint
    checkpoint = find_latest_checkpoint(experiment_dir)

    if checkpoint:
        print(f"\n{'='*70}")
        print(f" Found checkpoint: {checkpoint.name}")
        print(f" Modified: {checkpoint.stat().st_mtime}")
        print(f"{'='*70}")
        visualize_policy(checkpoint, num_episodes=3)
    else:
        print(f"\nNo checkpoint found in {experiment_dir}")
        print("Showing environment with random policy instead...")
        visualize_random_policy()

if __name__ == "__main__":
    main()
