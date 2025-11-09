"""Callback for visualizing training progress in real-time."""

import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from pidog_env import PiDogEnv


class VisualizeCallback(BaseCallback):
    """
    Callback to visualize the policy during training at specified intervals.

    This allows you to see the robot's behavior in real-time as it learns,
    showing checkpoints every N timesteps.
    """

    def __init__(
        self,
        visualize_freq: int = 50000,
        n_eval_episodes: int = 1,
        render_fps: int = 30,
        max_steps_per_episode: int = 500,
        use_camera: bool = False,
        verbose: int = 0,
    ):
        """
        Initialize the visualization callback.

        Args:
            visualize_freq: Visualize every N timesteps
            n_eval_episodes: Number of episodes to visualize
            render_fps: Frames per second for real-time rendering
            max_steps_per_episode: Maximum steps per visualization episode
            use_camera: Whether to use camera observations
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.visualize_freq = visualize_freq
        self.n_eval_episodes = n_eval_episodes
        self.render_fps = render_fps
        self.max_steps_per_episode = max_steps_per_episode
        self.use_camera = use_camera
        self.frame_time = 1.0 / render_fps
        self.last_visualization = 0

    def _on_step(self) -> bool:
        """
        Called at each step. Check if we should visualize.

        Returns:
            True to continue training, False to stop
        """
        # Check if it's time to visualize
        if self.num_timesteps - self.last_visualization >= self.visualize_freq:
            self.last_visualization = self.num_timesteps
            self._visualize_policy()

        return True

    def _visualize_policy(self):
        """Visualize the current policy in real-time."""
        print("\n" + "="*70)
        print(f" VISUALIZATION CHECKPOINT - {self.num_timesteps:,} timesteps")
        print("="*70)
        print("\nControls:")
        print("  - Left mouse: Rotate view")
        print("  - Right mouse: Zoom")
        print("  - Middle mouse: Pan")
        print("  - Space: Pause/Resume")
        print("  - Esc: Skip to next checkpoint")
        print(f"\nRendering at {self.render_fps} FPS (real-time)")
        print(f"Episodes: {self.n_eval_episodes}")
        print("\n" + "="*70 + "\n")

        # Create a separate environment for visualization
        # This avoids interfering with the training environment
        try:
            env = PiDogEnv(use_camera=self.use_camera)

            # Launch viewer with the environment's model and data
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                viewer.opt.sitegroup[0] = True  # Show sensor sites

                for episode in range(self.n_eval_episodes):
                    if not viewer.is_running():
                        print("\nVisualization interrupted by user")
                        break

                    print(f"\nEpisode {episode + 1}/{self.n_eval_episodes}")
                    obs, _ = env.reset()
                    episode_reward = 0
                    step_count = 0

                    while viewer.is_running() and step_count < self.max_steps_per_episode:
                        step_start = time.time()

                        # Get action from the current policy
                        action, _ = self.model.predict(obs, deterministic=True)

                        # Step environment
                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_reward += reward
                        step_count += 1

                        # Sync viewer with environment state
                        viewer.sync()

                        # Print progress
                        if step_count % 50 == 0:
                            forward_vel = env.data.qvel[0]
                            height = env.data.qpos[2]
                            print(f"  Step {step_count}: vel={forward_vel:.2f}m/s, "
                                  f"height={height:.3f}m, reward={reward:.2f}")

                        # Check if episode ended
                        if terminated or truncated:
                            print(f"  Episode ended: {step_count} steps, "
                                  f"total reward: {episode_reward:.2f}")
                            time.sleep(1.0)  # Pause before next episode
                            break

                        # Control frame rate for real-time visualization
                        elapsed = time.time() - step_start
                        if elapsed < self.frame_time:
                            time.sleep(self.frame_time - elapsed)

                    if not viewer.is_running():
                        break

            # Clean up visualization environment
            env.close()

        except Exception as e:
            print(f"\nVisualization error: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing training...")

        print("\n" + "="*70)
        print(" RESUMING TRAINING")
        print("="*70 + "\n")


class PeriodicVisualizeCallback(BaseCallback):
    """
    Callback to save and visualize model at periodic checkpoints.

    This is useful for longer training runs where you want to see
    progress at specific intervals (e.g., every 100K steps).
    """

    def __init__(
        self,
        save_freq: int = 100000,
        save_path: str = "./checkpoints",
        name_prefix: str = "rl_model",
        visualize: bool = True,
        n_eval_episodes: int = 2,
        render_fps: int = 30,
        use_camera: bool = False,
        verbose: int = 1,
    ):
        """
        Initialize periodic visualization callback.

        Args:
            save_freq: Save and visualize every N timesteps
            save_path: Directory to save checkpoints
            name_prefix: Prefix for checkpoint files
            visualize: Whether to visualize after saving
            n_eval_episodes: Number of episodes to visualize
            render_fps: Frames per second for visualization
            use_camera: Whether to use camera observations
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.name_prefix = name_prefix
        self.visualize = visualize
        self.n_eval_episodes = n_eval_episodes
        self.render_fps = render_fps
        self.use_camera = use_camera
        self.frame_time = 1.0 / render_fps

    def _on_step(self) -> bool:
        """Check if we should save/visualize at this step."""
        if self.n_calls % self.save_freq == 0:
            # Save checkpoint
            checkpoint_path = self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps"
            self.model.save(checkpoint_path)

            if self.verbose > 0:
                print(f"\nCheckpoint saved: {checkpoint_path}")

            # Visualize if enabled
            if self.visualize:
                self._visualize_checkpoint()

        return True

    def _visualize_checkpoint(self):
        """Visualize current policy checkpoint."""
        visualizer = VisualizeCallback(
            visualize_freq=float('inf'),  # Only visualize when called
            n_eval_episodes=self.n_eval_episodes,
            render_fps=self.render_fps,
            use_camera=self.use_camera,
            verbose=self.verbose,
        )
        # Set up the visualizer with current model
        visualizer.model = self.model
        visualizer.num_timesteps = self.num_timesteps

        # Run visualization
        visualizer._visualize_policy()
