#!/usr/bin/env python3
"""
Train PiDog using imitation learning from Sunfounder demonstrations.

This script:
1. Loads Sunfounder extracted actions
2. Executes them in the environment to collect full transitions
3. Trains a policy using behavioral cloning or GAIL
"""

import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pidog_env import PiDogEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Try to import imitation
try:
    from imitation.algorithms import bc
    from imitation.data.types import Transitions
    IMITATION_AVAILABLE = True
except ImportError:
    IMITATION_AVAILABLE = False
    print("WARNING: 'imitation' package not available, only RL training will work")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PiDog from Sunfounder demos")
    parser.add_argument(
        "--demos-path",
        type=str,
        default="datasets/sunfounder_demos.pkl",
        help="Path to Sunfounder demonstrations",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="bc",
        choices=["bc", "rl", "both"],
        help="Training method: bc (behavioral cloning), rl (reinforcement learning), both (BC then RL)",
    )
    parser.add_argument(
        "--gait",
        type=str,
        default="trot_forward",
        help="Specific gait to train on (or 'all' for all gaits)",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=100,
        help="Number of BC training epochs",
    )
    parser.add_argument(
        "--rl-timesteps",
        type=int,
        default=500_000,
        help="Number of RL training timesteps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for models and logs",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)",
    )
    return parser.parse_args()


def load_sunfounder_demos(demos_path, gait_name=None):
    """
    Load Sunfounder demonstrations.

    Args:
        demos_path: Path to demonstrations file
        gait_name: Specific gait to load, or None for all gaits

    Returns:
        numpy array of actions
    """
    print(f"Loading demonstrations from {demos_path}...")
    with open(demos_path, 'rb') as f:
        dataset = pickle.load(f)

    actions = dataset['actions']
    labels = dataset['labels']

    if gait_name and gait_name != 'all':
        # Filter for specific gait
        gait_indices = [i for i, label in enumerate(labels) if label == gait_name]
        if not gait_indices:
            raise ValueError(f"Gait '{gait_name}' not found in dataset")
        actions = actions[gait_indices]
        print(f"Loaded {len(actions)} steps for gait '{gait_name}'")
    else:
        print(f"Loaded {len(actions)} total steps from all gaits")

    return actions


def collect_transitions_from_actions(env, actions, n_repeats=3):
    """
    Execute actions in environment to collect full transitions.

    Args:
        env: PiDog environment
        actions: numpy array of actions to execute
        n_repeats: Number of times to repeat the action sequence

    Returns:
        Transitions object with obs, acts, next_obs, dones
    """
    print(f"Collecting transitions by executing {len(actions)} actions {n_repeats} times...")

    obs_list = []
    acts_list = []
    next_obs_list = []
    dones_list = []

    for repeat in range(n_repeats):
        obs, _ = env.reset()
        episode_steps = 0

        for action in actions:
            # Store current observation (ensure float32)
            obs_list.append(obs.astype(np.float32))
            acts_list.append(action.astype(np.float32))

            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_obs_list.append(next_obs.astype(np.float32))
            dones_list.append(done)

            obs = next_obs
            episode_steps += 1

            if done:
                print(f"  Repeat {repeat + 1}: Episode ended after {episode_steps} steps")
                obs, _ = env.reset()
                episode_steps = 0

        if episode_steps > 0:
            print(f"  Repeat {repeat + 1}: Completed {episode_steps} steps")

    # Create Transitions object with proper dtypes
    transitions = Transitions(
        obs=np.array(obs_list, dtype=np.float32),
        acts=np.array(acts_list, dtype=np.float32),
        infos=np.array([{}] * len(obs_list)),
        next_obs=np.array(next_obs_list, dtype=np.float32),
        dones=np.array(dones_list, dtype=bool),
    )

    print(f"Collected {len(transitions)} transitions")
    return transitions


def train_behavioral_cloning(transitions, env, args):
    """Train using Behavioral Cloning."""
    if not IMITATION_AVAILABLE:
        print("ERROR: 'imitation' package required for BC training")
        print("Install with: pip install imitation")
        return None

    print("\n" + "=" * 70)
    print("Training with Behavioral Cloning")
    print("=" * 70)

    # Determine device
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")

    # Create BC trainer
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=np.random.default_rng(42),
        device=device,
        batch_size=64,
    )

    # Train
    print(f"\nTraining for {args.bc_epochs} epochs...")
    bc_trainer.train(n_epochs=args.bc_epochs)

    print("BC training complete!")
    return bc_trainer.policy


def train_reinforcement_learning(env, args, initial_policy=None):
    """Train using Reinforcement Learning (PPO)."""
    print("\n" + "=" * 70)
    print("Training with Reinforcement Learning (PPO)")
    print("=" * 70)

    # Determine device
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")

    # Create PPO learner
    model = PPO(
        policy="MlpPolicy",
        env=env,
        batch_size=64,
        learning_rate=3e-4,
        n_steps=2048,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=args.log_dir,
        device=device,
    )

    # If initial policy provided (from BC), copy weights
    if initial_policy is not None:
        print("Initializing from BC policy...")
        # Note: This is a simple approach, might need adjustment based on policy architecture
        try:
            model.policy.load_state_dict(initial_policy.state_dict())
            print("Successfully initialized from BC policy")
        except Exception as e:
            print(f"Warning: Could not load BC policy weights: {e}")
            print("Starting RL training from scratch")

    # Train
    print(f"\nTraining for {args.rl_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=args.rl_timesteps,
        progress_bar=True,
    )

    print("RL training complete!")
    return model


def evaluate_policy(policy, env, n_episodes=10):
    """Evaluate trained policy."""
    print(f"\nEvaluating policy for {n_episodes} episodes...")

    episode_rewards = []
    episode_lengths = []
    forward_velocities = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        max_forward_vel = 0

        while not done and episode_length < 1000:
            # Get action from policy
            if hasattr(policy, 'predict'):
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action = policy(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            max_forward_vel = max(max_forward_vel, info.get('forward_velocity', 0))

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        forward_velocities.append(max_forward_vel)

        print(f"  Episode {episode + 1}: reward={episode_reward:.2f}, "
              f"length={episode_length}, max_vel={max_forward_vel:.3f} m/s")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    mean_velocity = np.mean(forward_velocities)

    print(f"\nEvaluation Results:")
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Mean episode length: {mean_length:.0f}")
    print(f"  Mean forward velocity: {mean_velocity:.3f} m/s")

    return mean_reward, std_reward


def main():
    """Main training function."""
    args = parse_args()

    # Set up experiment directory
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gait_suffix = args.gait if args.gait != 'all' else 'all_gaits'
        args.experiment_name = f"{args.method}_{gait_suffix}_{timestamp}"

    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    args.log_dir = str(experiment_dir / "logs")
    args.checkpoint_dir = str(experiment_dir / "checkpoints")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Training PiDog with Sunfounder Demonstrations")
    print("=" * 70)
    print(f"Experiment: {args.experiment_name}")
    print(f"Method: {args.method}")
    print(f"Gait: {args.gait}")
    print(f"Output directory: {experiment_dir}")
    print("=" * 70)

    # Load demonstrations
    demos_path = Path(__file__).parent.parent / args.demos_path
    actions = load_sunfounder_demos(str(demos_path), args.gait)

    # Create environment (without camera for faster training)
    print("\nCreating environment...")
    env = PiDogEnv(use_camera=False)

    # Collect transitions if using BC or both methods
    transitions = None
    if args.method in ["bc", "both"]:
        transitions = collect_transitions_from_actions(env, actions, n_repeats=5)

        # Save transitions
        transitions_path = experiment_dir / "transitions.pkl"
        with open(transitions_path, 'wb') as f:
            pickle.dump(transitions, f)
        print(f"Saved transitions to {transitions_path}")

    # Train based on method
    policy = None

    if args.method in ["bc", "both"]:
        policy = train_behavioral_cloning(transitions, env, args)

        # Save BC policy
        bc_policy_path = experiment_dir / "bc_policy"
        if hasattr(policy, 'save'):
            policy.save(bc_policy_path)
            print(f"BC policy saved to {bc_policy_path}")

        # Evaluate BC policy
        evaluate_policy(policy, env, n_episodes=5)

    if args.method in ["rl", "both"]:
        # Create vectorized environment for RL
        vec_env = DummyVecEnv([lambda: PiDogEnv(use_camera=False)])

        # Train RL (optionally starting from BC policy)
        initial_policy = policy if args.method == "both" else None
        model = train_reinforcement_learning(vec_env, args, initial_policy)

        # Save RL policy
        rl_policy_path = experiment_dir / "rl_policy"
        model.save(rl_policy_path)
        print(f"RL policy saved to {rl_policy_path}")

        # Evaluate RL policy
        evaluate_policy(model, env, n_episodes=10)

        policy = model

    # Clean up
    env.close()

    print(f"\n{'=' * 70}")
    print(f"Training completed!")
    print(f"Results saved in: {experiment_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
