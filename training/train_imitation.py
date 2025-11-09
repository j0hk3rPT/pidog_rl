"""Train PiDog using Imitation Learning (Behavioral Cloning and GAIL)."""

import argparse
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms import bc, gail
from imitation.data import rollout
from imitation.data.types import Transitions
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
import torch

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pidog_env import PiDogEnv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PiDog with Imitation Learning")
    parser.add_argument(
        "--method",
        type=str,
        default="bc",
        choices=["bc", "gail", "dagger"],
        help="Imitation learning method",
    )
    parser.add_argument(
        "--expert-data",
        type=str,
        required=True,
        help="Path to expert demonstrations (pickle file)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps (for GAIL)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=100,
        help="Number of training epochs (for BC)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu/cuda/auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for logs and checkpoints",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation",
    )
    return parser.parse_args()


def load_expert_data(expert_data_path: str) -> Transitions:
    """Load expert demonstrations from file."""
    print(f"Loading expert data from {expert_data_path}...")

    with open(expert_data_path, "rb") as f:
        data = pickle.load(f)

    # Convert to Transitions format if needed
    if isinstance(data, Transitions):
        transitions = data
    elif isinstance(data, dict):
        # Assume dict with keys: obs, acts, next_obs, dones
        transitions = Transitions(
            obs=data["obs"],
            acts=data["acts"],
            infos=data.get("infos", np.array([{}] * len(data["obs"]))),
            next_obs=data["next_obs"],
            dones=data["dones"],
        )
    else:
        raise ValueError(f"Unknown expert data format: {type(data)}")

    print(f"Loaded {len(transitions)} expert transitions")
    return transitions


def collect_expert_data(
    env: gym.Env,
    n_episodes: int = 10,
    policy_path: Optional[str] = None
) -> Transitions:
    """
    Collect expert demonstrations.

    If policy_path is provided, use that policy.
    Otherwise, use a simple hardcoded policy.
    """
    print(f"Collecting {n_episodes} expert episodes...")

    if policy_path:
        # Load expert policy
        expert_policy = PPO.load(policy_path)
    else:
        # Use hardcoded policy (simple walking gait)
        expert_policy = None

    # Collect rollouts
    if expert_policy:
        transitions = rollout.rollout(
            expert_policy,
            env,
            rollout.make_sample_until(min_episodes=n_episodes),
        )
    else:
        # Collect using hardcoded policy
        transitions = collect_hardcoded_demos(env, n_episodes)

    print(f"Collected {len(transitions)} transitions")
    return transitions


def collect_hardcoded_demos(env: gym.Env, n_episodes: int) -> Transitions:
    """
    Collect demonstrations using a hardcoded walking gait.

    This creates a simple alternating gait pattern for the quadruped.
    """
    obs_list = []
    acts_list = []
    next_obs_list = []
    dones_list = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0

        while not done:
            # Simple sinusoidal gait
            t = step * 0.1

            # Alternating leg pattern
            # Front legs and back legs move together
            action = np.array([
                0.3 * np.sin(t),          # Front left shoulder
                0.5 * np.cos(t),          # Front left knee
                0.3 * np.sin(t + np.pi),  # Front right shoulder
                0.5 * np.cos(t + np.pi),  # Front right knee
                0.3 * np.sin(t + np.pi),  # Back left shoulder
                0.5 * np.cos(t + np.pi),  # Back left knee
                0.3 * np.sin(t),          # Back right shoulder
                0.5 * np.cos(t),          # Back right knee
            ])

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            obs_list.append(obs)
            acts_list.append(action)
            next_obs_list.append(next_obs)
            dones_list.append(done)

            obs = next_obs
            step += 1

            if step > 1000:  # Max steps per episode
                break

        print(f"Episode {episode + 1}/{n_episodes} completed with {step} steps")

    transitions = Transitions(
        obs=np.array(obs_list),
        acts=np.array(acts_list),
        infos=np.array([{}] * len(obs_list)),
        next_obs=np.array(next_obs_list),
        dones=np.array(dones_list),
    )

    return transitions


def train_behavioral_cloning(args, transitions, env):
    """Train using Behavioral Cloning."""
    print("\n" + "=" * 60)
    print("Training with Behavioral Cloning")
    print("=" * 60)

    # Create BC trainer
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=np.random.default_rng(args.seed),
        device=args.device,
        batch_size=args.batch_size,
    )

    # Train
    print(f"\nTraining for {args.n_epochs} epochs...")
    bc_trainer.train(n_epochs=args.n_epochs)

    return bc_trainer.policy


def train_gail(args, transitions, env):
    """Train using Generative Adversarial Imitation Learning."""
    print("\n" + "=" * 60)
    print("Training with GAIL")
    print("=" * 60)

    # Create reward network
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )

    # Create PPO learner
    learner = PPO(
        env=env,
        policy="MlpPolicy",
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_steps=2048,
        verbose=1,
        device=args.device,
        tensorboard_log=args.log_dir,
    )

    # Create GAIL trainer
    gail_trainer = gail.GAIL(
        demonstrations=transitions,
        demo_batch_size=args.batch_size,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )

    # Train
    print(f"\nTraining for {args.total_timesteps:,} timesteps...")
    gail_trainer.train(args.total_timesteps)

    return learner.policy


def evaluate_policy(policy, env, n_episodes=10):
    """Evaluate trained policy."""
    print(f"\nEvaluating policy for {n_episodes} episodes...")

    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            # Get action from policy
            if hasattr(policy, 'predict'):
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action = policy(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}: reward={episode_reward:.2f}, length={episode_length}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    print(f"\nEvaluation Results:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean episode length: {mean_length:.0f}")

    return mean_reward, std_reward


def main():
    """Main training function."""
    args = parse_args()

    # Set up experiment directory
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.method}_{timestamp}"

    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    args.log_dir = str(experiment_dir / "logs")
    args.checkpoint_dir = str(experiment_dir / "checkpoints")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 60)
    print(f"Training PiDog with {args.method.upper()}")
    print("=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Method: {args.method}")
    print(f"Expert data: {args.expert_data}")
    print(f"Device: {args.device}")
    print(f"Output directory: {experiment_dir}")
    print("=" * 60)

    # Create environment
    print("\nCreating environment...")
    env = DummyVecEnv([lambda: PiDogEnv()])

    # Load or collect expert data
    if Path(args.expert_data).exists():
        transitions = load_expert_data(args.expert_data)
    else:
        print(f"Expert data not found at {args.expert_data}")
        print("Collecting hardcoded demonstrations instead...")
        transitions = collect_hardcoded_demos(env, n_episodes=20)

        # Save for future use
        with open(args.expert_data, "wb") as f:
            pickle.dump(transitions, f)
        print(f"Saved expert data to {args.expert_data}")

    # Train based on method
    if args.method == "bc":
        policy = train_behavioral_cloning(args, transitions, env)
    elif args.method == "gail":
        policy = train_gail(args, transitions, env)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Evaluate policy
    evaluate_policy(policy, env, n_episodes=args.eval_episodes)

    # Save policy
    policy_path = experiment_dir / f"{args.method}_policy"
    if hasattr(policy, 'save'):
        policy.save(policy_path)
    else:
        torch.save(policy.state_dict(), str(policy_path) + ".pth")
    print(f"\nPolicy saved to: {policy_path}")

    # Clean up
    env.close()

    print(f"\nTraining completed!")
    print(f"Results saved in: {experiment_dir}")


if __name__ == "__main__":
    main()
