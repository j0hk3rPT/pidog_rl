#!/usr/bin/env python3
"""
Complete workflow: Imitation Pre-training → RL Fine-tuning

This script demonstrates the full pipeline:
1. Generate expert demonstrations (hardcoded gait)
2. Pre-train with Behavioral Cloning (BC)
3. Fine-tune with RL (PPO)

This approach can significantly speed up RL training by starting
from a reasonable policy instead of random initialization.

Usage:
    # Full pipeline
    python train_pretrain_finetune.py --total-bc-epochs 200 --total-rl-timesteps 500000

    # Skip BC if you have a pre-trained model
    python train_pretrain_finetune.py --pretrained-model outputs/bc_model/bc_policy --total-rl-timesteps 500000
"""

import argparse
from pathlib import Path
from datetime import datetime
import pickle
import numpy as np
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

# Try to import imitation for BC
try:
    from imitation.algorithms import bc
    from imitation.data.types import Transitions
    IMITATION_AVAILABLE = True
except ImportError:
    IMITATION_AVAILABLE = False
    print("\nWarning: 'imitation' package not installed")
    print("Install it with: pip install imitation")
    print("BC pre-training will be skipped\n")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pidog_env import PiDogEnv
from training.visualization_callback import VisualizeCallback


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-train with BC, then fine-tune with RL"
    )

    # BC pre-training args
    parser.add_argument(
        "--skip-bc",
        action="store_true",
        help="Skip BC pre-training (use if you have pretrained model)",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="Path to pretrained model to use as initialization",
    )
    parser.add_argument(
        "--sunfounder-demos",
        type=str,
        default=None,
        help="Path to Sunfounder demonstration file (pkl)",
    )
    parser.add_argument(
        "--n-demo-episodes",
        type=int,
        default=50,
        help="Number of demonstration episodes to collect (default: 50, ignored if using Sunfounder demos)",
    )
    parser.add_argument(
        "--total-bc-epochs",
        type=int,
        default=200,
        help="Total BC training epochs (default: 200)",
    )

    # RL fine-tuning args
    parser.add_argument(
        "--total-rl-timesteps",
        type=int,
        default=500000,
        help="Total RL fine-tuning timesteps (default: 500000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments for RL (default: 4)",
    )
    parser.add_argument(
        "--visualize-freq",
        type=int,
        default=50000,
        help="Visualize every N timesteps during RL (default: 50000, 0=disable)",
    )

    # General args
    parser.add_argument(
        "--use-camera",
        action="store_true",
        help="Use camera observations (MultiInputPolicy)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for RL (default: 3e-4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    return parser.parse_args()


def collect_hardcoded_demonstrations(n_episodes: int = 50, use_camera: bool = False):
    """
    Collect demonstrations using a hardcoded walking gait.

    This creates a simple alternating gait pattern that the robot can learn from.
    It's not perfect, but it's much better than random initialization.
    """
    print("\n" + "="*70)
    print(" COLLECTING EXPERT DEMONSTRATIONS")
    print("="*70)
    print(f"Generating {n_episodes} episodes with hardcoded gait...")
    print("This may take a few minutes...\n")

    env = PiDogEnv(use_camera=use_camera)

    obs_list = []
    acts_list = []
    next_obs_list = []
    dones_list = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        episode_reward = 0

        while not done and step < 500:  # Max 500 steps per episode
            # Simple sinusoidal gait with forward bias
            t = step * 0.05  # Slower frequency for stable gait

            # Diagonal gait pattern (trot)
            # FL and RR move together, FR and RL move together
            action = np.array([
                # Front Left (FL)
                0.2 * np.sin(t) - 0.1,        # Hip: slightly backward bias
                0.4 * np.cos(t) - 0.2,        # Knee: lift and extend

                # Front Right (FR)
                0.2 * np.sin(t + np.pi) - 0.1,
                0.4 * np.cos(t + np.pi) - 0.2,

                # Rear Left (RL)
                0.2 * np.sin(t + np.pi) - 0.15,  # More backward for push
                0.4 * np.cos(t + np.pi) - 0.2,

                # Rear Right (RR)
                0.2 * np.sin(t) - 0.15,
                0.4 * np.cos(t) - 0.2,
            ])

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Store transition
            if use_camera:
                # For dict observations, only store vector part for BC
                obs_list.append(obs["vector"] if isinstance(obs, dict) else obs)
                next_obs_list.append(next_obs["vector"] if isinstance(next_obs, dict) else next_obs)
            else:
                obs_list.append(obs)
                next_obs_list.append(next_obs)

            acts_list.append(action)
            dones_list.append(done)

            obs = next_obs
            step += 1

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{n_episodes}: {step} steps, reward={episode_reward:.2f}")

    env.close()

    transitions = Transitions(
        obs=np.array(obs_list),
        acts=np.array(acts_list),
        infos=np.array([{}] * len(obs_list)),
        next_obs=np.array(next_obs_list),
        dones=np.array(dones_list),
    )

    print(f"\nCollected {len(transitions)} total transitions")
    print("="*70 + "\n")

    return transitions


def train_bc_pretraining(transitions, experiment_dir, args):
    """Train BC model for pre-training."""
    if not IMITATION_AVAILABLE:
        print("Skipping BC pre-training (imitation package not available)")
        return None

    print("\n" + "="*70)
    print(" PHASE 1: BEHAVIORAL CLONING PRE-TRAINING")
    print("="*70)
    print(f"Training BC for {args.total_bc_epochs} epochs...")
    print(f"This will create a baseline policy to initialize RL\n")

    # Create environment for BC trainer
    env = PiDogEnv(use_camera=args.use_camera)

    # Create BC trainer
    bc_trainer = bc.BC(
        observation_space=env.observation_space if not args.use_camera
                         else env.observation_space["vector"],  # BC uses vector obs only
        action_space=env.action_space,
        demonstrations=transitions,
        rng=np.random.default_rng(args.seed),
        batch_size=64,
    )

    # Train
    bc_trainer.train(n_epochs=args.total_bc_epochs)

    # Save BC policy
    bc_policy_path = experiment_dir / "bc_pretrained_policy"
    torch_save_path = str(bc_policy_path) + ".pth"

    import torch
    torch.save(bc_trainer.policy.state_dict(), torch_save_path)
    print(f"\nBC policy saved to: {torch_save_path}")

    env.close()

    print("\n" + "="*70)
    print(" BC PRE-TRAINING COMPLETE")
    print("="*70 + "\n")

    return bc_trainer.policy


def create_rl_model_from_bc(bc_policy, env, args):
    """Create PPO model initialized with BC policy weights."""
    print("\n" + "="*70)
    print(" PHASE 2: INITIALIZING RL MODEL FROM BC POLICY")
    print("="*70)
    print("Creating PPO model with BC-initialized weights...\n")

    policy_type = "MultiInputPolicy" if args.use_camera else "MlpPolicy"

    # Create PPO model
    model = PPO(
        policy_type,
        env,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
    )

    # Transfer BC weights to PPO policy
    if bc_policy is not None:
        try:
            # Copy weights from BC policy to PPO policy
            bc_state_dict = bc_policy.state_dict()
            ppo_state_dict = model.policy.state_dict()

            # Transfer compatible weights
            transferred = 0
            for key in ppo_state_dict.keys():
                if key in bc_state_dict:
                    if ppo_state_dict[key].shape == bc_state_dict[key].shape:
                        ppo_state_dict[key] = bc_state_dict[key]
                        transferred += 1

            model.policy.load_state_dict(ppo_state_dict)
            print(f"✓ Transferred {transferred} weight tensors from BC to PPO")
            print("✓ PPO policy initialized with BC weights\n")
        except Exception as e:
            print(f"Warning: Could not transfer BC weights: {e}")
            print("Continuing with random initialization\n")

    print("="*70 + "\n")
    return model


def make_env(rank, seed=0, use_camera=False):
    """Create a single environment instance."""
    def _init():
        env = PiDogEnv(use_camera=use_camera)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    """Main training pipeline."""
    args = parse_args()

    # Set up experiment directory
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"bc_rl_{timestamp}"

    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    demos_path = experiment_dir / "demonstrations.pkl"

    print("\n" + "="*70)
    print(" IMITATION PRE-TRAINING + RL FINE-TUNING PIPELINE")
    print("="*70)
    print(f"Experiment: {args.experiment_name}")
    print(f"Output directory: {experiment_dir}")
    print(f"\nPipeline:")
    print(f"  1. Generate demonstrations: {args.n_demo_episodes} episodes")
    print(f"  2. BC pre-training: {args.total_bc_epochs} epochs")
    print(f"  3. RL fine-tuning: {args.total_rl_timesteps:,} timesteps")
    print(f"Camera enabled: {args.use_camera}")
    print("="*70 + "\n")

    bc_policy = None

    # Phase 1: BC Pre-training (if not skipped)
    if not args.skip_bc and args.pretrained_model is None:
        # Collect or load demonstrations
        if args.sunfounder_demos:
            # Use Sunfounder PiDog expert demonstrations
            print(f"Loading Sunfounder PiDog demonstrations from {args.sunfounder_demos}")
            with open(args.sunfounder_demos, "rb") as f:
                transitions = pickle.load(f)
            print("✓ Loaded expert demonstrations from original PiDog gaits\n")
        elif demos_path.exists():
            print(f"Loading existing demonstrations from {demos_path}")
            with open(demos_path, "rb") as f:
                transitions = pickle.load(f)
        else:
            transitions = collect_hardcoded_demonstrations(
                n_episodes=args.n_demo_episodes,
                use_camera=args.use_camera
            )
            # Save for future use
            with open(demos_path, "wb") as f:
                pickle.dump(transitions, f)
            print(f"Saved demonstrations to {demos_path}\n")

        # Train BC
        bc_policy = train_bc_pretraining(transitions, experiment_dir, args)

    elif args.pretrained_model:
        print(f"\nLoading pretrained model from: {args.pretrained_model}")
        # Load pretrained BC policy
        import torch
        bc_policy = torch.load(args.pretrained_model)

    # Phase 2: Create RL environments
    print("Creating parallel environments for RL training...")
    if args.n_envs > 1:
        env = SubprocVecEnv([
            make_env(i, args.seed, args.use_camera)
            for i in range(args.n_envs)
        ])
    else:
        env = DummyVecEnv([make_env(0, args.seed, args.use_camera)])

    # Phase 3: Initialize RL model (with BC weights if available)
    model = create_rl_model_from_bc(bc_policy, env, args)

    # Phase 4: RL Fine-tuning
    print("\n" + "="*70)
    print(" PHASE 3: RL FINE-TUNING")
    print("="*70)
    print(f"Fine-tuning with PPO for {args.total_rl_timesteps:,} timesteps...")
    print("Starting from BC-initialized policy (should converge faster!)\n")

    # Set up callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=10000 // args.n_envs,
            save_path=str(checkpoint_dir),
            name_prefix="bc_rl_model",
        )
    ]

    if args.visualize_freq > 0:
        callbacks.append(
            VisualizeCallback(
                visualize_freq=args.visualize_freq,
                n_eval_episodes=2,
                render_fps=30,
                use_camera=args.use_camera,
                verbose=1,
            )
        )

    callback = CallbackList(callbacks)

    # Train
    try:
        model.learn(
            total_timesteps=args.total_rl_timesteps,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    # Save final model
    final_model_path = experiment_dir / "bc_rl_final_model"
    model.save(final_model_path)
    print(f"\n\nFinal model saved to: {final_model_path}")

    # Clean up
    env.close()

    print("\n" + "="*70)
    print(" PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {experiment_dir}")
    print(f"\nTo visualize the final trained model:")
    print(f"  python visualize_training.py\n")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
