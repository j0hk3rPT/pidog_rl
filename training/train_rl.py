"""Train PiDog using Reinforcement Learning (PPO from Stable-Baselines3)."""

import argparse
import os
from pathlib import Path
from datetime import datetime

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import torch

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pidog_env import PiDogEnv
from pidog_env.feature_extractors import PiDogCombinedExtractor, PiDogNatureCNNExtractor

# Try to import imitation for BC pretraining
try:
    from imitation.algorithms import bc
    from imitation.data.types import Transitions
    IMITATION_AVAILABLE = True
except ImportError:
    IMITATION_AVAILABLE = False
    print("Warning: 'imitation' package not available, BC pretraining disabled")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PiDog with RL")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "sac", "td3"],
        help="RL algorithm to use",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of steps per update (for PPO)",
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=1,
        help="Update the model every train_freq steps (for SAC/TD3). Higher = faster but less frequent updates.",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="Number of gradient steps per update (for SAC/TD3). -1 means as many as train_freq.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1_000_000,
        help="Replay buffer size (for SAC/TD3)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10_000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5_000,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of episodes for evaluation",
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
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training",
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
        "--use-camera",
        action="store_true",
        help="Enable camera observations (image input will be actual camera feed)",
    )
    parser.add_argument(
        "--disable-camera",
        action="store_true",
        help="Disable camera rendering (image input will be zeros for faster training)",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=84,
        help="Camera image width (default: 84)",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=84,
        help="Camera image height (default: 84)",
    )
    parser.add_argument(
        "--features-dim",
        type=int,
        default=256,
        help="Feature extractor output dimension (default: 256)",
    )
    parser.add_argument(
        "--use-nature-cnn",
        action="store_true",
        help="Use deeper Nature DQN architecture (slower but more powerful)",
    )
    parser.add_argument(
        "--pretrain-bc",
        action="store_true",
        help="Pretrain with behavioral cloning before RL",
    )
    parser.add_argument(
        "--bc-demos-path",
        type=str,
        default="datasets/sunfounder_demos.pkl",
        help="Path to demonstration data for BC pretraining",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=50,
        help="Number of BC pretraining epochs",
    )
    parser.add_argument(
        "--bc-gait",
        type=str,
        default="trot_forward",
        help="Specific gait to use from demos (or 'all')",
    )
    return parser.parse_args()


def make_env(rank, seed=0, use_camera=True, camera_width=84, camera_height=84):
    """Create a single environment instance."""
    def _init():
        env = PiDogEnv(
            use_camera=use_camera,
            camera_width=camera_width,
            camera_height=camera_height
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def load_demonstrations(demos_path, gait_name=None):
    """Load demonstration actions from pickle file."""
    import pickle

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

        # Get full cycle(s) - repeat multiple times for more training data
        cycle_length = 6 if 'trot' in gait_name else 49
        n_cycles = 10  # Use 10 cycles for more diverse transitions
        actions = actions[gait_indices[:cycle_length]]
        actions = np.tile(actions, (n_cycles, 1))  # Repeat the cycle
        print(f"Loaded {len(actions)} steps ({n_cycles} cycles) for gait '{gait_name}'")
    else:
        print(f"Loaded {len(actions)} total steps from all gaits")

    return actions


def collect_transitions(env, actions, n_repeats=5):
    """Execute actions in env to collect (obs, action, next_obs, done) transitions."""
    print(f"Collecting transitions by executing {len(actions)} actions {n_repeats} times...")

    obs_list = []
    acts_list = []
    next_obs_list = []
    dones_list = []

    for repeat in range(n_repeats):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Unwrap if reset returns (obs, info)

        for action in actions:
            # Handle dict observations (camera + vector) or simple vector observations
            if isinstance(obs, dict):
                # For dict obs, store as-is (BC can handle dict observations)
                obs_list.append(obs)
            else:
                # For vector obs, ensure float32
                obs_list.append(obs.astype(np.float32))

            acts_list.append(action.astype(np.float32))

            result = env.step(action)
            if len(result) == 5:
                next_obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = result

            # Handle dict vs vector observations
            if isinstance(next_obs, dict):
                next_obs_list.append(next_obs)
            else:
                next_obs_list.append(next_obs.astype(np.float32))

            dones_list.append(done)

            obs = next_obs

            if done:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]

        print(f"  Repeat {repeat + 1}/{n_repeats} complete")

    # Create transitions based on observation type
    if isinstance(obs_list[0], dict):
        # Dict observations - convert list of dicts to dict of arrays
        # Each key (e.g., 'image', 'vector') becomes a separate array
        obs_dict = {}
        next_obs_dict = {}

        # Get all keys from first observation
        keys = obs_list[0].keys()

        for key in keys:
            obs_dict[key] = np.array([obs[key] for obs in obs_list], dtype=np.float32)
            next_obs_dict[key] = np.array([obs[key] for obs in next_obs_list], dtype=np.float32)

        transitions = Transitions(
            obs=obs_dict,
            acts=np.array(acts_list, dtype=np.float32),
            infos=np.array([{}] * len(obs_list)),
            next_obs=next_obs_dict,
            dones=np.array(dones_list, dtype=bool),
        )
    else:
        # Vector observations - convert to numpy array
        transitions = Transitions(
            obs=np.array(obs_list, dtype=np.float32),
            acts=np.array(acts_list, dtype=np.float32),
            infos=np.array([{}] * len(obs_list)),
            next_obs=np.array(next_obs_list, dtype=np.float32),
            dones=np.array(dones_list, dtype=bool),
        )

    print(f"Collected {len(transitions)} transitions")
    return transitions


def pretrain_with_bc(model, transitions, args):
    """Pretrain policy using behavioral cloning."""
    if not IMITATION_AVAILABLE:
        print("ERROR: 'imitation' package required for BC pretraining")
        print("Install with: pip install imitation")
        return model

    print("\n" + "=" * 70)
    print("PRETRAINING WITH BEHAVIORAL CLONING")
    print("=" * 70)

    # Save original device
    original_device = model.device
    print(f"Model device: {original_device}")

    # Force BC training on CPU to avoid device placement issues with imitation library
    bc_device = 'cpu'
    print(f"BC training device: {bc_device} (imitation library works better on CPU)")

    # Move model to CPU for BC training
    if str(original_device) != 'cpu':
        print("Moving model to CPU for BC training...")
        model.policy.to(bc_device)

    # Get observation and action spaces from the model's env
    obs_space = model.observation_space
    act_space = model.action_space

    # Create BC trainer
    bc_trainer = bc.BC(
        observation_space=obs_space,
        action_space=act_space,
        demonstrations=transitions,
        rng=np.random.default_rng(42),
        device=bc_device,
        batch_size=64,
        policy=model.policy,  # Use the policy from our RL model
    )

    # Train
    print(f"\nTraining for {args.bc_epochs} epochs...")
    bc_trainer.train(n_epochs=args.bc_epochs)

    # Move model back to original device
    if str(original_device) != 'cpu':
        print(f"\nMoving model back to {original_device}...")
        model.policy.to(original_device)

    print("BC pretraining complete!")
    print("=" * 70 + "\n")

    return model


def create_algorithm(algorithm_name, env, args):
    """Create RL algorithm instance."""
    # Always use MultiInputPolicy for compatibility across training stages
    policy_type = "MultiInputPolicy"

    # Select feature extractor architecture
    if args.use_nature_cnn:
        extractor_class = PiDogNatureCNNExtractor
        print("Using Nature DQN architecture (deeper CNN)")
    else:
        extractor_class = PiDogCombinedExtractor
        print("Using standard combined CNN+MLP architecture")

    policy_kwargs = {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": {"features_dim": args.features_dim},
    }

    if not args.use_camera:
        print("Note: Camera disabled - image observations will be zeros")

    common_kwargs = {
        "env": env,
        "policy_kwargs": policy_kwargs,
        "learning_rate": args.learning_rate,
        "verbose": 1,
        "device": args.device,
        "tensorboard_log": args.log_dir,
        "seed": args.seed,
    }

    if algorithm_name == "ppo":
        return PPO(
            policy_type,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            **common_kwargs,
        )
    elif algorithm_name == "sac":
        return SAC(
            policy_type,
            batch_size=args.batch_size,
            gamma=0.99,
            tau=0.005,
            learning_starts=1000,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            buffer_size=args.buffer_size,
            optimize_memory_usage=True,  # Reduces memory by ~50% for image observations
            **common_kwargs,
        )
    elif algorithm_name == "td3":
        return TD3(
            policy_type,
            batch_size=args.batch_size,
            gamma=0.99,
            tau=0.005,
            learning_starts=1000,
            policy_delay=2,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            buffer_size=args.buffer_size,
            optimize_memory_usage=True,  # Reduces memory by ~50% for image observations
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


def main():
    """Main training function."""
    args = parse_args()

    # Handle camera flags (disable takes precedence)
    if args.disable_camera:
        args.use_camera = False

    # Set up experiment directory
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.algorithm}_{timestamp}"

    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    args.log_dir = str(experiment_dir / "logs")
    args.checkpoint_dir = str(experiment_dir / "checkpoints")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 60)
    print(f"Training PiDog with {args.algorithm.upper()}")
    print("=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel envs: {args.n_envs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Camera enabled: {args.use_camera}")
    if args.use_camera:
        print(f"Camera resolution: {args.camera_width}x{args.camera_height}")
        print(f"Feature dim: {args.features_dim}")
        print(f"Architecture: {'Nature DQN' if args.use_nature_cnn else 'Standard CNN+MLP'}")
    print(f"Output directory: {experiment_dir}")
    print("=" * 60)

    # Create vectorized environments
    print(f"\nCreating {args.n_envs} parallel environments...")
    if args.n_envs > 1:
        env = SubprocVecEnv([
            make_env(i, args.seed, args.use_camera, args.camera_width, args.camera_height)
            for i in range(args.n_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(0, args.seed, args.use_camera, args.camera_width, args.camera_height)
        ])

    # Create evaluation environment
    eval_env = DummyVecEnv([
        make_env(args.n_envs, args.seed, args.use_camera, args.camera_width, args.camera_height)
    ])

    # Create or load model
    if args.checkpoint is not None:
        print(f"\nLoading checkpoint from {args.checkpoint}...")
        if args.algorithm == "ppo":
            model = PPO.load(args.checkpoint, env=env, device=args.device)
        elif args.algorithm == "sac":
            model = SAC.load(args.checkpoint, env=env, device=args.device)
        elif args.algorithm == "td3":
            model = TD3.load(args.checkpoint, env=env, device=args.device)
    else:
        print(f"\nCreating new {args.algorithm.upper()} model...")
        model = create_algorithm(args.algorithm, env, args)

    # Print model info
    print(f"\nModel architecture:")
    print(model.policy)

    # BC Pretraining (if requested)
    if args.pretrain_bc and args.checkpoint is None:
        if args.use_camera:
            print("\nWARNING: BC pretraining with camera observations not yet supported")
            print("The imitation library has issues with dict observation spaces.")
            print("Skipping BC pretraining - will train from scratch with RL...")
            print("\nTIP: For BC pretraining, run without --use-camera, then fine-tune with camera:")
            print("  1. Train BC without camera: --pretrain-bc (no --use-camera)")
            print("  2. Fine-tune with camera: --checkpoint <bc_model> --use-camera")
        elif not IMITATION_AVAILABLE:
            print("\nWARNING: BC pretraining requested but 'imitation' not installed")
            print("Skipping BC pretraining...")
        else:
            print(f"\nLoading demonstrations for BC pretraining...")
            demos_path = Path(args.bc_demos_path)
            if not demos_path.exists():
                print(f"WARNING: Demo file not found: {demos_path}")
                print("Skipping BC pretraining...")
            else:
                # Load demonstrations
                demo_actions = load_demonstrations(str(demos_path), args.bc_gait)

                # Create single env for collecting transitions (no camera for BC)
                temp_env = PiDogEnv(
                    use_camera=False,
                    camera_width=args.camera_width,
                    camera_height=args.camera_height
                )

                # Collect transitions
                transitions = collect_transitions(temp_env, demo_actions, n_repeats=5)
                temp_env.close()

                # Pretrain with BC
                model = pretrain_with_bc(model, transitions, args)

                # Save BC pretrained model
                bc_model_path = experiment_dir / f"{args.algorithm}_bc_pretrained"
                model.save(bc_model_path)
                print(f"BC pretrained model saved to: {bc_model_path}")

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.n_envs,
        save_path=args.checkpoint_dir,
        name_prefix=f"{args.algorithm}_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.checkpoint_dir,
        log_path=args.log_dir,
        eval_freq=args.eval_freq // args.n_envs,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    # Train the model
    print(f"\nStarting training for {args.total_timesteps:,} timesteps...")
    print(f"TensorBoard logs: {args.log_dir}")
    print("Monitor training with: tensorboard --logdir=" + args.log_dir)
    print()

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # Save final model
    final_model_path = experiment_dir / f"{args.algorithm}_final_model"
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Clean up
    env.close()
    eval_env.close()

    print("\nTraining completed!")
    print(f"Results saved in: {experiment_dir}")


if __name__ == "__main__":
    main()
