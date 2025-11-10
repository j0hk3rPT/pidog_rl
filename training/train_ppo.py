"""Train PiDog using PPO (Proximal Policy Optimization) from Stable-Baselines3.

This script is optimized for quadruped locomotion with research-backed hyperparameters.
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import torch

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pidog_env import PiDogEnv
from pidog_env.feature_extractors import PiDogFlattenedExtractor

# Try to import sb3-extra-buffers for compression
try:
    from sb3_extra_buffers.compressed import (
        CompressedRolloutBuffer,
        find_buffer_dtypes
    )
    SB3_EXTRA_BUFFERS_AVAILABLE = True
except ImportError:
    SB3_EXTRA_BUFFERS_AVAILABLE = False
    print("Info: 'sb3-extra-buffers' not available, compression disabled"
          " (install with: pip install 'sb3-extra-buffers[fast,extra]')")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PiDog with PPO - Optimized for Quadruped Locomotion"
    )

    # Training parameters (optimized defaults for quadruped)
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2_000_000,
        help="Total training timesteps (default: 2M)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=16,
        help="Number of parallel environments (default: 16, optimal for CPU parallelization)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4, standard for PPO)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256, larger for stable continuous control)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=4096,
        help="Steps per rollout (default: 4096, captures full gait cycles)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of epochs for policy updates (default: 10)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda for advantage estimation (default: 0.95)",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clip range (default: 0.2)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.0,
        help="Entropy coefficient (default: 0.0, focus on exploitation)",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="Value function coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Max gradient norm for clipping (default: 0.5)",
    )

    # Checkpointing and evaluation
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50_000,
        help="Save checkpoint every N steps (default: 50K)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=25_000,
        help="Evaluate every N steps (default: 25K)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation (default: 10)",
    )

    # General settings
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use: cpu/cuda/auto (default: cpu)",
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
        help="Output directory for logs and checkpoints (default: outputs/)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated with timestamp)",
    )

    # Camera settings
    parser.add_argument(
        "--use-camera",
        action="store_true",
        help="Enable camera observations (slower but visual)",
    )
    parser.add_argument(
        "--disable-camera",
        action="store_true",
        help="Disable camera rendering for faster training (recommended for initial training)",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=64,
        help="Camera image width (default: 64, smaller=faster)",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=64,
        help="Camera image height (default: 64, smaller=faster)",
    )
    parser.add_argument(
        "--features-dim",
        type=int,
        default=256,
        help="Feature extractor output dimension (default: 256)",
    )

    # Curriculum learning and domain randomization
    parser.add_argument(
        "--curriculum-level",
        type=int,
        default=0,
        choices=[-1, 0, 1, 2, 3],
        help="Curriculum level: -1=standing, 0=basic walk, 1-3=progressive (default: 0)",
    )
    parser.add_argument(
        "--domain-randomization",
        action="store_true",
        default=True,
        help="Enable domain randomization (default: True)",
    )
    parser.add_argument(
        "--no-domain-randomization",
        action="store_false",
        dest="domain_randomization",
        help="Disable domain randomization",
    )

    # Memory optimization (optional)
    parser.add_argument(
        "--use-compression",
        action="store_true",
        help="Use compressed rollout buffer (requires sb3-extra-buffers, saves 70-95%% RAM)",
    )
    parser.add_argument(
        "--compression-method",
        type=str,
        default="zstd-3",
        help="Compression method (default: zstd-3, options: zstd-5, lz4-frame/1)",
    )

    return parser.parse_args()


def make_env(rank, seed=0, use_camera=True, camera_width=64, camera_height=64,
             domain_randomization=True, curriculum_level=0):
    """Create a single environment instance."""
    def _init():
        env = PiDogEnv(
            use_camera=use_camera,
            camera_width=camera_width,
            camera_height=camera_height,
            domain_randomization=domain_randomization,
            curriculum_level=curriculum_level,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def create_ppo_model(env, args, buffer_dtypes=None):
    """Create PPO model with optimized hyperparameters."""

    # Policy configuration
    policy_kwargs = {
        "features_extractor_class": PiDogFlattenedExtractor,
        "features_extractor_kwargs": {
            "features_dim": args.features_dim,
            "camera_width": args.camera_width,
            "camera_height": args.camera_height,
        },
    }

    # Common PPO kwargs
    ppo_kwargs = {
        "policy": "CnnPolicy",
        "env": env,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "device": args.device,
        "tensorboard_log": args.log_dir,
        "seed": args.seed,
    }

    # Add compressed rollout buffer if requested
    if args.use_compression:
        if not SB3_EXTRA_BUFFERS_AVAILABLE:
            raise ImportError(
                "Compression requested but sb3-extra-buffers not installed. "
                "Install with: pip install 'sb3-extra-buffers[fast,extra]'"
            )

        if buffer_dtypes is None:
            raise ValueError("buffer_dtypes must be provided when using compression!")

        ppo_kwargs["rollout_buffer_class"] = CompressedRolloutBuffer
        ppo_kwargs["rollout_buffer_kwargs"] = {
            "dtypes": buffer_dtypes,
            "compression_method": args.compression_method,
        }

        buffer_size = args.n_steps * args.n_envs
        print(f"✓ Using CompressedRolloutBuffer (size={buffer_size:,} = {args.n_steps}*{args.n_envs})")
        print(f"  Compression: {args.compression_method}")

    return PPO(**ppo_kwargs)


def main():
    """Main training function."""
    args = parse_args()

    # Handle camera flags
    if args.disable_camera:
        args.use_camera = False

    # Set up experiment directory
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        curriculum_name = {-1: "standing", 0: "walk", 1: "inter", 2: "adv", 3: "expert"}.get(args.curriculum_level, "unk")
        args.experiment_name = f"ppo_{curriculum_name}_{timestamp}"

    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    args.log_dir = str(experiment_dir / "logs")
    args.checkpoint_dir = str(experiment_dir / "checkpoints")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 70)
    print("TRAINING PIDOG WITH PPO (Optimized for Quadruped Locomotion)")
    print("=" * 70)
    print(f"Experiment: {args.experiment_name}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel envs: {args.n_envs}")
    print(f"Curriculum level: {args.curriculum_level} " +
          {-1: "(standing)", 0: "(walk)", 1: "(inter)", 2: "(adv)", 3: "(expert)"}.get(args.curriculum_level, ""))
    print(f"Domain randomization: {'Enabled' if args.domain_randomization else 'Disabled'}")
    print(f"Camera: {'Enabled' if args.use_camera else 'Disabled'}")
    if args.use_camera:
        print(f"  Resolution: {args.camera_width}x{args.camera_height}")
    print()
    print("PPO Hyperparameters (Research-Optimized for Quadruped):")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Rollout steps: {args.n_steps}")
    print(f"  Epochs per update: {args.n_epochs}")
    print(f"  Gamma: {args.gamma}")
    print(f"  GAE Lambda: {args.gae_lambda}")
    print(f"  Clip range: {args.clip_range}")
    print(f"  Entropy coef: {args.ent_coef}")
    print(f"  Value function coef: {args.vf_coef}")
    print(f"  Max grad norm: {args.max_grad_norm}")
    print(f"Output: {experiment_dir}")
    print("=" * 70)

    # Initialize compression BEFORE creating environments (if needed)
    buffer_dtypes = None
    if args.use_compression:
        print("\nInitializing compression...")
        obs_size = args.camera_height * args.camera_width * 3 + 31
        buffer_dtypes = find_buffer_dtypes(
            obs_shape=(obs_size,),
            elem_dtype=np.float32,
            compression_method=args.compression_method
        )
        print(f"✓ Compression initialized (obs_shape={obs_size}, method={args.compression_method})")

    # Create vectorized environments
    print(f"\nCreating {args.n_envs} parallel environments...")
    curriculum_desc = {-1: "standing only", 0: "basic walking", 1: "intermediate",
                      2: "advanced", 3: "expert"}
    print(f"  Observation: Box (21199,) - flattened image + sensors")
    print(f"  Camera: {'Enabled' if args.use_camera else 'Disabled (zeros)'}")
    print(f"  Curriculum: Level {args.curriculum_level} ({curriculum_desc.get(args.curriculum_level, 'unknown')})")

    VecEnvClass = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv

    env = VecEnvClass([
        make_env(i, args.seed, args.use_camera, args.camera_width, args.camera_height,
                args.domain_randomization, args.curriculum_level)
        for i in range(args.n_envs)
    ])

    # Create evaluation environment (same type as training)
    eval_env = VecEnvClass([
        make_env(args.n_envs, args.seed, args.use_camera, args.camera_width, args.camera_height,
                args.domain_randomization, args.curriculum_level)
    ])

    # Create or load model
    if args.checkpoint is not None:
        print(f"\nLoading checkpoint from {args.checkpoint}...")
        model = PPO.load(args.checkpoint, env=env, device=args.device)
        print("✓ Checkpoint loaded successfully")
    else:
        print("\nCreating new PPO model...")
        model = create_ppo_model(env, args, buffer_dtypes=buffer_dtypes)
        print("✓ Model created successfully")

    # Print model architecture
    print(f"\nModel architecture:")
    print(model.policy)

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.n_envs,
        save_path=args.checkpoint_dir,
        name_prefix="ppo_model",
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
    print(f"\n{'='*70}")
    print(f"STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Expected updates: {args.total_timesteps // (args.n_steps * args.n_envs):,}")
    print(f"TensorBoard: tensorboard --logdir={args.log_dir}")
    print()

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user.")

    # Save final model
    final_model_path = experiment_dir / "ppo_final_model"
    model.save(final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")

    # Clean up
    env.close()
    eval_env.close()

    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"Results saved in: {experiment_dir}")
    print(f"\nNext steps:")
    print(f"  1. Test model: python training/test_trained_model.py {final_model_path}.zip")
    print(f"  2. TensorBoard: tensorboard --logdir={args.log_dir}")
    print(f"  3. Continue training: --checkpoint {final_model_path}.zip")
    print()


if __name__ == "__main__":
    main()
