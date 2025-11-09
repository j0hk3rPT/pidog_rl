# PiDog RL - Reinforcement Learning for Quadruped Robots

A comprehensive reinforcement learning and imitation learning framework for training the PiDog quadruped robot using MuJoCo physics simulation. This project supports both traditional RL algorithms (PPO, SAC, TD3) and imitation learning methods (Behavioral Cloning, GAIL).

## Features

- **MuJoCo Physics Simulation**: High-fidelity physics simulation of PiDog quadruped robot
- **Gymnasium Environment**: Standard RL interface for easy integration with various algorithms
- **Multiple RL Algorithms**: PPO, SAC, TD3 via Stable-Baselines3
- **Imitation Learning**: Behavioral Cloning and GAIL support
- **JAX/Brax Support**: Ready for JAX-based training workflows
- **Docker Support**: Pre-configured Docker with ROCm/PyTorch for AMD GPUs
- **Comprehensive Tooling**: Training, evaluation, and demonstration collection scripts

## Project Structure

```
pidog_rl/
├── model/                 # MuJoCo robot models and meshes
│   ├── pidog.xml         # Main robot model
│   └── assets/           # 3D meshes and textures
├── pidog_env/            # Gymnasium environment
│   ├── __init__.py
│   └── pidog_env.py      # PiDog environment implementation
├── training/             # Training scripts
│   ├── train_rl.py       # RL training (PPO/SAC/TD3)
│   ├── train_imitation.py # Imitation learning (BC/GAIL)
│   ├── evaluate.py       # Model evaluation
│   └── collect_demos.py  # Collect expert demonstrations
├── configs/              # Training configurations
│   ├── ppo_default.yaml
│   └── imitation_bc.yaml
├── tests/                # Test scripts
│   └── sit.py           # Environment visualization test
├── build/                # Build automation
├── tools/                # Utility scripts
├── scripts/              # Helper scripts
│   └── quickstart.sh    # Quick start guide
├── Dockerfile            # Docker configuration (ROCm/PyTorch)
├── docker-compose.yml    # Docker Compose setup
└── pyproject.toml        # Python dependencies
```

## Quick Start

### Option 1: Docker (Recommended for AMD GPUs)

```bash
# Clone the repository
git clone git@github.com:Joel-Baptista/pidog_rl.git
cd pidog_rl

# Build and run with Docker Compose
docker-compose up -d pidog_rl

# Enter the container
docker exec -it pidog_rl_training bash

# Inside container: Run quick start script
./scripts/quickstart.sh
```

### Option 2: Local Installation (Linux)

```bash
# Install UV package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone git@github.com:Joel-Baptista/pidog_rl.git
cd pidog_rl

# Install dependencies
uv sync

# Test the environment
uv run python tests/sit.py
```

## Prerequisites

### For Docker Setup
- Docker and Docker Compose
- AMD GPU with ROCm support (for GPU acceleration)
- Linux system

### For Local Installation
- Python 3.11+
- [UV package manager](https://docs.astral.sh/uv/)
- MuJoCo dependencies (GLEW, OSMesa)

```bash
# Install MuJoCo dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y libglew-dev libosmesa6-dev libgl1-mesa-glx libglfw3
```

## Training

### 1. Collect Expert Demonstrations (for Imitation Learning)

```bash
# Collect demonstrations using hardcoded walking gait
uv run python training/collect_demos.py \
    --n-episodes 20 \
    --output-path datasets/expert_demos.pkl

# Or collect from a trained policy
uv run python training/collect_demos.py \
    --policy-path outputs/ppo_final_model.zip \
    --n-episodes 50 \
    --output-path datasets/expert_demos.pkl
```

### 2. Train with Reinforcement Learning

```bash
# Train with PPO (default)
uv run python training/train_rl.py \
    --algorithm ppo \
    --total-timesteps 1000000 \
    --n-envs 4 \
    --learning-rate 0.0003

# Train with SAC
uv run python training/train_rl.py \
    --algorithm sac \
    --total-timesteps 500000 \
    --learning-rate 0.0003

# Resume from checkpoint
uv run python training/train_rl.py \
    --algorithm ppo \
    --checkpoint outputs/ppo_20240101_120000/checkpoints/ppo_model_100000_steps.zip
```

### 3. Train with Imitation Learning

```bash
# Behavioral Cloning
uv run python training/train_imitation.py \
    --method bc \
    --expert-data datasets/expert_demos.pkl \
    --n-epochs 100 \
    --batch-size 64

# GAIL (Generative Adversarial Imitation Learning)
uv run python training/train_imitation.py \
    --method gail \
    --expert-data datasets/expert_demos.pkl \
    --total-timesteps 100000
```

### 4. Monitor Training

```bash
# TensorBoard
tensorboard --logdir=outputs/

# Access at http://localhost:6006
```

## Evaluation

```bash
# Evaluate trained model
uv run python training/evaluate.py \
    --model-path outputs/ppo_final_model.zip \
    --algorithm ppo \
    --n-episodes 10 \
    --render

# Record video
uv run python training/evaluate.py \
    --model-path outputs/ppo_final_model.zip \
    --algorithm ppo \
    --n-episodes 5 \
    --record-video outputs/pidog_walking.mp4
```

## Docker Usage

### Build Docker Image

```bash
# Build the image
docker-compose build

# Or build manually
docker build -t pidog_rl:latest .
```

### Run Training in Docker

```bash
# Start container
docker-compose up -d pidog_rl

# Enter container
docker exec -it pidog_rl_training bash

# Inside container: Train
uv run python training/train_rl.py --algorithm ppo --total-timesteps 100000

# View TensorBoard (access at http://localhost:6006)
docker-compose up -d tensorboard
```

### GPU Support (ROCm)

The Docker setup is configured for AMD GPUs with ROCm. For specific GPU versions:

```bash
# Set your GPU version (if needed)
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Run with GPU
docker-compose up -d pidog_rl
```

## Environment Details

The PiDog environment (`pidog_env.PiDogEnv`) implements the Gymnasium API:

**Observation Space** (27-dimensional):
- Joint positions (8): 4 legs × 2 joints (shoulder, knee)
- Joint velocities (8)
- Body orientation (4): quaternion
- Body linear velocity (3)
- Body angular velocity (3)
- Body height (1)

**Action Space** (8-dimensional):
- Target joint positions for 8 leg joints (normalized to [-1, 1])

**Rewards**:
- Forward velocity (target: 0.5 m/s)
- Height maintenance (target: 0.15 m)
- Stability (upright orientation)
- Energy efficiency (action penalty)

## Adding Custom Parts to PiDog

To add custom parts or modify meshes:

1. Place new meshes inside the `model/` folder
2. Run `./build/meshes.sh` to process meshes
3. Edit `model/pidog.xml` to add/replace geometries (reference `model/meshes.xml` for names)
4. Test changes: `uv run python tests/sit.py`

**Tip**: Use Blender for mesh manipulation before importing to MuJoCo

## Running Tests

```bash
# Test environment visualization
uv run python tests/sit.py

# Test environment step-by-step
uv run python -c "
import gymnasium as gym
from pidog_env import PiDogEnv
env = PiDogEnv()
obs, _ = env.reset()
print('Observation shape:', obs.shape)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print('Step successful! Reward:', reward)
"
```

## Troubleshooting

### MuJoCo Rendering Issues
```bash
# Use OSMesa for headless rendering
export MUJOCO_GL=osmesa

# Or EGL for GPU-accelerated rendering
export MUJOCO_GL=egl
```

### ROCm GPU Not Detected
```bash
# Check ROCm installation
rocm-smi

# Set GPU override if needed
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

## Built With

- [MuJoCo](https://mujoco.org/) - Physics simulation engine
- [Gymnasium](https://gymnasium.farama.org/) - RL environment interface
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [Imitation](https://imitation.readthedocs.io/) - Imitation learning library
- [JAX](https://jax.readthedocs.io/) & [Brax](https://github.com/google/brax) - JAX-based training
- [ROCm](https://rocm.docs.amd.com/) & [PyTorch](https://pytorch.org/) - Deep learning framework

## License

This project is licensed under the [CC0 1.0 Universal](LICENSE.md) Creative Commons License - see the LICENSE.md file for details

## Acknowledgments

- MuJoCo team for the excellent physics engine
- Stable-Baselines3 contributors
- PiDog robot design inspiration
