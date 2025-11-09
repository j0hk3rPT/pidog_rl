# PiDog RL Training Workflows

Complete guide to all available training methods for PiDog quadruped robot.

## Overview

This repository provides multiple training approaches, from basic to advanced:

1. **RL from Scratch** - Standard reinforcement learning (slow but simple)
2. **RL with Real-Time Visualization** - See training progress as it happens
3. **Imitation Pre-training + RL** - **RECOMMENDED** - 3-4x faster convergence
4. **Imitation Only** - Quick baseline from demonstrations

## Quick Start (Recommended Workflow)

### Step 1: Extract Expert Demonstrations

Use the original Sunfounder PiDog gaits as expert demonstrations:

```bash
docker-compose run --rm pidog_rl python extract_sunfounder_demos.py \
    --n-cycles 20 \
    --output-file demonstrations/sunfounder_demos.pkl
```

**What this does:**
- Extracts walking and trotting gaits from official Sunfounder PiDog
- Converts coordinate-based control to joint angles
- Saves ~1000-1500 high-quality transitions

**Time:** ~2-3 minutes

### Step 2: Pre-train with Behavioral Cloning + Fine-tune with RL

```bash
docker-compose run --rm pidog_rl python train_pretrain_finetune.py \
    --sunfounder-demos demonstrations/sunfounder_demos.pkl \
    --total-bc-epochs 300 \
    --total-rl-timesteps 500000 \
    --visualize-freq 50000 \
    --n-envs 4
```

**What this does:**
1. Loads Sunfounder demonstrations
2. Pre-trains policy with Behavioral Cloning (5-10 min)
3. Fine-tunes with PPO reinforcement learning (30-60 min)
4. Shows real-time visualization every 50K steps

**Total time:** ~45-75 minutes

**Expected results:**
- Robot walks immediately (from BC)
- Rapidly improves speed and stability (from RL)
- Converges 3-4x faster than RL from scratch

### Step 3: Visualize Trained Model

```bash
docker-compose run --rm pidog_rl python visualize_training.py
```

**What this does:**
- Finds latest checkpoint
- Shows robot walking in real-time at 30 FPS
- Interactive 3D viewer

## Training Methods Comparison

| Method | Time to Walk | Training Time | Final Quality | Best For |
|--------|--------------|---------------|---------------|----------|
| **Imitation + RL** ⭐ | Immediate | 45-75 min | Excellent | **Production** |
| RL with Visualization | ~500K steps | 1-2 hours | Very Good | Learning/Debugging |
| RL from Scratch | ~500K steps | 1-2 hours | Good | Baseline |
| Imitation Only | Immediate | 10-15 min | Limited | Quick Demo |

⭐ **Recommended method**

## Detailed Workflows

### Workflow 1: Imitation Pre-training + RL (BEST)

Full pipeline with Sunfounder expert demonstrations:

```bash
# 1. Extract demonstrations (one-time setup)
python extract_sunfounder_demos.py \
    --n-cycles 20 \
    --output-file demonstrations/sunfounder_demos.pkl

# 2. Train with BC + RL
python train_pretrain_finetune.py \
    --sunfounder-demos demonstrations/sunfounder_demos.pkl \
    --total-bc-epochs 300 \
    --total-rl-timesteps 500000 \
    --visualize-freq 50000
```

**Advantages:**
- ✅ Fastest convergence (3-4x speedup)
- ✅ Uses proven gaits from official PiDog
- ✅ Better final performance
- ✅ More stable training
- ✅ Real-time visualization of progress

**See:** `IMITATION_PRETRAINING_GUIDE.md` for complete documentation

### Workflow 2: RL with Real-Time Visualization

Train with PPO and see progress during training:

```bash
python train_with_visualization.py \
    --total-timesteps 1000000 \
    --visualize-freq 50000 \
    --n-eval-episodes 2 \
    --n-envs 4
```

**Advantages:**
- ✅ See robot learning in real-time
- ✅ Catch problems early
- ✅ Monitor training progress visually
- ✅ Interactive viewer during training

**See:** `VISUALIZATION_GUIDE.md` for complete documentation

### Workflow 3: Standard RL from Scratch

Basic PPO training without pre-training:

```bash
python training/train_rl.py \
    --algorithm ppo \
    --total-timesteps 1000000 \
    --n-envs 4 \
    --save-freq 10000
```

**Advantages:**
- ✅ Simple and straightforward
- ✅ No dependencies on demonstrations
- ✅ Full control over training

**Disadvantages:**
- ❌ Slow initial progress
- ❌ Random initialization
- ❌ Requires more timesteps

### Workflow 4: Imitation Learning Only

Quick baseline using Behavioral Cloning:

```bash
# Extract demonstrations first
python extract_sunfounder_demos.py --n-cycles 30

# Train BC only
python training/train_imitation.py \
    --method bc \
    --expert-data demonstrations/sunfounder_demos.pkl \
    --n-epochs 500
```

**Advantages:**
- ✅ Very fast training (10 minutes)
- ✅ Robot walks immediately
- ✅ Good for demos

**Disadvantages:**
- ❌ Limited by demonstration quality
- ❌ Won't improve beyond demonstrations
- ❌ No reward optimization

## File Structure

```
pidog_rl/
├── README_TRAINING_WORKFLOWS.md       # This file - training overview
├── IMITATION_PRETRAINING_GUIDE.md    # Detailed BC→RL guide
├── VISUALIZATION_GUIDE.md             # Real-time visualization docs
│
├── extract_sunfounder_demos.py        # Extract official PiDog gaits
├── train_pretrain_finetune.py         # BC→RL pipeline (RECOMMENDED)
├── train_with_visualization.py        # RL with real-time checkpoints
├── visualize_training.py              # Visualize trained models
│
├── training/
│   ├── train_rl.py                    # Standard RL training
│   ├── train_imitation.py             # BC/GAIL training
│   └── visualization_callback.py      # Real-time viz callback
│
├── pidog_env/                         # Environment implementation
│   ├── pidog_env.py                   # Main environment
│   └── feature_extractors.py         # CNN feature extractors
│
└── model/
    └── pidog.xml                      # MuJoCo model definition
```

## Command Reference

### Extract Demonstrations

```bash
# Sunfounder gaits (recommended)
python extract_sunfounder_demos.py --n-cycles 20

# Alternative: hardcoded simple gait
# (automatically done in train_pretrain_finetune.py if needed)
```

### Training Commands

```bash
# RECOMMENDED: BC + RL with Sunfounder demos
python train_pretrain_finetune.py \
    --sunfounder-demos demonstrations/sunfounder_demos.pkl \
    --total-bc-epochs 300 \
    --total-rl-timesteps 500000

# RL with visualization
python train_with_visualization.py \
    --total-timesteps 500000 \
    --visualize-freq 50000

# Standard RL
python training/train_rl.py \
    --total-timesteps 1000000 \
    --n-envs 4

# BC only
python training/train_imitation.py \
    --method bc \
    --expert-data demonstrations/sunfounder_demos.pkl \
    --n-epochs 300
```

### Visualization Commands

```bash
# Visualize latest checkpoint
python visualize_training.py

# Visualize specific model
python visualize_training.py --checkpoint outputs/experiment/model.zip
```

## Environment Options

All training scripts support these common options:

```bash
--use-camera              # Use camera observations (slower, more powerful)
--n-envs N                # Number of parallel environments (default: 4)
--learning-rate F         # Learning rate (default: 3e-4)
--seed N                  # Random seed (default: 42)
--output-dir PATH         # Output directory (default: outputs)
--experiment-name NAME    # Experiment name (auto-generated if not set)
```

## Hardware Requirements

### Minimum (CPU only)
- 4 CPU cores
- 8 GB RAM
- Training time: ~1-2 hours for 500K steps

### Recommended (GPU)
- 8+ CPU cores
- 16 GB RAM
- AMD GPU with ROCm support
- Training time: ~30-60 min for 500K steps

## Troubleshooting

### Training is slow
```bash
# Increase parallel environments
--n-envs 8

# Use CPU-friendly settings
--n-steps 1024 --batch-size 32
```

### Robot falls immediately
```bash
# Use imitation pre-training (recommended)
python train_pretrain_finetune.py --sunfounder-demos demonstrations/sunfounder_demos.pkl

# Or train longer
--total-timesteps 2000000
```

### Visualization errors
```bash
# Check X11 forwarding
xhost +local:docker

# Verify DISPLAY
echo $DISPLAY

# See VISUALIZATION_GUIDE.md for more help
```

### BC doesn't improve RL
```bash
# Use more BC epochs
--total-bc-epochs 500

# Use Sunfounder demos instead of hardcoded
--sunfounder-demos demonstrations/sunfounder_demos.pkl

# Lower RL learning rate to preserve BC knowledge
--learning-rate 1e-4
```

## Tips for Best Results

### 1. Start with Recommended Workflow

Always start with imitation pre-training:
```bash
python extract_sunfounder_demos.py --n-cycles 20
python train_pretrain_finetune.py --sunfounder-demos demonstrations/sunfounder_demos.pkl
```

### 2. Monitor Training

Use visualization to catch problems early:
```bash
--visualize-freq 50000  # Show progress every 50K steps
```

### 3. Tune Hyperparameters

If training is unstable:
```bash
--learning-rate 1e-4    # Lower learning rate
--n-envs 8              # More parallel environments
--total-bc-epochs 500   # More BC pre-training
```

### 4. Use TensorBoard

Monitor metrics during training:
```bash
tensorboard --logdir outputs/experiment_name/logs
```

### 5. Save Checkpoints Frequently

```bash
--save-freq 5000  # Save every 5000 steps
```

## Next Steps

1. **Start training:** Follow "Quick Start" above
2. **Monitor progress:** Use TensorBoard and visualization
3. **Iterate:** Adjust hyperparameters based on results
4. **Deploy:** Test on real robot hardware

## Documentation

- `IMITATION_PRETRAINING_GUIDE.md` - Complete BC→RL workflow
- `VISUALIZATION_GUIDE.md` - Real-time visualization features
- `training/README.md` - Training implementation details
- `pidog_env/README.md` - Environment documentation

## Support

For issues or questions:
1. Check the troubleshooting sections in this guide
2. Review specific documentation files
3. Check GitHub issues

## Summary

**For fastest results and best performance:**

```bash
# One-time setup: Extract expert demonstrations
python extract_sunfounder_demos.py --n-cycles 20

# Train with BC + RL
python train_pretrain_finetune.py \
    --sunfounder-demos demonstrations/sunfounder_demos.pkl \
    --total-bc-epochs 300 \
    --total-rl-timesteps 500000 \
    --visualize-freq 50000

# Visualize results
python visualize_training.py
```

This gives you:
- ✅ Immediate walking ability (from BC)
- ✅ Rapid improvement (from RL)
- ✅ Real-time progress monitoring
- ✅ 3-4x faster than RL from scratch
