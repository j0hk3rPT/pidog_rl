# Curriculum Learning & Domain Randomization Guide

This guide explains the curriculum learning and domain randomization features for robust PiDog locomotion training.

## Overview

The training system now uses:

1. **Contact-force based fall detection** - Only terminates on actual falls (torso contact > 1.0N)
2. **Fall-specific penalties** - Penalizes actual falls, not early terminations
3. **Curriculum learning** - Progressive difficulty levels (0-3)
4. **Domain randomization** - Randomizes physics parameters for sim-to-real transfer

## Key Philosophy

**Let the robot learn from near-falls and recovery**

- Old approach: Terminate early when tilted > 50° → Robot never learns recovery
- New approach: Only terminate on hard falls → Robot learns to recover from tilts

## Curriculum Levels

### Level 0: Beginner (Default)
**Goal**: Learn basic forward walking

- Target velocity: 0.5 m/s
- Joint noise: ±0.1 rad
- Height variation: ±0.02m
- Initial orientation: Perfectly upright
- **Best for**: Initial training from scratch

**Training command**:
```bash
python training/train_rl.py \
  --curriculum-level 0 \
  --total-timesteps 2_000_000 \
  --n-envs 16
```

### Level 1: Intermediate
**Goal**: Faster walking with initial perturbations

- Target velocity: 0.6 m/s (+20%)
- Joint noise: ±0.15 rad (+50% variation)
- Height variation: ±0.03m
- Initial orientation: Small tilt (±2.8°)
- **Best for**: Fine-tuning after basic walking works

**Training command**:
```bash
python training/train_rl.py \
  --curriculum-level 1 \
  --checkpoint outputs/level0/ppo_final_model.zip \
  --total-timesteps 1_000_000 \
  --learning-rate 1e-4
```

### Level 2: Advanced
**Goal**: Robust walking under challenging conditions

- Target velocity: 0.7 m/s (+40%)
- Joint noise: ±0.2 rad (+100% variation)
- Height variation: ±0.04m
- Initial orientation: Moderate tilt (±5.7°)
- **Best for**: Training robust recovery behaviors

**Training command**:
```bash
python training/train_rl.py \
  --curriculum-level 2 \
  --checkpoint outputs/level1/ppo_final_model.zip \
  --total-timesteps 1_000_000 \
  --learning-rate 5e-5
```

### Level 3: Expert
**Goal**: Maximum robustness and speed

- Target velocity: 0.8 m/s (+60%)
- Joint noise: ±0.25 rad (+150% variation)
- Height variation: ±0.05m
- Initial orientation: Large tilt (±8.5°)
- **Best for**: Final hardening before deployment

**Training command**:
```bash
python training/train_rl.py \
  --curriculum-level 3 \
  --checkpoint outputs/level2/ppo_final_model.zip \
  --total-timesteps 1_000_000 \
  --learning-rate 5e-5
```

## Domain Randomization

Domain randomization helps with **sim-to-real transfer** by training on diverse physics parameters.

### Randomized Parameters

| Parameter | Variation | Why? |
|-----------|-----------|------|
| Body mass | ±20% | Battery charge, payload, manufacturing tolerance |
| Ground friction | ±30% | Different floor surfaces (tile, carpet, wood) |
| Joint damping | ±25% | Servo wear, temperature effects |
| Actuator gains | ±15% | Voltage variation, servo aging |

### Enabling/Disabling

Domain randomization is **enabled by default**:

```bash
# Enabled (default)
python training/train_rl.py --domain-randomization

# Disabled (for debugging or comparing)
python training/train_rl.py --no-domain-randomization
```

## Fall Detection

### Old Approach (Removed)
- Terminated when:
  - Height < 0.05m
  - Tilt > 50°
  - Quaternion w < 0.5
  - Belly touches ground

**Problem**: Robot never learned to recover from tilts

### New Approach (Current)
- Only terminates when: **Torso touches ground**

**Benefits**:
- Simple and reliable detection
- Robot can recover from large tilts
- Learns robust balancing behaviors
- Natural curriculum: Easy poses → harder poses
- Fall penalty (-20.0) only applied to actual falls

### How It Works

```python
# In pidog_env.py:
def _detect_torso_fall(self):
    """Detect actual fall: torso/body touching ground."""
    # Checks if any torso geom is in contact with ground:
    # - chest_c0, chest_c1, chest_c2
    # - torso_left_c0, torso_left_c1, torso_left_c2
    # - torso_right_c0, torso_right_c1, torso_right_c2
    # - body_c0, body_c1, body_c2

    # If torso touches ground → FALL
    # Otherwise → Keep going, learn to recover!
```

## Recommended Training Progression

### Option A: Progressive Curriculum (Best Results)

Train through all levels sequentially:

```bash
# Level 0: Basic walking (30-60 min)
python training/train_rl.py \
  --curriculum-level 0 \
  --total-timesteps 2_000_000 \
  --n-envs 16 \
  --experiment-name curriculum_level0

# Level 1: Initial perturbations (20-30 min)
python training/train_rl.py \
  --curriculum-level 1 \
  --checkpoint outputs/curriculum_level0/ppo_final_model.zip \
  --total-timesteps 1_000_000 \
  --learning-rate 1e-4 \
  --experiment-name curriculum_level1

# Level 2: Robust recovery (20-30 min)
python training/train_rl.py \
  --curriculum-level 2 \
  --checkpoint outputs/curriculum_level1/ppo_final_model.zip \
  --total-timesteps 1_000_000 \
  --learning-rate 5e-5 \
  --experiment-name curriculum_level2

# Level 3: Expert performance (20-30 min)
python training/train_rl.py \
  --curriculum-level 3 \
  --checkpoint outputs/curriculum_level2/ppo_final_model.zip \
  --total-timesteps 1_000_000 \
  --learning-rate 5e-5 \
  --experiment-name curriculum_level3
```

**Total time**: ~90-150 minutes for expert-level policy

### Option B: Jump to Level 1 (Faster)

If you have limited time, start at Level 1:

```bash
python training/train_rl.py \
  --curriculum-level 1 \
  --total-timesteps 2_500_000 \
  --n-envs 16
```

### Option C: Two-Phase + Curriculum (Best of Both)

Combine curriculum learning with two-phase training:

**Stage 1: Fast proprioceptive (Level 0)**
```bash
python training/train_rl.py \
  --disable-camera \
  --curriculum-level 0 \
  --total-timesteps 2_000_000 \
  --n-envs 16 \
  --experiment-name stage1_curriculum0
```

**Stage 2: Visual + harder curriculum (Level 2)**
```bash
python training/train_rl.py \
  --use-camera \
  --camera-width 64 --camera-height 64 \
  --curriculum-level 2 \
  --checkpoint outputs/stage1_curriculum0/ppo_final_model.zip \
  --total-timesteps 1_500_000 \
  --n-envs 8 \
  --learning-rate 1e-4 \
  --experiment-name stage2_curriculum2_visual
```

## Monitoring Training

### TensorBoard Metrics

```bash
tensorboard --logdir=outputs/
```

Key metrics to watch:
- `rollout/ep_rew_mean`: Should increase steadily
- `rollout/ep_len_mean`: Longer episodes = more stable
- `custom/fall_rate`: Track how often actual falls occur (should decrease)

### Episode Info

The environment now tracks:
- `has_fallen`: Boolean flag for actual fall detection
- `curriculum_level`: Current difficulty level
- `forward_velocity`: Instantaneous speed
- `body_height`: Current height

## Troubleshooting

### Robot keeps falling at higher levels

**Solution**: Train longer at lower levels first
```bash
# Extend Level 0 training
python training/train_rl.py \
  --curriculum-level 0 \
  --total-timesteps 3_000_000  # Increased from 2M
```

### No falls detected (episodes run full 5000 steps)

**Check**: Verify torso geoms exist in your model
- Run with debug logging to see if contacts are being detected
- Check that geom names match in your MJCF model:
  - `chest_c0`, `torso_left_c0`, `body_c0`, etc.

### Domain randomization makes training unstable

**Solution**: Disable initially, enable for fine-tuning
```bash
# Phase 1: Learn basic policy without randomization
python training/train_rl.py \
  --no-domain-randomization \
  --total-timesteps 2_000_000

# Phase 2: Fine-tune with randomization
python training/train_rl.py \
  --domain-randomization \
  --checkpoint outputs/.../ppo_final_model.zip \
  --learning-rate 1e-4 \
  --total-timesteps 1_000_000
```

### Robot not recovering from tilts

**This is expected early in training!**

The robot learns recovery behaviors gradually. If after 2M+ timesteps at Level 1+ you still don't see recovery:

1. Check fall detection is working: Add logging to `_detect_torso_fall()`
2. Increase upright stability reward in `_compute_reward()` (try 3.0x instead of 2.0x)
3. Reduce fall penalty (try -10.0 instead of -20.0) to encourage exploration

## Next Steps

After curriculum training:
1. **Evaluate**: Use `scripts/visualize_walk.py` to see the policy in action
2. **Real hardware**: Export with `scripts/export_for_pi.py`
3. **Further tuning**: Try BC pretraining with `--pretrain-bc` at Level 0

For more details:
- Two-phase training: See `TWO_STAGE_TRAINING.md`
- SAC memory optimization: See `docs/SAC_MEMORY_GUIDE.md`
- Quick start: See `QUICKSTART.md`
