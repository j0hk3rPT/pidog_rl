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

### Level -1: Standing Only (NEW!)
**Goal**: Learn to stand stable before walking

- **No forward motion required** - rewards staying still
- Rewards: Height maintenance, upright posture, neutral joint positions
- Penalties: Movement, falling, leg collisions
- **Best for**: Complete beginners - establishes stable standing first
- **Duration**: 500K-1M timesteps (~15-30 min)

**Training command**:
```bash
python training/train_ppo.py \
  --curriculum-level -1 \
  --disable-camera \
  --total-timesteps 1_000_000 \
  --experiment-name standing_only
```

**What the robot learns**:
- ✅ Maintain upright balance
- ✅ Keep legs in neutral position
- ✅ Stay at correct height (0.14m)
- ✅ Minimal joint movement (efficient standing)
- ❌ No walking yet!

---

### Level 0: Basic Walking
**Goal**: Learn basic forward walking (after standing)

- Target velocity: 0.5 m/s
- Joint noise: ±0.1 rad
- Height variation: ±0.02m
- Initial orientation: Perfectly upright
- **Best for**: After learning to stand, or starting fresh with walking

**Training command**:
```bash
python training/train_ppo.py \
  --curriculum-level 0 \
  --checkpoint outputs/standing_only/ppo_final_model.zip \
  --disable-camera \
  --total-timesteps 2_000_000 \
  --learning-rate 1e-4 \
  --experiment-name basic_walking
```

**Note:** Load the standing checkpoint! This gives the robot a head start with balance.

### Level 1: Intermediate
**Goal**: Faster walking with initial perturbations

- Target velocity: 0.6 m/s (+20%)
- Joint noise: ±0.15 rad (+50% variation)
- Height variation: ±0.03m
- Initial orientation: Small tilt (±2.8°)
- **Best for**: After basic walking, before advanced training

**Training command**:
```bash
python training/train_ppo.py \
  --curriculum-level 1 \
  --checkpoint outputs/basic_walking/ppo_final_model.zip \
  --disable-camera \
  --total-timesteps 1_000_000 \
  --learning-rate 1e-4 \
  --experiment-name intermediate_walking
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
python training/train_ppo.py \
  --curriculum-level 2 \
  --checkpoint outputs/intermediate_walking/ppo_final_model.zip \
  --disable-camera \
  --total-timesteps 1_000_000 \
  --learning-rate 5e-5 \
  --experiment-name advanced_walking
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
python training/train_ppo.py \
  --curriculum-level 3 \
  --checkpoint outputs/advanced_walking/ppo_final_model.zip \
  --disable-camera \
  --total-timesteps 1_000_000 \
  --learning-rate 5e-5 \
  --experiment-name expert_walking
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

### Option A: Standing-First Curriculum (BEST for Beginners!)

**RECOMMENDED APPROACH**: Learn to stand → walk → advanced skills

This is the **fastest and most reliable** way to train a walking robot!

```bash
# Step 1: Standing only (15-30 min) - NO CHECKPOINT
python training/train_ppo.py \
  --curriculum-level -1 \
  --disable-camera \
  --total-timesteps 1_000_000 \
  --experiment-name standing_only

# Step 2: Basic walking (30-45 min) - LOAD STANDING CHECKPOINT
python training/train_ppo.py \
  --curriculum-level 0 \
  --checkpoint outputs/standing_only/ppo_final_model.zip \
  --disable-camera \
  --total-timesteps 2_000_000 \
  --learning-rate 1e-4 \
  --experiment-name basic_walking

# Step 3: Intermediate (20-30 min) - LOAD WALKING CHECKPOINT
python training/train_ppo.py \
  --curriculum-level 1 \
  --checkpoint outputs/basic_walking/ppo_final_model.zip \
  --disable-camera \
  --total-timesteps 1_000_000 \
  --learning-rate 1e-4 \
  --experiment-name intermediate_walking

# Step 4: Advanced (20-30 min) - LOAD INTERMEDIATE CHECKPOINT
python training/train_ppo.py \
  --curriculum-level 2 \
  --checkpoint outputs/intermediate_walking/ppo_final_model.zip \
  --disable-camera \
  --total-timesteps 1_000_000 \
  --learning-rate 5e-5 \
  --experiment-name advanced_walking

# Step 5: Expert (20-30 min) - LOAD ADVANCED CHECKPOINT
python training/train_ppo.py \
  --curriculum-level 3 \
  --checkpoint outputs/advanced_walking/ppo_final_model.zip \
  --disable-camera \
  --total-timesteps 1_000_000 \
  --learning-rate 5e-5 \
  --experiment-name expert_walking
```

**Total time**: ~90-150 minutes for expert-level policy

**Why standing first?**
- ✅ **Faster convergence**: Standing is easier, robot learns balance quickly
- ✅ **Better foundation**: Each level builds on previous checkpoint
- ✅ **Progressive difficulty**: Gradual increase in challenge
- ✅ **Less frustration**: Robot falls less when it starts with balance

---

### Option B: Direct Walking (Skip Standing)

If you want to skip standing and go straight to walking (NOT recommended):

```bash
python training/train_ppo.py \
  --curriculum-level 0 \
  --disable-camera \
  --total-timesteps 2_500_000 \
  --experiment-name direct_walking
```

**Pros**: Simpler (one command)
**Cons**:
- ⚠️ Slower convergence (no standing foundation)
- ⚠️ Robot falls more initially
- ⚠️ May take 50% longer to learn walking

---

### Option C: Two-Phase + Standing-First (Ultimate Performance)

Combine curriculum learning with two-phase training for fastest results:

**Phase 1: Standing (no camera, super fast)**
```bash
python training/train_ppo.py \
  --curriculum-level -1 \
  --disable-camera \
  --total-timesteps 1_000_000 \
  --experiment-name phase1_standing
```

**Phase 2: Walking (no camera, fast) - BUILD ON STANDING**
```bash
python training/train_ppo.py \
  --curriculum-level 0 \
  --checkpoint outputs/phase1_standing/ppo_final_model.zip \
  --disable-camera \
  --total-timesteps 2_000_000 \
  --learning-rate 1e-4 \
  --experiment-name phase2_walking
```

**Phase 3: Visual + advanced curriculum - BUILD ON WALKING**
```bash
python training/train_ppo.py \
  --curriculum-level 2 \
  --checkpoint outputs/phase2_walking/ppo_final_model.zip \
  --use-camera \
  --camera-width 64 \
  --camera-height 64 \
  --total-timesteps 1_500_000 \
  --learning-rate 5e-5 \
  --experiment-name phase3_visual_advanced
```

**Total time**: ~60-90 minutes for robust visual walking policy!

**Key insight**: Each phase builds on the previous checkpoint!

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

**Solution**: Train longer at lower levels first OR check you're loading checkpoints

```bash
# Make sure you're LOADING THE CHECKPOINT from previous level!
python training/train_ppo.py \
  --curriculum-level 1 \
  --checkpoint outputs/basic_walking/ppo_final_model.zip \  # IMPORTANT!
  --total-timesteps 1_500_000  # Train longer if needed
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
python training/train_ppo.py \
  --curriculum-level 0 \
  --no-domain-randomization \
  --disable-camera \
  --total-timesteps 2_000_000

# Phase 2: Fine-tune with randomization for robustness
python training/train_ppo.py \
  --curriculum-level 1 \
  --checkpoint outputs/.../ppo_final_model.zip \
  --domain-randomization \
  --disable-camera \
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
