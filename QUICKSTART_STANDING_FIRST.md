# Quick Start: Standing-First Training

The **fastest way** to train a walking robot: teach it to stand first!

## The Progression (Each builds on previous!)

```
Standing (-1) → Walking (0) → Intermediate (1) → Advanced (2) → Expert (3)
   ↓              ↓              ↓                  ↓              ↓
  1M steps      2M steps       1M steps          1M steps       1M steps
  15-30 min     30-45 min      20-30 min         20-30 min      20-30 min
```

**Total time**: ~90-150 minutes for expert walking

## Step-by-Step Commands

### 1️⃣ Standing Only (Start Here!)

```bash
python training/train_ppo.py \
  --curriculum-level -1 \
  --disable-camera \
  --total-timesteps 1_000_000 \
  --experiment-name standing_only
```

**What this does**: Robot learns to stand upright and balanced
**Time**: ~15-30 minutes
**No checkpoint needed** - training from scratch

---

### 2️⃣ Basic Walking (Build on Standing)

```bash
python training/train_ppo.py \
  --curriculum-level 0 \
  --checkpoint outputs/standing_only/ppo_final_model.zip \
  --disable-camera \
  --total-timesteps 2_000_000 \
  --learning-rate 1e-4 \
  --experiment-name basic_walking
```

**What this does**: Robot learns to walk forward (already knows how to balance!)
**Time**: ~30-45 minutes
**Checkpoint**: Loads standing model → faster learning

---

### 3️⃣ Intermediate (Build on Walking)

```bash
python training/train_ppo.py \
  --curriculum-level 1 \
  --checkpoint outputs/basic_walking/ppo_final_model.zip \
  --disable-camera \
  --total-timesteps 1_000_000 \
  --learning-rate 1e-4 \
  --experiment-name intermediate_walking
```

**What this does**: Faster walking + handles small perturbations
**Time**: ~20-30 minutes
**Checkpoint**: Loads walking model → builds on established gait

---

### 4️⃣ Advanced (Build on Intermediate)

```bash
python training/train_ppo.py \
  --curriculum-level 2 \
  --checkpoint outputs/intermediate_walking/ppo_final_model.zip \
  --disable-camera \
  --total-timesteps 1_000_000 \
  --learning-rate 5e-5 \
  --experiment-name advanced_walking
```

**What this does**: Robust walking + recovery from tilts
**Time**: ~20-30 minutes
**Checkpoint**: Loads intermediate model → adds recovery behaviors

---

### 5️⃣ Expert (Build on Advanced)

```bash
python training/train_ppo.py \
  --curriculum-level 3 \
  --checkpoint outputs/advanced_walking/ppo_final_model.zip \
  --disable-camera \
  --total-timesteps 1_000_000 \
  --learning-rate 5e-5 \
  --experiment-name expert_walking
```

**What this does**: Maximum speed + robustness
**Time**: ~20-30 minutes
**Checkpoint**: Loads advanced model → final hardening

---

## Key Points

### ✅ DO:
- **Load checkpoint at each step** (except Step 1)
- Lower learning rate when loading checkpoint (1e-4 or 5e-5)
- Train standing first (fastest path to walking)
- Use `--disable-camera` for faster training

### ❌ DON'T:
- Skip the standing phase (slower convergence)
- Forget to load checkpoint (loses all previous learning!)
- Use same learning rate as initial training (causes forgetting)
- Train with camera initially (much slower)

## Monitoring Progress

Start TensorBoard:
```bash
tensorboard --logdir=outputs/
```

Watch these metrics:
- `rollout/ep_rew_mean`: Should increase steadily
- `rollout/ep_len_mean`: Longer episodes = robot stays upright longer
- `train/approx_kl`: Should stay < 0.02 (stable learning)

## What if I Want to Skip Standing?

You can, but it's slower:

```bash
python training/train_ppo.py \
  --curriculum-level 0 \
  --disable-camera \
  --total-timesteps 2_500_000
```

This takes ~60-90 min (vs 45 min with standing first) and robot falls more initially.

## Adding Visual Observations Later

After you have a good walking policy (level 0-2), add camera:

```bash
python training/train_ppo.py \
  --curriculum-level 2 \
  --checkpoint outputs/advanced_walking/ppo_final_model.zip \
  --use-camera \
  --camera-width 64 \
  --camera-height 64 \
  --total-timesteps 1_500_000 \
  --learning-rate 5e-5 \
  --experiment-name visual_walking
```

This adds visual perception while keeping the walking skills learned without camera.

## Troubleshooting

### "ModuleNotFoundError: No module named 'pidog_env'"
Run from project root: `cd /path/to/pidog_rl && python training/train_ppo.py ...`

### "Checkpoint not found"
Check the path: `ls outputs/standing_only/ppo_final_model.zip`

### Robot keeps falling
- Make sure you loaded the checkpoint from previous level!
- Train longer at current level before advancing
- Check that fall detection is working (torso should touch ground)

### Training too slow
- Use `--disable-camera` (3x faster than with camera)
- Reduce `--n-envs` if CPU is maxed out (try 8 instead of 16)
- Use smaller camera resolution (32x32 instead of 64x64)

## Full Example Session

```bash
# Terminal 1: Start TensorBoard
tensorboard --logdir=outputs/

# Terminal 2: Train standing → walking → expert
cd /path/to/pidog_rl

# Standing (15-30 min)
python training/train_ppo.py --curriculum-level -1 --disable-camera --total-timesteps 1_000_000 --experiment-name standing

# Walking (30-45 min)
python training/train_ppo.py --curriculum-level 0 --checkpoint outputs/standing/ppo_final_model.zip --disable-camera --total-timesteps 2_000_000 --learning-rate 1e-4 --experiment-name walking

# Advanced (20-30 min)
python training/train_ppo.py --curriculum-level 2 --checkpoint outputs/walking/ppo_final_model.zip --disable-camera --total-timesteps 1_000_000 --learning-rate 5e-5 --experiment-name advanced

# View the trained robot
python training/test_trained_model.py outputs/advanced/ppo_final_model.zip
```

## Next Steps

After training:
1. **Test model**: `python training/test_trained_model.py outputs/expert_walking/ppo_final_model.zip`
2. **Export for hardware**: `python scripts/export_for_pi.py outputs/expert_walking/ppo_final_model.zip`
3. **Continue training**: Add `--checkpoint` to any command to keep improving

For more details, see:
- `CURRICULUM_LEARNING.md` - Complete curriculum guide
- `docs/PPO_HYPERPARAMETERS.md` - Hyperparameter explanations
- `TWO_STAGE_TRAINING.md` - Camera vs no-camera training
