# Two-Stage Training Guide

This guide explains how to efficiently train the PiDog robot using a two-stage approach: fast proprioceptive learning followed by visual fine-tuning.

## Why Two-Stage Training?

Training with camera observations is **significantly slower** due to:
- MuJoCo camera rendering overhead
- Larger neural network (CNN + MLP vs MLP only)
- More compute per environment step

**Two-stage training** lets you:
1. **Stage 1**: Learn basic locomotion quickly using proprioceptive sensors (30-60 min)
2. **Stage 2**: Add visual perception and fine-tune (20-40 min)

Total time: ~50-100 min vs 2-3 hours training with camera from scratch.

## Architecture Compatibility

The environment **always uses Dict observation space** (`MultiInputPolicy`), making checkpoints transferable:

- **Camera disabled**: Image observations are zeros (ignored by policy)
- **Camera enabled**: Image observations are actual camera feed

Same policy architecture → **seamless checkpoint transfer** ✓

## Stage 1: Fast Proprioceptive Learning

Train without camera rendering for maximum speed:

```bash
docker-compose run --rm pidog_rl python training/train_rl.py \
  --disable-camera \
  --total-timesteps=2_000_000 \
  --n-envs 16 \
  --batch-size 128 \
  --n-steps 2048 \
  --learning-rate 3e-4 \
  --features-dim 128 \
  --experiment-name stage1_proprioceptive
```

**Stage 1 settings explained:**
- `--disable-camera`: No camera rendering (fast!)
- `--n-envs 16`: Use all CPU cores (no rendering overhead)
- `--batch-size 128`: Moderate batch size for CPU
- `--features-dim 128`: Lighter network for faster training

**Expected results:**
- Training time: ~30-60 minutes on 16-core CPU
- Policy learns: walking, balance, obstacle avoidance (from ultrasonic sensor)
- Model saved: `outputs/stage1_proprioceptive/ppo_final_model.zip`

## Stage 2: Visual Fine-Tuning

Load Stage 1 checkpoint and enable camera:

```bash
docker-compose run --rm pidog_rl python training/train_rl.py \
  --use-camera \
  --camera-width 64 \
  --camera-height 64 \
  --checkpoint outputs/stage1_proprioceptive/ppo_final_model.zip \
  --total-timesteps=1_000_000 \
  --n-envs 8 \
  --batch-size 256 \
  --learning-rate 1e-4 \
  --features-dim 128 \
  --experiment-name stage2_visual
```

**Stage 2 settings explained:**
- `--use-camera`: Enable actual camera rendering
- `--checkpoint`: Load Stage 1 trained model
- `--learning-rate 1e-4`: **Lower LR** to prevent catastrophic forgetting
- `--n-envs 8`: Fewer envs due to camera rendering overhead
- Camera resolution `64x64`: Faster than default `84x84`
- Fewer timesteps: Fine-tuning, not training from scratch

**Expected results:**
- Training time: ~20-40 minutes on 8-core CPU
- Policy learns: visual perception, visual obstacle detection
- Retains: locomotion skills from Stage 1
- Model saved: `outputs/stage2_visual/ppo_final_model.zip`

## Alternative: Direct Training with Camera

If you prefer single-stage training, use optimized camera settings:

```bash
docker-compose run --rm pidog_rl python training/train_rl.py \
  --use-camera \
  --camera-width 64 \
  --camera-height 64 \
  --features-dim 128 \
  --total-timesteps=3_000_000 \
  --n-envs 8 \
  --batch-size 256 \
  --learning-rate 3e-4 \
  --experiment-name direct_visual
```

**Trade-offs:**
- ✓ Simpler workflow (one command)
- ✗ Slower convergence (~2-3 hours)
- ✗ No ability to debug proprioception separately

## Key Parameters Reference

### Speed Optimizations
- `--disable-camera`: Fastest training (no rendering)
- `--camera-width/height 64`: Smaller images = faster
- `--features-dim 128`: Lighter network (vs 256)
- `--n-envs 16`: More parallelism (when no camera)

### Quality Optimizations
- `--use-nature-cnn`: Deeper CNN for complex vision tasks
- `--features-dim 256`: More capacity (slower)
- `--camera-width/height 84`: Higher resolution
- More timesteps: Better convergence

### Transfer Learning
- `--checkpoint <path>`: Load pretrained model
- `--learning-rate 1e-4`: Lower LR for fine-tuning (vs 3e-4 for training)
- Fewer timesteps: Fine-tuning requires less training

## Monitoring Training

Start TensorBoard to monitor progress:

```bash
tensorboard --logdir=outputs/
```

Navigate to http://localhost:6006 to view:
- `rollout/ep_rew_mean`: Average episode reward
- `train/value_loss`: Value function learning
- `rollout/ep_len_mean`: Episode length (longer = robot survives longer)

## Troubleshooting

### Checkpoint loading fails
- Ensure you're using `--checkpoint` with the `.zip` file
- Check that the experiment directory exists

### Training too slow
- Reduce `--n-envs` (less parallelism but more stable)
- Use `--disable-camera` for Stage 1
- Reduce camera resolution to 48x48 or 32x32

### Policy forgets Stage 1 skills
- Lower learning rate in Stage 2 (try `1e-5`)
- Reduce Stage 2 timesteps
- Check that Stage 1 model actually learned to walk

### Out of memory
- Reduce `--batch-size` (128 → 64)
- Reduce `--n-envs` (16 → 8)
- Reduce `--features-dim` (256 → 128)

## Next Steps

After training, you can:

1. **Evaluate the model**: See `scripts/visualize_walk.py`
2. **Deploy to hardware**: Export to real PiDog robot
3. **Further fine-tuning**: Add BC pretraining with `--pretrain-bc`

For more details, see `QUICKSTART.md` and `CLAUDE.md`.
