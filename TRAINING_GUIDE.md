# PiDog RL Training Guide

Train the PiDog quadruped robot to move fast forward using Reinforcement Learning.

## Quick Start

### 1. Train the Model (Easiest Method)

Run the quick-start script:

```bash
bash train_fast_forward.sh
```

This will train for 1 million timesteps using PPO with sensible defaults.

### 2. Monitor Training Progress

In a separate terminal, start TensorBoard:

```bash
docker-compose run --rm -p 6006:6006 pidog_rl tensorboard --logdir outputs --host 0.0.0.0
```

Then open your browser to: http://localhost:6006

You'll see:
- Episode rewards over time
- Forward velocity progress
- Success rate
- Policy loss and other metrics

### 3. Test the Trained Model

After training, test your model:

```bash
docker-compose run --rm pidog_rl python training/test_trained_model.py \
    --model-path outputs/pidog_fast_forward/checkpoints/ppo_model_final.zip \
    --algorithm ppo \
    --episodes 5 \
    --deterministic
```

## Advanced Training Options

### Custom Training Configuration

```bash
docker-compose run --rm pidog_rl python training/train_rl.py \
    --algorithm ppo \
    --total-timesteps 2000000 \
    --n-envs 8 \
    --learning-rate 3e-4 \
    --n-steps 4096 \
    --batch-size 128 \
    --experiment-name my_custom_training
```

### Try Different Algorithms

**PPO (Default - Recommended for quadrupeds):**
- Stable and sample-efficient
- Good for continuous control
- Works well with parallel environments

```bash
docker-compose run --rm pidog_rl python training/train_rl.py --algorithm ppo
```

**SAC (Soft Actor-Critic):**
- Off-policy algorithm
- Can be more sample-efficient
- Better exploration

```bash
docker-compose run --rm pidog_rl python training/train_rl.py --algorithm sac
```

**TD3 (Twin Delayed DDPG):**
- Deterministic policy
- Good for precise control
- Lower variance

```bash
docker-compose run --rm pidog_rl python training/train_rl.py --algorithm td3
```

### Resume Training from Checkpoint

```bash
docker-compose run --rm pidog_rl python training/train_rl.py \
    --checkpoint outputs/pidog_fast_forward/checkpoints/ppo_model_100000_steps.zip \
    --total-timesteps 2000000
```

## Environment Details

### Observation Space (27 dimensions)
- Joint positions (8): Hip and knee angles for 4 legs
- Joint velocities (8): Angular velocities
- Body orientation (4): Quaternion
- Linear velocity (3): X, Y, Z velocities
- Angular velocity (3): Roll, pitch, yaw rates
- Body height (1): Z-position

### Action Space (8 dimensions)
- Target joint angles for 8 leg joints
- Normalized to [-1, 1] range
- Scaled to actual servo limits: -90° to 180°

### Reward Function

The robot is rewarded for:
- **Forward velocity** (3x weight): Main objective - move fast!
- **Height maintenance** (1x weight): Stay at ~0.14m height
- **Upright stability** (1.5x weight): Don't flip over
- **Energy efficiency** (small penalty): Don't waste power
- **Straight movement** (penalty for lateral drift): Move forward, not sideways

### Physical Constraints

Based on real SunFounder PiDog servos:
- **Range**: -90° to 180° (extended to support found neutral angles)
- **Max speed**: 7.0 rad/s (400°/s)
- **Max torque**: 0.137 Nm
- **Neutral standing**: Hip=-30°, Knee=-45° (found by systematic search)

## Training Tips

### 1. Start Small
Begin with 1M timesteps to see if the robot learns basic walking. Then scale up.

### 2. Watch the Metrics
- **Episode reward** should increase over time
- **Forward velocity** should trend upward
- **Episode length** should increase (fewer falls)

### 3. Typical Training Time
- **1M timesteps**: ~2-4 hours (depending on hardware)
- **Good walking**: Usually emerges around 200k-500k steps
- **Fast running**: May require 1M-2M steps

### 4. Hardware Acceleration
The training will automatically use GPU if available (ROCm in this container).

Check GPU usage:
```bash
docker-compose run --rm pidog_rl python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 5. Hyperparameter Tuning

If training is unstable:
- Decrease learning rate: `--learning-rate 1e-4`
- Increase batch size: `--batch-size 128`
- Decrease n-steps: `--n-steps 1024`

If training is too slow:
- Increase parallel envs: `--n-envs 8`
- Increase learning rate: `--learning-rate 5e-4`

## Expected Results

After successful training, the robot should:
- ✅ Stand upright without falling
- ✅ Move forward at ~0.3-0.5 m/s (or faster!)
- ✅ Maintain stable gait
- ✅ Self-balance and recover from perturbations

## Troubleshooting

### Robot keeps falling immediately
- Check that neutral angles are correct (-30° hip, -45° knee)
- Increase height maintenance reward weight
- Reduce learning rate

### Robot doesn't move forward
- Check forward velocity reward is positive
- Ensure action space is not saturating
- Try increasing velocity reward weight

### Training diverges (reward goes to -inf)
- Reduce learning rate
- Increase batch size
- Check for NaN values in observations

### No GPU acceleration
- Verify ROCm installation: `rocm-smi`
- Check PyTorch CUDA availability
- Set `--device cuda` explicitly

## File Structure

```
pidog_rl/
├── pidog_env/
│   └── pidog_env.py          # Gymnasium environment
├── training/
│   ├── train_rl.py           # Main training script
│   └── test_trained_model.py # Model evaluation
├── outputs/                   # Training results (created automatically)
│   └── [experiment_name]/
│       ├── logs/             # TensorBoard logs
│       └── checkpoints/      # Model checkpoints
├── model/
│   └── pidog.xml             # MuJoCo model
└── train_fast_forward.sh     # Quick-start script
```

## Next Steps

1. **Train the model**: Run `bash train_fast_forward.sh`
2. **Monitor progress**: Open TensorBoard
3. **Test results**: Use `test_trained_model.py`
4. **Iterate**: Adjust rewards, hyperparameters, try different algorithms
5. **Deploy**: Export best model for real robot (future work)

Good luck training! 🐕🤖
