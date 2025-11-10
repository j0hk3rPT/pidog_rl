# PiDog RL Quick Start Guide

This guide will help you visualize the robot walking and train an imitation learning model.

## Setup

### 1. Allow X11 Access (for GUI window)

First, allow Docker to access your X11 display:

```bash
xhost +local:docker
```

### 2. Start the Docker container

```bash
docker-compose run --rm pidog_rl
```

Once inside the container, you'll be in `/workspace/pidog_rl`.

### 3. Test the Setup (Optional but Recommended)

```bash
python3 scripts/test_setup.py
```

This will verify that all dependencies are installed and the environment works correctly.

## Visualizing the Robot Walking

The visualization will open an **interactive MuJoCo window** showing the robot in 3D. You can rotate the camera, zoom, and watch the robot walk in real-time!

### Option 1: Visualize Sunfounder Gaits (Recommended)

Show the robot walking using the extracted Sunfounder demonstrations:

```bash
# Trot forward (default) - fast, 6 steps per cycle
# Use --step-delay to slow down for better viewing (default is 0.01s)
python3 scripts/visualize_walk.py --mode sunfounder --gait trot_forward --n-cycles 10 --step-delay 0.05

# Faster viewing (less delay)
python3 scripts/visualize_walk.py --mode sunfounder --gait trot_forward --n-cycles 10 --step-delay 0.01

# Other available gaits:
python3 scripts/visualize_walk.py --mode sunfounder --gait walk_forward --n-cycles 5 --step-delay 0.05
python3 scripts/visualize_walk.py --mode sunfounder --gait trot_backward --n-cycles 5 --step-delay 0.05
python3 scripts/visualize_walk.py --mode sunfounder --gait trot_left --n-cycles 5 --step-delay 0.05
```

**Interactive Viewer Controls:**
- **Left mouse**: Rotate camera
- **Right mouse**: Move camera
- **Scroll wheel**: Zoom in/out
- **Double-click**: Select/track object
- **Ctrl+R**: Reset camera view

You'll also see progress in the terminal:
```
Cycle 1/10: reward=2.45, avg_reward=0.408, height=0.142m, vel=0.012m/s
Cycle 2/10: reward=3.21, avg_reward=0.472, height=0.138m, vel=0.024m/s
...
```

### Option 2: Visualize Hardcoded Trot

Use a simple hardcoded sinusoidal gait:

```bash
python3 scripts/visualize_walk.py --mode hardcoded --n-cycles 10
```

### Option 3: Visualize Trained Policy

After training (see below), visualize the trained model:

```bash
python3 scripts/visualize_walk.py --mode policy --policy-path outputs/[experiment_name]/rl_policy.zip --n-cycles 10
```

## Training an Imitation Learning Model

### Quick Start: Behavioral Cloning (BC)

Train using behavioral cloning on the trot_forward gait:

```bash
python3 scripts/train_from_sunfounder.py \
    --method bc \
    --gait trot_forward \
    --bc-epochs 100
```

### Train on All Gaits

```bash
python3 scripts/train_from_sunfounder.py \
    --method bc \
    --gait all \
    --bc-epochs 200
```

### Reinforcement Learning (RL) Only

Train from scratch using PPO:

```bash
python3 scripts/train_from_sunfounder.py \
    --method rl \
    --rl-timesteps 500000
```

### Combined Approach: BC + RL (Recommended)

Use behavioral cloning as initialization, then fine-tune with RL:

```bash
python3 scripts/train_from_sunfounder.py \
    --method both \
    --gait trot_forward \
    --bc-epochs 100 \
    --rl-timesteps 500000
```

This approach:
1. Pre-trains using BC on Sunfounder demonstrations (fast, gets a good baseline)
2. Fine-tunes with RL to optimize for the reward function (improves performance)

## Monitoring Training

### TensorBoard

Start TensorBoard to monitor training progress:

```bash
tensorboard --logdir outputs --bind_all
```

Then open your browser to `http://localhost:6006`

## File Structure

- `datasets/sunfounder_demos.pkl` - Extracted demonstration actions from Sunfounder
- `scripts/extract_sunfounder_demos.py` - Script that extracts the demonstrations
- `scripts/visualize_walk.py` - Visualization script
- `scripts/train_from_sunfounder.py` - Training script
- `outputs/` - Training outputs (models, logs, checkpoints)

## Available Gaits

The extracted Sunfounder demonstrations include:

- `trot_forward` - Fast trotting gait moving forward (6 steps per cycle)
- `trot_backward` - Trotting backward (6 steps per cycle)
- `trot_left` - Trotting while turning left (6 steps per cycle)
- `trot_right` - Trotting while turning right (6 steps per cycle)
- `walk_forward` - Slower walking gait forward (49 steps per cycle)
- `walk_backward` - Walking backward (49 steps per cycle)
- `walk_left` - Walking while turning left (49 steps per cycle)
- `walk_right` - Walking while turning right (49 steps per cycle)

## Action Space

The MuJoCo environment uses:
- **Observation**: Joint positions, velocities, IMU data, body orientation, etc. (31 dimensions)
- **Action**: 8 continuous joint angles normalized to [-1, 1]
  - Maps to servo range: -π/2 to π radians (-90° to 180°)
- **Legs**: [back_right, front_right, back_left, front_left]
- **Joints per leg**: [hip/shoulder, knee] (2 DOF per leg)

## Tips

1. **Start with BC on a single gait** (e.g., `trot_forward`) to quickly get a working baseline
2. **Use `--method both`** for best results - BC provides good initialization, RL optimizes performance
3. **Monitor with TensorBoard** to track training progress and rewards
4. **Adjust training time** based on performance:
   - BC: 50-200 epochs typically sufficient
   - RL: 500k-1M timesteps for good performance
5. **Visualize frequently** to see how the policy is improving

## Troubleshooting

### Display Issues

If you get display/rendering errors, try setting:

```bash
export MUJOCO_GL=osmesa  # For headless rendering
# or
export MUJOCO_GL=glfw    # For windowed rendering
```

### GPU Issues (ROCm)

Check GPU availability:

```bash
rocm-smi
```

Verify PyTorch sees the GPU:

```python
python3 -c "import torch; print(torch.cuda.is_available())"
```

## Next Steps

After training a model:
1. Visualize it using `visualize_walk.py` with `--mode policy`
2. Export it for deployment on the real robot using `scripts/export_for_pi.py`
3. Fine-tune using additional RL training
4. Try combining multiple gaits for more robust behavior
