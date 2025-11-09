# PiDog Visualization Guide

This guide explains how to visualize your PiDog training in real-time.

## Overview

The visualization system has been improved to provide:

1. **Real-time rendering** at 30 FPS (not fast-forward)
2. **Fixed GLX errors** - proper window lifecycle management
3. **Training checkpoints** - see the robot learning in real-time during training
4. **Smooth animations** - no more flickering windows

## Quick Start

### 1. Visualize Existing Training

To visualize a trained model or show the environment with random actions:

```bash
docker-compose run --rm pidog_rl python visualize_training.py
```

This will:
- Look for checkpoints in `outputs/pidog_production/`
- If found, visualize the trained policy
- If not found, show the environment with random actions
- Render at 30 FPS in real-time (not fast-forward)

### 2. Train with Real-Time Visualization

To train the robot and see its progress at regular intervals:

```bash
# Train for 500K steps, visualize every 50K steps
docker-compose run --rm pidog_rl python train_with_visualization.py \
    --total-timesteps 500000 \
    --visualize-freq 50000 \
    --n-eval-episodes 2
```

This will:
- Train the robot using PPO
- Every 50,000 timesteps, pause training and show you 2 episodes in real-time
- Save checkpoints every 10,000 steps
- Continue training after each visualization

**During visualization:**
- **Left mouse**: Rotate view
- **Right mouse**: Zoom
- **Middle mouse**: Pan
- **Space**: Pause/Resume
- **Esc**: Skip to next checkpoint and continue training

## Options

### train_with_visualization.py

```bash
python train_with_visualization.py [OPTIONS]

Options:
  --total-timesteps N       Total training timesteps (default: 500000)
  --visualize-freq N        Visualize every N timesteps (default: 50000)
  --n-eval-episodes N       Episodes to show per visualization (default: 2)
  --render-fps N            Frames per second (default: 30)
  --n-envs N                Parallel environments (default: 4)
  --learning-rate F         Learning rate (default: 3e-4)
  --save-freq N             Save checkpoint every N steps (default: 10000)
  --use-camera              Use camera observations (slower)
  --output-dir PATH         Output directory (default: outputs)
  --experiment-name NAME    Experiment name (default: auto-generated)
  --seed N                  Random seed (default: 42)
```

### Examples

**Quick test with frequent visualizations:**
```bash
docker-compose run --rm pidog_rl python train_with_visualization.py \
    --total-timesteps 100000 \
    --visualize-freq 25000 \
    --n-eval-episodes 1
```

**Long training with periodic checkpoints:**
```bash
docker-compose run --rm pidog_rl python train_with_visualization.py \
    --total-timesteps 2000000 \
    --visualize-freq 200000 \
    --n-eval-episodes 3 \
    --save-freq 50000
```

**Training with camera observations:**
```bash
docker-compose run --rm pidog_rl python train_with_visualization.py \
    --total-timesteps 1000000 \
    --visualize-freq 100000 \
    --use-camera
```

## How It Works

### Real-Time Rendering

The visualization system now controls the frame rate to match real-time:

- **Frame time**: 1/30 seconds (30 FPS)
- **Smooth animations**: Each step waits the appropriate time
- **No fast-forward**: You see the robot move at actual speed

### Training Checkpoints

The `VisualizeCallback` class allows you to see training progress:

1. Training runs normally with parallel environments
2. At specified intervals (e.g., every 50K steps):
   - Training pauses
   - A viewer window opens
   - The current policy runs for N episodes
   - You see the robot's behavior in real-time
3. After visualization:
   - Window closes automatically
   - Training resumes

### Fixed GLX Error

The previous error:
```
X Error of failed request:  GLXBadWindow
  Major opcode of failed request:  149 (GLX)
```

Was caused by improper viewer lifecycle management during resets. This is now fixed by:
- Keeping the viewer open across episode resets
- Only resetting the environment state, not the viewer
- Adding proper cleanup on viewer close

## Using Visualization Callbacks in Your Code

You can add real-time visualization to any training script:

```python
from training.visualization_callback import VisualizeCallback

# Create callback
visualize_callback = VisualizeCallback(
    visualize_freq=50000,      # Show every 50K steps
    n_eval_episodes=2,         # Show 2 episodes
    render_fps=30,             # At 30 FPS
    max_steps_per_episode=500, # Max 500 steps per episode
    verbose=1
)

# Train with callback
model.learn(
    total_timesteps=1000000,
    callback=visualize_callback
)
```

## Troubleshooting

### "GLXBadWindow" error still occurs

Make sure you're using the updated `visualize_training.py` script. The fix includes:
- Proper time.sleep() calls
- Brief pauses before resets
- Checking viewer.is_running() before operations

### Visualization is still too fast

Check that the script is using the updated version with `render_fps` parameter:
```python
# Should see this in the output
print(f"\nRendering at {render_fps} FPS (real-time)")
```

### Cannot see the viewer window

Make sure X11 forwarding is set up correctly:
```bash
# On host, allow docker containers to connect
xhost +local:docker

# Verify DISPLAY is set
echo $DISPLAY
```

### Training is too slow with visualization

Reduce visualization frequency:
```bash
# Only visualize every 100K steps instead of 50K
python train_with_visualization.py --visualize-freq 100000
```

Or reduce episodes per visualization:
```bash
# Show only 1 episode instead of 2
python train_with_visualization.py --n-eval-episodes 1
```

## Technical Details

### Frame Rate Control

```python
frame_time = 1.0 / render_fps  # e.g., 1/30 = 0.0333 seconds

while running:
    step_start = time.time()

    # Do simulation step
    env.step(action)
    viewer.sync()

    # Wait to maintain frame rate
    elapsed = time.time() - step_start
    if elapsed < frame_time:
        time.sleep(frame_time - elapsed)
```

### Episode Reset Timing

```python
if terminated or truncated:
    print(f"Episode ended at step {step_count}")
    time.sleep(0.5)  # Brief pause so you can see final state
    obs, _ = env.reset()  # Reset environment, keep viewer
```

This gives you time to see the final state before the robot resets to starting position.

## See Also

- `visualize_training.py` - Main visualization script
- `train_with_visualization.py` - Training with checkpoints
- `training/visualization_callback.py` - Callback implementation
- `training/train_rl.py` - Standard training without visualization
