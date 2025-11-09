# PiDog Multi-Sensor RL Training Guide

## Overview

This project now supports **full sensor integration** for training reinforcement learning policies using:
- **Camera** (OV5647 5MP RGB) - 84×84×3 images
- **Ultrasonic** (HC-SR04) - Distance sensing up to 4.5m
- **IMU** (MPU6050 6-DOF) - 3-axis accelerometer + 3-axis gyroscope

The training uses **MultiInputPolicy** that combines:
- **CNN** for processing camera images
- **MLP** for processing proprioceptive/sensor data
- **Fusion layer** for combining both modalities

---

## Quick Start

### Test Sensor Integration

```bash
# Run sensor integration tests
docker-compose run --rm pidog_rl python3 tests/test_sensors.py
```

All tests should pass ✓

---

## Training Options

### Option 1: Vector-Only Training (Fastest)

Train without camera using only proprioceptive sensors (31D vector):

```bash
docker-compose run --rm pidog_rl python3 training/train_rl.py \
    --algorithm ppo \
    --total-timesteps 1000000 \
    --n-envs 4 \
    --learning-rate 3e-4 \
    --experiment-name pidog_vector_only
```

**Training time:** ~2-4 hours for 1M steps
**Use case:** Fast iteration, baseline performance

---

### Option 2: Camera + Sensors (Recommended)

Train with full sensor suite using MultiInputPolicy:

```bash
docker-compose run --rm pidog_rl python3 training/train_rl.py \
    --algorithm ppo \
    --total-timesteps 5000000 \
    --n-envs 4 \
    --learning-rate 1e-4 \
    --batch-size 256 \
    --use-camera \
    --camera-width 84 \
    --camera-height 84 \
    --features-dim 256 \
    --experiment-name pidog_vision_full
```

**Training time:** ~1-3 days for 5M steps (with GPU)
**Use case:** Vision-based navigation, obstacle avoidance, deployment

**Key differences from vector-only:**
- Lower learning rate (1e-4 instead of 3e-4)
- Larger batch size (256 instead of 64)
- More training steps (5M instead of 1M)
- GPU strongly recommended

---

### Option 3: Advanced - Nature DQN Architecture

Use deeper CNN for complex visual tasks:

```bash
docker-compose run --rm pidog_rl python3 training/train_rl.py \
    --algorithm ppo \
    --total-timesteps 10000000 \
    --n-envs 4 \
    --learning-rate 1e-4 \
    --batch-size 512 \
    --use-camera \
    --use-nature-cnn \
    --features-dim 512 \
    --experiment-name pidog_nature_cnn
```

**Training time:** ~3-7 days for 10M steps
**Use case:** Complex environments with obstacles, research

---

## Training Parameters Explained

### Camera Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use-camera` | False | Enable camera observations |
| `--camera-width` | 84 | Camera image width |
| `--camera-height` | 84 | Camera image height |
| `--features-dim` | 256 | Feature extractor output dimension |
| `--use-nature-cnn` | False | Use deeper Nature DQN architecture |

### RL Algorithm Settings

| Parameter | Vector-Only | With Camera | Description |
|-----------|-------------|-------------|-------------|
| `--learning-rate` | 3e-4 | 1e-4 | Lower for vision (more stable) |
| `--batch-size` | 64 | 256-512 | Larger for CNN training |
| `--total-timesteps` | 1M | 5-10M | More samples needed for vision |
| `--n-envs` | 4 | 4-8 | Parallel environments |

---

## Observation Space

### Vector-Only Mode (31 dimensions)

```python
obs = np.array([
    # Joint states
    joint_positions[8],      # [0:8]   - Hip/knee angles for 4 legs
    joint_velocities[8],     # [8:16]  - Joint angular velocities

    # IMU orientation
    body_quaternion[4],      # [16:20] - Body orientation (w,x,y,z)

    # IMU motion
    linear_velocity[3],      # [20:23] - Body velocity (vx,vy,vz)
    angular_velocity[3],     # [23:26] - Gyroscope (ωx,ωy,ωz)

    # Height/distance
    body_height[1],          # [26]    - Estimated height
    ultrasonic_distance[1],  # [27]    - Forward distance to obstacle

    # IMU accelerometer
    acceleration[3],         # [28:31] - Accelerometer (ax,ay,az)
])
```

### MultiInput Mode (Dict)

```python
obs = {
    "image": np.uint8[84, 84, 3],    # RGB camera feed [0-255]
    "vector": np.float32[31]         # Same as vector-only above
}
```

---

## Neural Network Architecture

### Standard Combined Extractor (Default)

```
Input: {"image": (84,84,3), "vector": (31,)}
│
├─ CNN Branch (Image)
│  ├─ Conv2D(3→32, k=8, s=4) + ReLU    → (20,20,32)
│  ├─ Conv2D(32→64, k=4, s=2) + ReLU   → (9,9,64)
│  ├─ Conv2D(64→64, k=3, s=1) + ReLU   → (7,7,64)
│  ├─ Flatten                           → (3136,)
│  └─ FC(3136→512) + ReLU              → (512,)
│
├─ MLP Branch (Vector)
│  ├─ FC(31→128) + ReLU                → (128,)
│  └─ FC(128→128) + ReLU               → (128,)
│
└─ Fusion
   └─ Concat(512+128) → FC(640→256) + ReLU → (256,) → Policy/Value heads
```

**Total parameters:** ~1.5M (trainable)

### Nature DQN Extractor (Advanced)

Similar to above but with:
- Deeper CNN: additional conv layers
- Deeper MLP: 256→256→256
- Dropout regularization (p=0.1)
- Total parameters: ~3M

---

## Monitoring Training

### TensorBoard

```bash
# In another terminal
docker-compose run --rm pidog_rl tensorboard --logdir outputs/ --bind_all

# Open browser to http://localhost:6006
```

**Key metrics to watch:**
- `rollout/ep_rew_mean` - Average episode reward (should increase)
- `train/loss` - Policy/value loss (should decrease then stabilize)
- `rollout/ep_len_mean` - Episode length (longer = more stable)

### Console Output

Training will print:

```
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 342         |
|    ep_rew_mean          | 125.3       |
| time/                   |             |
|    fps                  | 1247        |
|    total_timesteps      | 250000      |
-----------------------------------------
```

---

## File Structure

```
pidog_rl/
├── model/
│   ├── pidog.xml              # Main robot model (updated with sensors)
│   ├── pidog_sensors.xml      # NEW: Sensor definitions
│   ├── pidog_actuators.xml    # Servo specifications
│   └── pidog_assets.xml       # Visual meshes
│
├── pidog_env/
│   ├── pidog_env.py           # UPDATED: MultiInput observation space
│   └── feature_extractors.py # NEW: CNN+MLP architectures
│
├── training/
│   ├── train_rl.py            # UPDATED: MultiInputPolicy support
│   ├── evaluate.py
│   └── collect_demos.py
│
├── tests/
│   └── test_sensors.py        # NEW: Sensor integration tests
│
└── outputs/
    └── [experiment_name]/
        ├── logs/              # TensorBoard logs
        └── checkpoints/       # Model checkpoints
```

---

## Expected Training Results

### Phase 1: Vector-Only (1M steps, ~3 hours)

| Metric | Initial | After 1M steps |
|--------|---------|----------------|
| Ep Reward | -10 to -20 | 50 to 150 |
| Ep Length | 50-100 | 300-1000 |
| Forward Velocity | 0.1 m/s | 0.4-0.6 m/s |

### Phase 2: Camera + Sensors (5M steps, ~2 days)

| Metric | Initial | After 5M steps |
|--------|---------|----------------|
| Ep Reward | -15 to -25 | 80 to 200 |
| Ep Length | 50-100 | 400-1000 |
| Forward Velocity | 0.05 m/s | 0.5-0.7 m/s |

**Note:** Vision-based training takes longer to converge but enables obstacle avoidance and navigation.

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution:** Reduce batch size and/or number of parallel environments:

```bash
--batch-size 128 --n-envs 2
```

### Issue: Training unstable (NaN loss)

**Solution:** Lower learning rate:

```bash
--learning-rate 5e-5
```

### Issue: Camera shows all zeros

**Problem:** Camera rendering might be failing

**Solution:**
1. Check MUJOCO_GL environment variable: `export MUJOCO_GL=egl`
2. Verify sensor test passes: `python3 tests/test_sensors.py`

### Issue: Very slow training

**Check:**
- GPU utilization: `watch -n 1 nvidia-smi` (NVIDIA) or `rocm-smi` (AMD)
- CPU usage: Should be near 100% with multi-env
- Consider reducing camera resolution: `--camera-width 64 --camera-height 64`

---

## Sensor Specifications (Real Hardware)

### Camera: OV5647
- Resolution: 2592×1944 still, 1080p30 video
- Training uses: 84×84 RGB (downsampled)
- FOV: ~62° horizontal, ~49° vertical

### Ultrasonic: HC-SR04
- Range: 2cm - 450cm
- Accuracy: ±3mm
- Beam angle: <15°
- Returns -1.0 when no obstacle detected

### IMU: MPU6050
- Accelerometer: ±16g range (training uses)
- Gyroscope: ±1000°/s range
- Update rate: Simulated at ~100Hz

---

## Next Steps After Training

### 1. Evaluate Trained Policy

```bash
docker-compose run --rm pidog_rl python3 training/evaluate.py \
    --model-path outputs/pidog_vision_full/checkpoints/rl_model_1000000_steps.zip \
    --n-eval-episodes 10 \
    --render
```

### 2. Export for Hardware Deployment

```bash
docker-compose run --rm pidog_rl python3 scripts/export_for_pi.py \
    --model-path outputs/pidog_vision_full/ppo_final_model.zip \
    --output-dir pi_deployment/
```

### 3. Real Robot Integration

See `WORKFLOW.md` for hardware deployment instructions.

---

## Advanced Training Strategies

### Curriculum Learning

Start with simple tasks, gradually increase difficulty:

```bash
# Stage 1: Standing stable (100k steps)
python3 training/train_rl.py --experiment-name stage1_stand --total-timesteps 100000

# Stage 2: Walking forward (1M steps)
python3 training/train_rl.py --checkpoint outputs/stage1_stand/ppo_final_model.zip \
    --experiment-name stage2_walk --total-timesteps 1000000

# Stage 3: Obstacle avoidance with camera (5M steps)
python3 training/train_rl.py --checkpoint outputs/stage2_walk/ppo_final_model.zip \
    --use-camera --experiment-name stage3_vision --total-timesteps 5000000
```

### Domain Randomization

Modify `pidog_env.py` to randomize:
- Ground friction
- Servo delays/noise
- Body mass
- Sensor noise

This improves sim-to-real transfer.

---

## Comparing Architectures

Run parallel experiments:

```bash
# Terminal 1: Standard CNN
docker-compose run --rm pidog_rl python3 training/train_rl.py \
    --use-camera --experiment-name cnn_standard

# Terminal 2: Nature DQN
docker-compose run --rm pidog_rl python3 training/train_rl.py \
    --use-camera --use-nature-cnn --experiment-name cnn_nature

# Terminal 3: Vector-only baseline
docker-compose run --rm pidog_rl python3 training/train_rl.py \
    --experiment-name mlp_baseline
```

Compare in TensorBoard:
```bash
tensorboard --logdir outputs/ --bind_all
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pidog_rl_sensor_integration,
  title = {PiDog Multi-Sensor RL Training},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/pidog_rl}
}
```

---

## References

- **Stable-Baselines3:** https://stable-baselines3.readthedocs.io/
- **MuJoCo Documentation:** https://mujoco.readthedocs.io/
- **SunFounder PiDog:** https://docs.sunfounder.com/projects/pidog/
- **Sensor Integration Guide:** See `SENSOR_INTEGRATION_GUIDE.md`

---

**Last Updated:** 2025-11-09
**Status:** ✅ All sensors integrated and tested
