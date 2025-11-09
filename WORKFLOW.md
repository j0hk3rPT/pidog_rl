# PiDog Neural Network Gait Learning Workflow

Complete workflow from simulation to Raspberry Pi deployment.

## Step 1: Verify Model in Simulation (You are here!)

### Build and Run Docker Container

```bash
# Build Docker image
./scripts/docker_run_gui.sh

# Enter container
docker exec -it pidog_rl_training bash
```

### Test Visualization

```bash
# Inside container: See the 3D model
uv run python tests/test_walk.py
```

This opens a MuJoCo viewer window showing your PiDog model walking with a hardcoded gait.

**What to check:**
- Does the model look correct? (body shape, leg positions)
- Do the joint movements match your real robot?
- Is the walking gait realistic?

### Adjust Model if Needed

If the model doesn't match reality:

```bash
# Edit the model
vim model/pidog.xml

# Edit joint limits, masses, or dimensions
# Then reload the visualization
uv run python tests/test_walk.py
```

## Step 2: Train Neural Network Gait

Once the model looks good, train a neural network to learn walking:

```bash
# Start training with PPO
uv run python training/train_rl.py \
    --algorithm ppo \
    --total-timesteps 1000000 \
    --n-envs 4 \
    --experiment-name pidog_walk_v1

# Monitor training (in another terminal)
tensorboard --logdir=outputs/
# Open http://localhost:6006
```

**Training outputs:**
- Checkpoints: `outputs/pidog_walk_v1/checkpoints/`
- Logs: `outputs/pidog_walk_v1/logs/`
- Final model: `outputs/pidog_walk_v1/ppo_final_model.zip`

**What to monitor:**
- Episode reward (should increase over time)
- Episode length (should reach max steps if stable)
- Forward velocity (should approach target 0.5 m/s)

## Step 3: Evaluate Trained Policy

Test the learned gait:

```bash
# Visualize the trained policy
uv run python training/evaluate.py \
    --model-path outputs/pidog_walk_v1/ppo_final_model.zip \
    --algorithm ppo \
    --n-episodes 10 \
    --render

# Record a video
uv run python training/evaluate.py \
    --model-path outputs/pidog_walk_v1/ppo_final_model.zip \
    --algorithm ppo \
    --n-episodes 5 \
    --record-video outputs/pidog_walk_v1.mp4
```

**What to check:**
- Is the gait smooth?
- Does the robot walk forward consistently?
- Does it maintain balance?

## Step 4: Export for Raspberry Pi

Export the trained neural network for deployment:

```bash
# Export policy
uv run python scripts/export_for_pi.py \
    --model-path outputs/pidog_walk_v1/ppo_final_model.zip \
    --algorithm ppo \
    --output-dir pi_deployment
```

**Output files:**
- `pi_deployment/policy.pth` - Neural network weights
- `pi_deployment/obs_stats.npz` - Observation normalization
- `pi_deployment/pi_inference.py` - Inference script
- `pi_deployment/README.md` - Deployment instructions

## Step 5: Deploy to Raspberry Pi

### Copy Files to Pi

```bash
# From your machine
scp -r pi_deployment/ pi@raspberrypi.local:~/pidog_rl/
```

### Install Dependencies on Pi

```bash
# SSH into Raspberry Pi
ssh pi@raspberrypi.local

# Install PyTorch (CPU version for Pi)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install NumPy
pip3 install numpy
```

### Integrate with PiDog Control

```python
# On Raspberry Pi: your_control_script.py
from pi_deployment.pi_inference import PiDogPolicy
from pidog import Pidog  # Your PiDog library

# Load trained policy
policy = PiDogPolicy("pi_deployment/policy.pth", "pi_deployment/obs_stats.npz")

# Initialize robot
dog = Pidog()

# Control loop
while True:
    # 1. Get observation from sensors
    observation = get_observation(dog)  # Implement this

    # 2. Predict action with neural network
    action = policy.predict(observation)

    # 3. Send to servos
    send_to_servos(dog, action)  # Implement this

    # 4. Small delay
    time.sleep(0.05)  # 20Hz control loop
```

### Observation Collection (You Need to Implement)

```python
def get_observation(dog):
    """
    Collect 27-dimensional observation from PiDog sensors.

    Returns: numpy array shape (27,)
    """
    obs = np.zeros(27)

    # Joint positions (8) - read from servo encoders
    obs[0:8] = [
        dog.read_servo_angle("back_right_hip"),
        dog.read_servo_angle("back_right_knee"),
        dog.read_servo_angle("front_right_hip"),
        dog.read_servo_angle("front_right_knee"),
        dog.read_servo_angle("back_left_hip"),
        dog.read_servo_angle("back_left_knee"),
        dog.read_servo_angle("front_left_hip"),
        dog.read_servo_angle("front_left_knee"),
    ]

    # Joint velocities (8) - calculate from position changes
    obs[8:16] = calculate_velocities()

    # Body orientation (4) - from IMU quaternion
    obs[16:20] = dog.read_imu_quaternion()

    # Body linear velocity (3) - from IMU
    obs[20:23] = dog.read_imu_linear_velocity()

    # Body angular velocity (3) - from IMU gyro
    obs[23:26] = dog.read_imu_angular_velocity()

    # Body height (1) - estimate or use distance sensor
    obs[26] = estimate_height()

    return obs
```

### Action Application (You Need to Implement)

```python
def send_to_servos(dog, action):
    """
    Send normalized action [-1, 1] to servos.

    Args:
        action: numpy array shape (8,) with values in [-1, 1]
    """
    # Scale from [-1, 1] to [0, 180] degrees
    angles = (action + 1.0) * 90.0  # Maps to [0, 180]

    # Send to each servo
    dog.set_servo_angle("back_right_hip", angles[0])
    dog.set_servo_angle("back_right_knee", angles[1])
    dog.set_servo_angle("front_right_hip", angles[2])
    dog.set_servo_angle("front_right_knee", angles[3])
    dog.set_servo_angle("back_left_hip", angles[4])
    dog.set_servo_angle("back_left_knee", angles[5])
    dog.set_servo_angle("front_left_hip", angles[6])
    dog.set_servo_angle("front_left_knee", angles[7])
```

## Troubleshooting

### Simulation doesn't match reality

- Adjust joint limits in `model/pidog.xml`
- Adjust masses and inertias
- Adjust friction coefficients
- Retrain after changes

### Policy doesn't work on real robot

- Check observation values are in same range as training
- Verify servo mapping is correct
- Start with lower servo speeds for safety
- Add observation normalization if needed

### Robot is unstable

- Increase training timesteps
- Adjust reward function to prioritize stability
- Reduce action frequency (control loop rate)
- Add observation filtering (low-pass filter)

## Next Steps

Once basic walking works:

1. **Collect real robot data** - Record actual sensor data while manually controlling
2. **Fine-tune in sim** - Use real data to improve simulation accuracy
3. **Advanced gaits** - Train for different speeds, turning, etc.
4. **Terrain adaptation** - Train on varied surfaces
5. **Behavior cloning** - Learn from demonstrations if available

## References

- MuJoCo: https://mujoco.org/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- PiDog: https://github.com/sunfounder/pidog
