# PiDog Sensor Integration Guide for MuJoCo RL Training

## Overview
This document provides specifications for PiDog sensors and implementation guidelines for integrating them into MuJoCo simulation for RL training.

---

## PiDog Real Hardware Sensor Specifications

### 1. Camera Module - OV5647

**Manufacturer:** SunFounder (OV5647 sensor)
**Connection:** Raspberry Pi CSI (Camera Serial Interface)

| Specification | Value |
|--------------|-------|
| **Sensor** | OV5647 5MP |
| **Resolution (Still)** | 2592 × 1944 pixels (5MP) |
| **Resolution (Video)** | 1080p @ 30fps<br>720p @ 60fps<br>640×480 @ 90fps |
| **Field of View (FOV)** | ~62° horizontal, ~49° vertical (estimated from OV5647 specs) |
| **Focal Length** | ~3.04mm (fixed focus) |
| **Focal Ratio** | F2.0 |
| **Interface** | CSI (MIPI) |
| **Features** | Color recognition, face detection, object tracking via OpenCV |

**Mounting Location:** Front of head (nose area)

**Recommended Simulation Parameters:**
- Training resolution: 84×84 or 64×64 RGB (downsampled for efficiency)
- Frame skip: 4-8 frames (effective 15-30 Hz for RL)
- Color space: RGB or Grayscale (single channel)

---

### 2. Ultrasonic Distance Sensor - HC-SR04

**Model:** HC-SR04
**Connection:** GPIO pins (4-pin: VCC, Trig, Echo, GND)

| Specification | Value |
|--------------|-------|
| **Range (Min)** | 2 cm |
| **Range (Max)** | 400-450 cm (practical) |
| **Resolution** | 0.3 cm (3mm accuracy) |
| **Operating Voltage** | 5V DC |
| **Operating Current** | ~16 mA |
| **Frequency** | 40 kHz ultrasonic pulses |
| **Beam Angle** | <15° (narrow cone) |
| **Trigger Pulse** | 10μs TTL pulse |
| **Dimensions** | 46 × 20.5 × 15 mm |

**Mounting Location:** Front of head (above nose)

**Distance Calculation Formula:**
```
distance_cm = echo_pulse_time_μs / 58
```

**Recommended Simulation Parameters:**
- Sensor type: MuJoCo `rangefinder`
- Max range: 4.5m (matches hardware)
- Update rate: ~40-60 Hz
- Add Gaussian noise: σ = 0.003m (3mm)

---

### 3. IMU - MPU6050 (6-DOF)

**Model:** MPU6050
**Connection:** I2C bus (address 0x68)

#### Accelerometer Specifications

| Specification | Value |
|--------------|-------|
| **Axes** | 3-axis (X, Y, Z) |
| **Range Options** | ±2g, ±4g, ±8g, ±16g (programmable) |
| **Default Range** | ±2g |
| **Sensitivity** | 16384 LSB/g (at ±2g range) |
| **Resolution** | 16-bit (signed) |
| **Noise** | ~0.01 m/s² typical |

#### Gyroscope Specifications

| Specification | Value |
|--------------|-------|
| **Axes** | 3-axis (Roll, Pitch, Yaw rates) |
| **Range Options** | ±250°/s, ±500°/s, ±1000°/s, ±2000°/s |
| **Default Range** | ±250°/s |
| **Sensitivity** | 131 LSB/°/s (at ±250°/s range) |
| **Resolution** | 16-bit (signed) |
| **Noise** | ~0.005 rad/s typical |

#### Additional Features
- **Temperature Sensor:** On-chip (not used for RL)
- **DMP (Digital Motion Processor):** Hardware sensor fusion (optional)
- **Update Rate:** Up to 1000 Hz (typically 100-200 Hz in practice)

**Mounting Location:** Center of body (electronics compartment)

**Recommended Simulation Parameters:**
- Use MuJoCo `<accelerometer>` and `<gyro>` sensors
- Accel range: ±16g (allow for impacts)
- Gyro range: ±1000°/s
- Add sensor noise: accel σ=0.01 m/s², gyro σ=0.005 rad/s

---

## MuJoCo Sensor Implementation

### XML Sensor Configuration

#### 1. Camera Sensor

MuJoCo cameras are defined in the worldbody/body and rendered programmatically:

```xml
<!-- In your pidog.xml under <body name="head"> -->
<camera name="pidog_camera"
        pos="0.02 0.04 0.01"
        euler="0 0 0"
        fovy="62"
        mode="fixed"/>
```

**Rendering in Python:**
```python
import mujoco

# Create renderer
renderer = mujoco.Renderer(model, height=84, width=84)

# Render from camera
renderer.update_scene(data, camera="pidog_camera")
rgb_array = renderer.render()  # Returns (84, 84, 3) uint8 array

# Optional: Depth rendering
renderer.enable_depth_rendering()
depth_array = renderer.render()  # Returns depth in meters
```

**Key Parameters:**
- `fovy`: Field of view (vertical) in degrees - use 49° to match OV5647
- `pos`: Position relative to parent body (adjust to nose location)
- `euler`: Orientation (0,0,0 = forward-facing)

---

#### 2. Rangefinder (Ultrasonic) Sensor

```xml
<!-- Add site attachment point in <body name="head"> -->
<site name="ultrasonic_site"
      pos="0.03 0.05 0.015"
      euler="0 0 0"
      size="0.005"/>

<!-- In <sensor> section -->
<sensor>
  <rangefinder name="ultrasonic"
               site="ultrasonic_site"
               cutoff="4.5"
               noise="0.003"/>
</sensor>
```

**Key Parameters:**
- `cutoff`: Maximum detection range (4.5m = 450cm)
- `noise`: Gaussian noise std dev (0.003m = 3mm)
- `site`: Attachment point on head

**Reading in Python:**
```python
# Sensor data is in mjData.sensordata
ultrasonic_distance = data.sensordata[sensor_id]  # Returns distance in meters
```

**Important:** MuJoCo rangefinder measures along the site's Z-axis. Ensure the site orientation points forward.

---

#### 3. IMU Sensors (Accelerometer + Gyroscope)

```xml
<!-- Add IMU site in <body name="base"> (main body) -->
<site name="imu_site"
      pos="0.03 0.06 0.0"
      size="0.005"/>

<!-- In <sensor> section -->
<sensor>
  <!-- 3-axis accelerometer -->
  <accelerometer name="imu_accel"
                 site="imu_site"
                 noise="0.01"/>

  <!-- 3-axis gyroscope -->
  <gyro name="imu_gyro"
        site="imu_site"
        noise="0.005"/>

  <!-- Optional: magnetometer (not used in current PiDog) -->
  <magnetometer name="imu_mag"
                site="imu_site"
                noise="0.01"/>
</sensor>
```

**Reading in Python:**
```python
# Accelerometer returns 3 values: [ax, ay, az] in m/s²
accel_data = data.sensordata[accel_sensor_id:accel_sensor_id+3]

# Gyroscope returns 3 values: [ωx, ωy, ωz] in rad/s
gyro_data = data.sensordata[gyro_sensor_id:gyro_sensor_id+3]
```

**Units:**
- Accelerometer: m/s² (includes gravity: ~9.81 m/s² on Z when upright)
- Gyroscope: rad/s (angular velocity around each axis)
- All in MuJoCo world frame (not sensor-local frame)

---

## Observation Space Design for RL

### Option 1: Vector-Only (Current + Sensors)

Extend current 27D observation with sensor readings (no camera images):

```python
observation_space = gym.spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(31,),  # 27 + 1 ultrasonic + 3 accel (gyro already in obs)
    dtype=np.float32
)

# Observation indices:
# [0:8]    - Joint positions
# [8:16]   - Joint velocities
# [16:20]  - Body quaternion (replace with IMU in real deployment)
# [20:23]  - Linear velocity (replace with IMU in real deployment)
# [23:26]  - Angular velocity (replace with IMU gyro in real deployment)
# [26]     - Body height (replace with ultrasonic/estimate)
# [27]     - Ultrasonic distance (NEW)
# [28:31]  - IMU accelerometer (NEW - add to verify gravity vector)
```

**Pros:** Simple, fast training, small model
**Cons:** No vision, relies on position sensing
**Use Case:** Initial training, sim-to-real transfer with state estimation

---

### Option 2: MultiInput (Vision + Vector)

Combine camera images with state vectors using Dict space:

```python
observation_space = gym.spaces.Dict({
    "image": gym.spaces.Box(
        low=0,
        high=255,
        shape=(84, 84, 3),  # or (1, 84, 84) for grayscale
        dtype=np.uint8
    ),
    "vector": gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(31,),  # All proprioceptive + sensor data
        dtype=np.float32
    )
})
```

**Policy:** Use `MultiInputPolicy` from Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=256)

        # CNN for images
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # MLP for vector observations
        self.mlp = nn.Sequential(
            nn.Linear(31, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Combined features
        self.combined = nn.Sequential(
            nn.Linear(self._get_cnn_output_dim() + 64, 256),
            nn.ReLU(),
        )

    def forward(self, observations):
        image_features = self.cnn(observations["image"].float() / 255.0)
        vector_features = self.mlp(observations["vector"])
        combined = torch.cat([image_features, vector_features], dim=1)
        return self.combined(combined)

# Training with MultiInputPolicy
model = PPO(
    "MultiInputPolicy",
    env,
    policy_kwargs={"features_extractor_class": CustomCombinedExtractor},
    verbose=1
)
```

**Pros:** Vision + proprioception, closer to real deployment
**Cons:** Slower training, larger model, requires more samples
**Use Case:** Vision-based navigation, obstacle avoidance

---

### Option 3: Vision-Only (Pure CNN)

Use only camera + minimal sensors (like real vision-based robotics):

```python
observation_space = gym.spaces.Box(
    low=0,
    high=255,
    shape=(84, 84, 3),
    dtype=np.uint8
)

# Policy: CnnPolicy with frame stacking
from stable_baselines3.common.vec_env import VecFrameStack

env = VecFrameStack(env, n_stack=4)  # Stack 4 frames for motion info
model = PPO("CnnPolicy", env)
```

**Pros:** Fully deployable without state estimation, end-to-end learning
**Cons:** Very sample inefficient, requires enormous training time
**Use Case:** Research, when proprioceptive sensors unavailable

---

## Modified pidog_env.py Structure

### Example: Vector-Only with Real Sensors

```python
def _get_obs(self):
    """Get observation with real sensor integration."""

    # Joint states (from encoders in real robot)
    joint_pos = self.data.qpos[7:15].copy()
    joint_vel = self.data.qvel[6:14].copy()

    # IMU data (replace simulation values)
    if self.use_real_sensors:
        # Read from MPU6050 via I2C
        accel = self.read_mpu6050_accel()  # [ax, ay, az] in m/s²
        gyro = self.read_mpu6050_gyro()    # [ωx, ωy, ωz] in rad/s
        quat = self.estimate_quaternion_from_imu()  # Sensor fusion
    else:
        # Simulation (current implementation)
        quat = self.data.qpos[3:7].copy()
        accel = self.data.sensordata[accel_sensor_id:accel_sensor_id+3]
        gyro = self.data.sensordata[gyro_sensor_id:gyro_sensor_id+3]

    # Ultrasonic distance sensor
    if self.use_real_sensors:
        ultrasonic_dist = self.read_hcsr04_distance()  # in meters
    else:
        ultrasonic_dist = self.data.sensordata[range_sensor_id]

    # Height estimation (no direct sensor on real robot)
    if self.use_real_sensors:
        # Estimate from kinematics or ultrasonic pointing down
        body_height = self.estimate_height_from_legs()
    else:
        body_height = self.data.qpos[2]

    # Velocity estimation (integrate IMU or use visual odometry)
    if self.use_real_sensors:
        linear_vel = self.integrate_accel_to_velocity(accel)
    else:
        linear_vel = self.data.qvel[0:3].copy()

    obs = np.concatenate([
        joint_pos,        # [0:8]
        joint_vel,        # [8:16]
        quat,             # [16:20]
        linear_vel,       # [20:23]
        gyro,             # [23:26]
        [body_height],    # [26]
        [ultrasonic_dist],# [27]
        accel,            # [28:31] - NEW
    ])

    return obs
```

---

## Sensor Noise Models for Sim-to-Real Transfer

### Recommendation: Add Realistic Noise in Simulation

```xml
<!-- In pidog_actuators.xml or main XML -->
<sensor>
  <!-- Ultrasonic: ±3mm accuracy -->
  <rangefinder name="ultrasonic" site="ultrasonic_site"
               cutoff="4.5" noise="0.003"/>

  <!-- Accelerometer: typical MPU6050 noise -->
  <accelerometer name="imu_accel" site="imu_site"
                 noise="0.01"/>

  <!-- Gyroscope: typical MPU6050 noise -->
  <gyro name="imu_gyro" site="imu_site"
        noise="0.005"/>
</sensor>
```

### Additional Noise Sources to Consider

1. **IMU Bias Drift:** Add slowly varying bias to accel/gyro
2. **Latency:** Delay sensor readings by 10-20ms
3. **Quantization:** Round sensor values to match ADC resolution
4. **Dropout:** Randomly drop sensor readings (1-5% rate)
5. **Calibration Error:** Add constant offset to sensor readings

---

## Training Strategy Recommendations

### Phase 1: Simulation with Proprioception (Current)
- Train with perfect state information (27D obs)
- Achieve stable walking in simulation
- **Duration:** 1-2M steps (~2-4 hours)

### Phase 2: Add Sensor Noise
- Replace perfect state with noisy IMU + ultrasonic
- Add domain randomization (friction, mass, actuator delays)
- **Duration:** 2-5M steps (~4-10 hours)

### Phase 3: Add Vision (Optional)
- Integrate camera rendering to observation
- Switch to MultiInputPolicy
- Train with vision + proprioception
- **Duration:** 10-50M steps (~1-3 days with GPU)

### Phase 4: Real Hardware Deployment
- Export trained policy
- Integrate with ROS/direct servo control
- Calibrate sensors and servos
- Fine-tune with real robot data (DAGGER/online learning)

---

## File Modifications Checklist

### To Add Sensors to MuJoCo Simulation:

- [ ] **model/pidog.xml** (or new pidog_sensors.xml)
  - Add `<site>` tags for sensor mounting points
  - Add `<camera>` in head body
  - Include sensor XML file

- [ ] **model/pidog_sensors.xml** (new file)
  - Define all sensors: rangefinder, accelerometer, gyro
  - Set noise parameters

- [ ] **pidog_env/pidog_env.py**
  - Modify `_get_obs()` to read sensor data
  - Add camera rendering with mujoco.Renderer
  - Update observation space (Box or Dict)
  - Add sensor data processing/filtering

- [ ] **training/train_rl.py**
  - Switch policy: MlpPolicy → MultiInputPolicy or CnnPolicy
  - Add policy_kwargs for custom feature extractor
  - Increase training steps (vision needs more samples)

- [ ] **configs/ppo_default.yaml**
  - Add sensor-specific hyperparameters
  - Adjust learning rate (lower for vision: 1e-4)
  - Increase batch size for CNN (256-512)

---

## Sensor Data Access in MuJoCo

### Finding Sensor IDs

```python
# Print all sensors
for i in range(model.nsensor):
    sensor_name = model.sensor(i).name
    sensor_type = model.sensor(i).type
    sensor_adr = model.sensor_adr[i]
    sensor_dim = model.sensor_dim[i]
    print(f"{sensor_name}: type={sensor_type}, adr={sensor_adr}, dim={sensor_dim}")

# Get sensor data
ultrasonic_id = model.sensor("ultrasonic").id
ultrasonic_adr = model.sensor_adr[ultrasonic_id]
ultrasonic_dim = model.sensor_dim[ultrasonic_id]
ultrasonic_data = data.sensordata[ultrasonic_adr:ultrasonic_adr+ultrasonic_dim]
```

### Sensor Type Constants (mujoco.mjtSensor)

```python
import mujoco

mujoco.mjtSensor.mjSENS_RANGEFINDER    # Distance sensor
mujoco.mjtSensor.mjSENS_ACCELEROMETER  # Linear acceleration
mujoco.mjtSensor.mjSENS_GYRO          # Angular velocity
mujoco.mjtSensor.mjSENS_CAMPROJECTION  # Camera projection data
```

---

## Next Steps

1. **Review this guide** and decide on observation space design
2. **Create sensor XML file** with camera, rangefinder, and IMU definitions
3. **Modify pidog_env.py** to integrate sensor readings
4. **Test in simulation** with simple standing/walking
5. **Retrain policy** with sensor observations
6. **Validate** sensor readings match expected ranges
7. **Deploy to hardware** with real sensor integration

---

## References

- MuJoCo Documentation: https://mujoco.readthedocs.io/
- SunFounder PiDog Docs: https://docs.sunfounder.com/projects/pidog/
- Stable-Baselines3 MultiInput: https://stable-baselines3.readthedocs.io/
- MPU6050 Datasheet: https://invensense.tdk.com/products/motion-tracking/6-axis/mpu-6050/
- HC-SR04 Datasheet: Standard ultrasonic sensor specifications

---

**Document Version:** 1.0
**Last Updated:** 2025-11-09
**Author:** Generated for PiDog RL Training Project
