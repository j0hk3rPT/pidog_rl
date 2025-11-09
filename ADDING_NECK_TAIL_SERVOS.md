# Adding Neck and Tail Servo Control to PiDog

## Overview

The current PiDog model has **8 actuated joints** (leg servos only). The real PiDog robot has additional servos:
- **3 neck servos** (neck1, neck2, neck3) - for head movement
- **1 tail servo** - for balance

This document explains how to add these servos to enable:
1. **Head stability control** - keep camera/sonar level for better sensor visibility
2. **Balance assistance** - use tail for dynamic stability (like real quadrupeds)

---

## Current Model Structure

### Existing Actuators (8 total)
```
8 leg servos:
- back_right_hip, back_right_knee
- front_right_hip, front_right_knee
- back_left_hip, back_left_knee
- front_left_hip, front_left_knee
```

### Existing Rigid Bodies (no joints)
```
Neck chain:
- neck1 → neck2 → neck3 → head

Currently FIXED with euler rotations, no joints
```

### Missing
- **Tail body** - not in current model
- **Neck joints** - neck bodies exist but are rigid
- **Actuators** for neck and tail

---

## Implementation Plan

### Step 1: Add Joints to Neck Bodies

Modify `model/pidog.xml` to add hinge joints at each neck segment:

```xml
<!-- In <body name="base"> section, before neck1 -->

<body name="neck1" pos="0.028 0.135 0.026" euler="0 1.57075 0.785375">

    <!-- ADD THIS JOINT -->
    <joint name="neck1_joint"
           type="hinge"
           axis="0 0 1"
           range="-0.785 0.785"
           pos="0 0 0"/>

    <geom name="neck1_c0" mass="0.004678" mesh="neck1_c0" class="collision"/>
    <!-- ... rest of neck1 geoms ... -->

    <body name="neck2" pos="0.0 0.0175 0.0025" euler="0.0 1.57075 0.0">

        <!-- ADD THIS JOINT -->
        <joint name="neck2_joint"
               type="hinge"
               axis="0 1 0"
               range="-0.524 0.524"
               pos="0 0 0"/>

        <!-- ... rest of neck2 ... -->

        <body name="neck3" pos="0.0015 0.0115 -0.0075" euler="-0.785375 0 0.0">

            <!-- ADD THIS JOINT -->
            <joint name="neck3_joint"
                   type="hinge"
                   axis="0 1 0"
                   range="-0.524 0.524"
                   pos="0 0 0"/>

            <!-- ... rest of neck3 and head ... -->
```

**Joint ranges:**
- neck1: ±45° (±0.785 rad) - yaw (left/right)
- neck2: ±30° (±0.524 rad) - pitch (up/down)
- neck3: ±30° (±0.524 rad) - pitch (nod)

---

### Step 2: Add Tail Body and Joint

Add tail to base body in `model/pidog.xml`:

```xml
<!-- In <body name="base"> section, after the 4 legs -->

<!-- Tail for balance -->
<body name="tail" pos="-0.02 0.04 0.02" euler="0 -0.785 0">

    <!-- Tail joint - can swing up/down for balance -->
    <joint name="tail_joint"
           type="hinge"
           axis="1 0 0"
           range="-1.047 0.524"
           pos="0 0 0"/>

    <!-- Tail segment (simple cylinder) -->
    <geom name="tail_geom"
          type="capsule"
          size="0.008 0.06"
          mass="0.01"
          rgba="0.3 0.3 0.3 1"
          fromto="0 0 0 0 -0.05 -0.04"/>

    <!-- Tail tip (for visual/collision) -->
    <geom name="tail_tip"
          type="sphere"
          size="0.01"
          pos="0 -0.05 -0.04"
          rgba="0.2 0.2 0.2 1"/>
</body>
```

**Tail range:**
- Range: -60° to +30° (±1.047 to 0.524 rad)
- Axis: pitch (up/down swing)

---

### Step 3: Add Actuators

Modify `model/pidog_actuators.xml` to add 4 new servos:

```xml
<actuator>
    <!-- ... existing 8 leg actuators ... -->

    <!-- ==================== NECK SERVOS ==================== -->

    <!-- Neck 1 - Yaw (left/right) -->
    <position name="neck1_actuator"
              joint="neck1_joint"
              ctrlrange="-0.785 0.785"
              forcerange="-0.137 0.137"
              kp="3.0"
              gear="1"/>

    <!-- Neck 2 - Pitch (up/down) -->
    <position name="neck2_actuator"
              joint="neck2_joint"
              ctrlrange="-0.524 0.524"
              forcerange="-0.137 0.137"
              kp="3.0"
              gear="1"/>

    <!-- Neck 3 - Pitch (nod) -->
    <position name="neck3_actuator"
              joint="neck3_joint"
              ctrlrange="-0.524 0.524"
              forcerange="-0.137 0.137"
              kp="3.0"
              gear="1"/>

    <!-- ==================== TAIL SERVO ==================== -->

    <position name="tail_actuator"
              joint="tail_joint"
              ctrlrange="-1.047 0.524"
              forcerange="-0.137 0.137"
              kp="3.0"
              gear="1"/>
</actuator>
```

**Note:** Lower `kp` (3.0 vs 5.0) for neck/tail since they have less load.

---

### Step 4: Update Environment Code

Modify `pidog_env/pidog_env.py`:

#### 4.1 Update Joint Count

```python
# OLD:
self.n_joints = 8  # 4 legs × 2 joints per leg

# NEW:
self.n_joints = 12  # 8 leg + 3 neck + 1 tail
self.n_leg_joints = 8
self.n_neck_joints = 3
self.n_tail_joints = 1
```

#### 4.2 Update Action Space

```python
# Action: 12 continuous values
self.action_space = spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(self.n_joints,),  # Now 12 instead of 8
    dtype=np.float32
)
```

#### 4.3 Update Observation Space

```python
# OLD vector observation: 31 dimensions
# NEW vector observation: 39 dimensions
#   - Joint positions: 12 (8 leg + 3 neck + 1 tail)
#   - Joint velocities: 12
#   - Body quaternion: 4
#   - Linear velocity: 3
#   - Angular velocity: 3
#   - Height: 1
#   - Ultrasonic: 1
#   - IMU accel: 3
# TOTAL: 39 dimensions

vector_obs_dim = 12 + 12 + 4 + 3 + 3 + 1 + 1 + 3  # = 39
```

#### 4.4 Update `_get_obs()` Method

```python
def _get_obs(self):
    # Joint positions (ALL 12 joints now)
    joint_pos = self.data.qpos[7:19].copy()  # Was [7:15], now [7:19]

    # Joint velocities (ALL 12 joints)
    joint_vel = self.data.qvel[6:18].copy()  # Was [6:14], now [6:18]

    # ... rest of observations ...

    vector_obs = np.concatenate([
        joint_pos,        # [0:12]  - 12 joints
        joint_vel,        # [12:24] - 12 velocities
        body_quat,        # [24:28]
        body_lin_vel,     # [28:31]
        body_ang_vel,     # [31:34]
        body_height,      # [34]
        ultrasonic_dist,  # [35]
        imu_accel,        # [36:39]
    ]).astype(np.float32)
```

#### 4.5 Update `_scale_action()` Method

```python
def _scale_action(self, action):
    """Scale actions for 12 joints (8 leg + 3 neck + 1 tail)."""
    scaled_action = np.zeros(12)

    # Legs (indices 0-7): same as before
    for i in range(8):
        low, high = self.servo_specs["range"]
        scaled_action[i] = low + (action[i] + 1.0) * 0.5 * (high - low)

    # Neck servos (indices 8-10): different ranges
    # neck1: -45° to +45° yaw
    neck1_range = (-np.pi/4, np.pi/4)
    scaled_action[8] = neck1_range[0] + (action[8] + 1.0) * 0.5 * (neck1_range[1] - neck1_range[0])

    # neck2, neck3: -30° to +30° pitch
    neck_pitch_range = (-np.pi/6, np.pi/6)
    scaled_action[9] = neck_pitch_range[0] + (action[9] + 1.0) * 0.5 * (neck_pitch_range[1] - neck_pitch_range[0])
    scaled_action[10] = neck_pitch_range[0] + (action[10] + 1.0) * 0.5 * (neck_pitch_range[1] - neck_pitch_range[0])

    # Tail (index 11): -60° to +30°
    tail_range = (-np.pi/3, np.pi/6)
    scaled_action[11] = tail_range[0] + (action[11] + 1.0) * 0.5 * (tail_range[1] - tail_range[0])

    # Apply velocity limiting (for all 12 joints)
    # ... rest of velocity limiting code ...
```

#### 4.6 Update Reset Method

```python
def reset(self, seed=None, options=None):
    # ... existing reset code ...

    # Initialize leg joints (indices 7-14 in qpos)
    for i in range(4):
        self.data.qpos[7 + i*2] = self.neutral_hip + np.random.uniform(-0.1, 0.1)
        self.data.qpos[7 + i*2 + 1] = self.neutral_knee + np.random.uniform(-0.1, 0.1)

    # Initialize neck joints (indices 15-17 in qpos) - start neutral (0)
    self.data.qpos[15] = 0.0  # neck1 - straight ahead
    self.data.qpos[16] = 0.0  # neck2 - level
    self.data.qpos[17] = 0.0  # neck3 - level

    # Initialize tail (index 18 in qpos) - slight downward angle
    self.data.qpos[18] = -0.3  # -17° down

    # ... rest of reset ...
```

---

### Step 5: Add Head Stability Reward

Add to `_compute_reward()` method:

```python
def _compute_reward(self):
    # ... existing reward components ...

    # ============= 7. HEAD STABILITY (for sensor visibility) =============
    # Reward for keeping head level so camera/ultrasonic can see properly
    # Get neck joint positions
    neck1_angle = self.data.qpos[15]  # Yaw
    neck2_angle = self.data.qpos[16]  # Pitch
    neck3_angle = self.data.qpos[17]  # Pitch

    # Penalize deviation from level (0 radians)
    # Small weight since this is auxiliary, not critical
    head_stability_penalty = -0.5 * (abs(neck1_angle) + abs(neck2_angle) + abs(neck3_angle))

    # Combine rewards
    reward = (
        3.0 * velocity_reward +
        1.0 * obstacle_penalty +
        1.5 * upright_reward +
        1.0 * stationary_penalty +
        action_penalty +
        lateral_penalty +
        head_stability_penalty  # NEW!
    )

    return reward
```

**Rationale:**
- Keeping head level helps camera see forward and ultrasonic detect obstacles
- Small penalty weight (0.5) - it's helpful but not critical
- Robot learns to naturally stabilize head while moving

---

## Testing the Changes

### Test 1: Verify Model Loads

```python
import mujoco
model = mujoco.MjModel.from_xml_path('model/pidog.xml')
print(f"Total actuators: {model.nu}")  # Should be 12
print(f"Total DOF: {model.nv}")  # Should be 6 (free) + 12 (actuated) = 18
```

### Test 2: Check Joint Names

```python
for i in range(model.njnt):
    print(f"Joint {i}: {model.joint(i).name}")

# Expected output:
# Joint 0: (free joint - no name)
# Joint 1: back_right_hip
# Joint 2: back_right_knee
# ...
# Joint 8: front_left_knee
# Joint 9: neck1_joint
# Joint 10: neck2_joint
# Joint 11: neck3_joint
# Joint 12: tail_joint
```

### Test 3: Test Environment

```python
from pidog_env import PiDogEnv

env = PiDogEnv(use_camera=False)
print(f"Action space: {env.action_space}")  # Box(12,)
print(f"Observation space: {env.observation_space}")  # Box(39,) or Dict with vector: Box(39,)

obs, _ = env.reset()
action = env.action_space.sample()
obs, reward, term, trunc, info = env.step(action)

print(f"Observation shape: {obs.shape if not isinstance(obs, dict) else obs['vector'].shape}")
print("✓ Environment works with 12 joints")
```

---

## Training Considerations

### Increased Complexity
- **More actions**: 12 vs 8 (50% increase)
- **Harder to learn**: More DOF = larger action space
- **Solution**: Longer training (2-5M steps instead of 1M)

### Potential Benefits
- **Better balance**: Tail helps with dynamic stability
- **Better sensing**: Stable head = more reliable camera/ultrasonic
- **More natural gait**: Real quadrupeds use head/tail for balance

### Reward Weight Tuning

You may need to adjust weights:

```python
# If head wobbles too much:
head_stability_penalty = -1.0 * (...)  # Increase from 0.5 to 1.0

# If tail doesn't help balance:
# Add explicit tail usage reward based on body angular momentum
```

---

## Alternative: Simplified Approach

If adding neck/tail is too complex, you can:

### Option A: Keep Head Fixed, Add Only Tail
- Don't add neck joints (keep head rigid)
- Only add tail servo for balance
- Simpler: only 9 total joints instead of 12

### Option B: Add Neck Later
1. Train with 8 leg joints first
2. Get good walking policy
3. Then add neck servos and fine-tune

### Option C: Virtual Head Stabilization
- Keep neck fixed in simulation
- Assume real robot will use PID controller to stabilize head
- Don't train neck control in RL

---

## Real Hardware Notes

### PiDog Neck Servos
The real PiDog has **3 neck servos** that control:
- **Servo 1**: Yaw (rotate head left/right)
- **Servo 2**: Pitch (tilt head up/down)
- **Servo 3**: Roll (tilt head side-to-side) *OR* additional pitch

You may need to verify the actual servo configuration on your physical robot.

### Tail
Real PiDog might not have a tail servo. If not:
- You can skip the tail implementation
- Or add it in simulation for training, then ignore in deployment
- Tail helps in simulation for balance learning

---

## Summary

**To add neck and tail control:**

1. ✅ Add 3 hinge joints to neck (neck1, neck2, neck3)
2. ✅ Add 1 tail body with hinge joint
3. ✅ Add 4 actuators to pidog_actuators.xml
4. ✅ Update environment: n_joints = 12
5. ✅ Update observation space: 39 dimensions
6. ✅ Add head stability reward
7. ✅ Test model loads correctly
8. ✅ Retrain with longer timesteps (2-5M)

**Complexity:** Medium (requires XML editing and environment updates)
**Benefit:** More realistic control, better balance, stable sensors
**Alternative:** Train with 8 joints first, add neck/tail later

---

**Status:** Ready to implement when you're comfortable with current 8-joint training
**Recommendation:** Master 8-joint training first, then add complexity

---

**Last Updated:** 2025-11-09
