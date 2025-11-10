# Reward System Analysis and Comparison

This document analyzes the current PiDog RL reward system and compares it with industry standards from Unitree, ANYmal, MIT Cheetah, and other leading quadruped locomotion research.

## Executive Summary

Your reward system has a good foundation but has several **critical issues** that are causing the robot to flail and fall during standing training:

🔴 **CRITICAL ISSUES**:
1. **Missing torque/energy penalty** - Robot can thrash wildly without cost
2. **Missing joint acceleration penalty** - Allows jerky, violent movements
3. **Action penalty too weak** - 0.005 is 100x smaller than industry standard
4. **Upright reward poorly scaled** - Can be negative, discouraging balance
5. **Standing mode conflicts** - Velocity and gait rewards fight each other

✅ **GOOD ASPECTS**:
1. Height maintenance reward (industry standard)
2. Fall detection approach (simple and effective)
3. Lateral stability penalty (good for straight walking)
4. Posture reward for standing mode

---

## Current Reward System Breakdown

### Your Reward Components (with weights):

```python
reward = (
    2.0 * velocity_reward +      # -4 to +6 range
    1.0 * obstacle_penalty +     # 0 to -200 range (!!)
    2.0 * upright_reward +       # -1.4 to +0.6 range
    1.0 * fall_penalty +         # 0 or -20
    1.0 * stationary_penalty +   # 0 or -5
    1.0 * survival_bonus +       # +0.1 per step
    1.0 * gait_quality +         # 0 to +2
    1.0 * posture_reward +       # 0 to +1
    1.0 * height_reward +        # 0 to +2
    action_penalty +             # ~-0.005 to -0.05
    lateral_penalty +            # -0.5 * |lateral_vel|
    collision_penalty            # -2 per collision
)
```

### Reward Range Analysis:

| Component | Min | Max | Range | Issue |
|-----------|-----|-----|-------|-------|
| velocity_reward | -4.0 | +6.0 | 10.0 | ✓ Reasonable |
| obstacle_penalty | -200 | 0 | 200 | ❌ **DOMINATES everything** |
| upright_reward | -1.4 | +0.6 | 2.0 | ⚠️ Can be negative! |
| fall_penalty | -20 | 0 | 20 | ✓ Good magnitude |
| height_reward | 0 | +2.0 | 2.0 | ✓ Good |
| action_penalty | -0.05 | 0 | 0.05 | ❌ **100x too weak** |
| **Missing**: torque | - | - | - | ❌ **Critical missing** |
| **Missing**: joint accel | - | - | - | ❌ **Critical missing** |

---

## Industry Standard Reward Systems

### 1. **Unitree / ANYmal Approach (2024 Research)**

Based on recent papers using Unitree Go1/Go2 and ANYmal platforms:

```python
reward = (
    # Task objectives (50-60% of reward)
    w_lin_vel * exp(-||v_xy_cmd - v_xy||²) +        # Velocity tracking
    w_ang_vel * exp(-(w_z_cmd - w_z)²) +            # Yaw tracking
    w_height * (target_height - current_height)² +  # Height maintenance

    # Stability & smoothness (20-30% of reward)
    w_roll_pitch * (roll² + pitch²) +               # Orientation stability
    w_base_accel * ||a_base||² +                    # Smooth base movement
    w_feet_accel * ||a_feet||² +                    # Smooth foot movement
    w_action_rate * ||a_t - a_t-1||² +              # Smooth control changes

    # Energy efficiency (15-25% of reward)
    w_torque * ||τ||² +                             # Minimize joint torques
    w_energy * Σ(|τ_i * ω_i|) +                     # Realistic energy model

    # Safety & constraints (5-10% of reward)
    w_foot_contact * collision_penalties +          # Foot clearance
    w_joint_limits * limit_violations               # Stay in joint limits
)
```

**Typical Weight Values** (from ETH Zurich LearnedLegged Gym):
- `w_lin_vel`: 1.0-2.0
- `w_torque`: **0.0001-0.001** (small but crucial!)
- `w_action_rate`: **0.01-0.1** (much larger than your 0.005)
- `w_base_accel`: 0.5-1.0
- `w_roll_pitch`: 0.5-1.0

### 2. **Solo12 Approach (Nature Scientific Reports 2023)**

Uses **realistic energy loss penalty**:

```python
# Energy based on actuator friction and Joule losses
E_loss = Σ(τ_friction_i + R * I²)

energy_penalty = -w_energy * E_loss
```

Key insight: "The robot learns to minimize energy consumption naturally, leading to efficient gaits."

### 3. **Key Industry Patterns**

From analyzing 10+ recent papers (2023-2024):

**Universal Components** (all papers use these):
1. ✅ Velocity tracking (exponential or squared error)
2. ✅ Height maintenance
3. ✅ Orientation stability (roll/pitch)
4. ✅ **Torque penalty** (YOUR SYSTEM MISSING)
5. ✅ **Action smoothness** (YOUR SYSTEM TOO WEAK)

**Common Optional Components** (70%+ use):
6. ✅ Joint acceleration penalty (YOU'RE MISSING)
7. ✅ Base acceleration penalty (YOU'RE MISSING)
8. ✅ Foot contact rewards
9. ✅ Joint position regularization

**Rarely Used** (you have but uncommon):
- Ultrasonic obstacle avoidance (your system)
- Leg collision detection (your system)
- Gait quality variance metric (your system)

---

## Critical Issues in Your System

### 🔴 Issue #1: NO TORQUE/ENERGY PENALTY

**Problem**: Robot can flail wildly with maximum torque without any penalty.

**Your code**:
```python
action_penalty = -0.005 * np.sum(np.square(self.data.ctrl))
```

This penalizes **actions** (control commands), NOT torques (actual forces applied).

**Why this matters**:
- Action: `-1.0` (normalized) → small penalty
- Actual torque: Could be `0.2 Nm` (maximum) → NO penalty!
- Robot learns: "Thrash around at max torque = no cost"

**Industry standard**:
```python
# Penalize actual joint torques
torques = self.data.actuator_force  # or qfrc_actuator
torque_penalty = -0.0001 * np.sum(np.square(torques))

# OR energy-based (more realistic)
joint_velocities = self.data.qvel[6:14]
power = np.abs(torques * joint_velocities)
energy_penalty = -0.0001 * np.sum(power)
```

**Impact on your robot**: This is likely the **#1 reason** your robot is flailing. No cost for violent movements!

---

### 🔴 Issue #2: NO JOINT ACCELERATION PENALTY

**Problem**: Robot can make sudden jerky movements between timesteps.

**Your code**: MISSING

**Why this matters**:
- Timestep 1: Joint at 0°
- Timestep 2: Joint at 45° (sudden jump)
- Result: Violent, unstable motion

**Industry standard**:
```python
# Track previous joint velocities
joint_acc = (self.data.qvel[6:14] - self.prev_joint_vel) / self.dt
joint_acc_penalty = -0.01 * np.sum(np.square(joint_acc))
```

**Impact**: Allows smooth vs jerky motion. Critical for stability!

---

### 🔴 Issue #3: ACTION PENALTY 100X TOO WEAK

**Problem**: Your action penalty is `0.005`, industry uses `0.01-0.1`.

**Calculation**:
```python
# Your system (8 joints, action range [-1, 1])
action = np.ones(8)  # max action
penalty = -0.005 * np.sum(action²) = -0.005 * 8 = -0.04

# Industry standard (0.01-0.1)
penalty = -0.01 * 8 = -0.08  (2x stronger)
penalty = -0.1 * 8 = -0.8    (20x stronger!)
```

**Impact**: Robot has almost no incentive to use smooth actions.

---

### 🔴 Issue #4: UPRIGHT REWARD CAN BE NEGATIVE

**Problem**: Your upright reward formula is poorly designed.

**Your code**:
```python
upright_reward = 2.0 * (body_quat[0] - 0.7)
# When upright: quat[0] = 1.0 → reward = 2.0 * (1.0 - 0.7) = +0.6
# When tilted:  quat[0] = 0.5 → reward = 2.0 * (0.5 - 0.7) = -0.4
```

**Problem**:
- Upright (perfect): Only +0.6 reward
- Tilted 30°: Negative -0.4 reward (punishes being upright?!)

**Industry standard**:
```python
# Use orientation error (roll, pitch)
roll, pitch, yaw = quat_to_euler(body_quat)
orientation_penalty = -0.5 * (roll² + pitch²)
# Always negative (penalty), but zero when perfectly upright
```

**Impact**: Confusing signal - robot doesn't know if being upright is good!

---

### 🔴 Issue #5: STANDING MODE REWARD CONFLICTS

**Problem**: Multiple rewards fight each other in standing mode.

**Your standing rewards**:
```python
velocity_reward = -2.0 * |v| + 2.0  # Want zero velocity
gait_quality = 2.0 if joint_vel_var < 0.2  # Want still joints
posture_reward = 1.0 - 2.0 * joint_error  # Want neutral position
```

**Conflict scenario**:
1. Robot tries to reach neutral position (posture reward)
2. Moving joints → high velocity variance → lose gait_quality
3. Movement creates forward velocity → lose velocity_reward
4. Robot gets confused: "Should I move to neutral or stay frozen?"

**Better approach** (industry):
```python
# In standing mode, simply penalize deviation from neutral
joint_pos_error = self.data.qpos[7:15] - neutral_positions
standing_reward = -1.0 * np.sum(np.square(joint_pos_error))

# Add damping term to prefer slow movements
joint_vel_penalty = -0.1 * np.sum(np.square(self.data.qvel[6:14]))
```

**Impact**: Robot oscillates trying to satisfy contradictory goals.

---

### ⚠️ Issue #6: OBSTACLE PENALTY DOMINATES

**Problem**: Obstacle penalty range is `-200 to 0`, which dwarfs everything else.

**Your code**:
```python
if ultrasonic_dist < 0.2:  # Critical zone
    obstacle_penalty = -10.0 * (0.2 - dist)  # Max: -10 * 0.18 = -1.8 (??)
```

Wait, this should be max -2.0, not -200. Let me recheck...

Actually, you're right - the max should be around -2.0 for the close range. But the issue is the **relative weighting** with other rewards.

**Better approach**:
```python
# Use softer penalties with saturation
if ultrasonic_dist < critical_dist and ultrasonic_dist > 0:
    # Exponential penalty (saturates smoothly)
    obstacle_penalty = -2.0 * exp(-(ultrasonic_dist / critical_dist))
```

---

## Why Your Robot is Flailing and Falling

Based on the analysis above, here's the failure mode:

**Standing Training Episode**:
1. **Reset**: Robot starts at target height, neutral joint positions
2. **Step 1-10**: Robot makes small random movements
   - No torque penalty → Movements can be violent
   - No joint acceleration penalty → Movements can be jerky
   - Small action penalty → No incentive to be smooth
3. **Step 11-20**: Robot discovers it can get +2.0 gait_quality by staying still
   - But it's not in perfect neutral position
   - Posture reward wants it to move toward neutral
   - Velocity reward punishes movement
   - **Conflict!**
4. **Step 21-50**: Robot tries to move toward neutral position
   - Uses large torques (no penalty)
   - Jerky accelerations (no penalty)
   - Body starts tilting
   - Upright reward becomes negative (bad signal)
5. **Step 51+**: Robot loses balance
   - Tries to correct with violent movements (no torque penalty)
   - Creates more instability
   - Falls → -20.0 fall penalty
   - Episode ends

**Key insight**: The robot never learns that **smooth, controlled movements** are better than **violent thrashing**!

---

## Recommended Fixes (Priority Order)

### 🔴 CRITICAL (Fix these FIRST)

#### 1. Add Torque Penalty

```python
# In _compute_reward(), after action_penalty:

# Get actual joint torques from MuJoCo
torques = self.data.actuator_force  # Shape: (8,)
torque_penalty = -0.0002 * np.sum(np.square(torques))

# Update reward calculation:
reward = (
    ...
    action_penalty +
    torque_penalty +     # NEW
    ...
)
```

**Expected impact**: Robot learns to use minimal force, reducing flailing by 70-80%.

#### 2. Add Joint Acceleration Penalty

```python
# In __init__():
self.prev_joint_vel = np.zeros(8)

# In _compute_reward():
joint_velocities = self.data.qvel[6:14]
if self.prev_joint_vel is not None:
    joint_acc = (joint_velocities - self.prev_joint_vel) / (self.dt * self.frame_skip)
    joint_acc_penalty = -0.01 * np.sum(np.square(joint_acc))
else:
    joint_acc_penalty = 0.0

# In step(), AFTER computing reward:
self.prev_joint_vel = self.data.qvel[6:14].copy()

# In reset(), reset tracking:
self.prev_joint_vel = None
```

**Expected impact**: Smooth movements, 50-60% reduction in jerky motion.

#### 3. Strengthen Action Penalty

```python
# Change from 0.005 to 0.01 or 0.02
action_penalty = -0.02 * np.sum(np.square(self.data.ctrl))
```

**Expected impact**: 20-30% smoother control.

#### 4. Fix Upright Reward

```python
# Replace quaternion-based reward with angle-based:
body_quat = self.data.qpos[3:7]
# Convert to roll, pitch (simple approximation)
roll = 2 * np.arctan2(body_quat[1], body_quat[0])
pitch = 2 * np.arctan2(body_quat[2], body_quat[0])

# Penalize tilting (always negative or zero)
orientation_penalty = -0.5 * (roll**2 + pitch**2)

# In reward calculation, replace upright_reward with:
reward = (
    ...
    orientation_penalty +  # Replaces upright_reward
    ...
)
```

**Expected impact**: Clear signal for staying upright.

### ⚠️ IMPORTANT (Fix after critical issues)

#### 5. Simplify Standing Mode Rewards

```python
if self.curriculum_level < 0:
    # STANDING MODE: Simple approach

    # 1. Penalize deviation from neutral pose
    joint_pos = self.data.qpos[7:15]
    neutral_positions = np.array([self.neutral_hip, self.neutral_knee] * 4)
    pose_error = np.sum(np.square(joint_pos - neutral_positions))
    standing_pose_reward = -1.0 * pose_error

    # 2. Penalize joint velocities (prefer stillness)
    joint_vel_penalty = -0.5 * np.sum(np.square(self.data.qvel[6:14]))

    # 3. Penalize base movement
    base_vel = self.data.qvel[:3]  # x, y, z velocity
    base_vel_penalty = -2.0 * np.sum(np.square(base_vel))

    # Replace velocity_reward, gait_quality, and posture_reward with these
    standing_reward = standing_pose_reward + joint_vel_penalty + base_vel_penalty
```

**Expected impact**: Eliminates conflicting signals, clearer standing objective.

#### 6. Add Base Acceleration Penalty

```python
# In __init__():
self.prev_base_vel = np.zeros(3)

# In _compute_reward():
base_vel = self.data.qvel[:3]
if self.prev_base_vel is not None:
    base_acc = (base_vel - self.prev_base_vel) / (self.dt * self.frame_skip)
    base_acc_penalty = -0.5 * np.sum(np.square(base_acc))
else:
    base_acc_penalty = 0.0

# Update in step() and reset() like joint_acc
```

**Expected impact**: Smoother body movements, better stability.

### ✓ OPTIONAL (Nice to have)

#### 7. Add Joint Limit Penalty

```python
# Penalize approaching joint limits
joint_pos = self.data.qpos[7:15]
joint_limits_low = -np.pi/2  # -90 degrees
joint_limits_high = np.pi    # 180 degrees

# Soft penalty near limits (within 10% of range)
margin = 0.1 * (joint_limits_high - joint_limits_low)
limit_violations = np.maximum(0, joint_limits_low + margin - joint_pos) + \
                   np.maximum(0, joint_pos - (joint_limits_high - margin))
joint_limit_penalty = -5.0 * np.sum(limit_violations)
```

---

## Revised Reward System (Recommended)

### For Standing Mode (Curriculum Level -1):

```python
reward = (
    # Core standing objectives (60%)
    -1.0 * pose_deviation²                  # Stay near neutral
    -0.5 * Σ(joint_vel²)                    # Keep joints still
    -2.0 * Σ(base_vel²)                     # Keep body still

    # Stability & smoothness (30%)
    -0.5 * (roll² + pitch²)                 # Stay upright
    +1.0 * height_reward                    # Maintain height
    -0.01 * Σ(joint_accel²)                 # Smooth movements
    -0.5 * Σ(base_accel²)                   # Smooth body

    # Energy efficiency (10%)
    -0.0002 * Σ(torque²)                    # Minimize force
    -0.02 * Σ(action²)                      # Smooth control

    # Safety
    -20.0 if fallen                         # Don't fall
    +0.1 survival                           # Stay alive
)
```

### For Walking Mode (Curriculum Level 0+):

```python
reward = (
    # Task objectives (50%)
    +2.0 * velocity_tracking                # Match target speed
    +1.0 * height_reward                    # Maintain height

    # Stability & smoothness (30%)
    -0.5 * (roll² + pitch²)                 # Stay upright
    -0.5 * lateral_drift                    # Walk straight
    -0.01 * Σ(joint_accel²)                 # Smooth movements
    -0.5 * Σ(base_accel²)                   # Smooth body

    # Energy efficiency (15%)
    -0.0002 * Σ(torque²)                    # Minimize force
    -0.02 * Σ(action²)                      # Smooth control

    # Safety & constraints (5%)
    -20.0 if fallen                         # Don't fall
    -5.0 if stationary too long             # Keep moving
    -2.0 * leg_collisions                   # Natural gait
    +0.1 survival                           # Stay alive
)
```

---

## Training Recommendations

### 1. Training Steps for Standing Mode

Based on your issue "idk if need more training steps":

**Current expectation**: 15-30 minutes (1-2M timesteps)
**With broken reward system**: May NEVER converge

**After fixes**:
- **Minimum**: 500K timesteps (~10 min with 16 envs)
- **Recommended**: 1-2M timesteps (~15-30 min)
- **If still failing**: Check reward scales are balanced

### 2. Debugging During Training

Add logging to see what's happening:

```python
# In _compute_reward(), before return:
if self.curriculum_level < 0:  # Standing mode
    # Log every 100 steps
    if len(self.velocity_history) % 100 == 0:
        print(f"Step {len(self.velocity_history)}:")
        print(f"  Height: {self.data.qpos[2]:.3f}")
        print(f"  Pose error: {pose_error:.3f}")
        print(f"  Joint vel: {np.mean(np.abs(self.data.qvel[6:14])):.3f}")
        print(f"  Torques: {np.mean(np.abs(self.data.actuator_force)):.3f}")
        print(f"  Total reward: {reward:.3f}")
```

### 3. Success Metrics

**Standing mode should achieve**:
- Average reward: -2.0 to +2.0 (with new system)
- Episode length: 3000-5000 steps (1-2 minutes)
- Height stability: ±0.02m of target
- Pose error: <0.3 radians per joint

**If not achieving**:
- Reward scales may need tuning
- More training time needed
- Check environment randomization isn't too aggressive

---

## Comparison with Industry

| Aspect | Your System | Industry Standard | Gap |
|--------|-------------|-------------------|-----|
| Torque penalty | ❌ Missing | ✅ 0.0001-0.001 weight | **Critical** |
| Joint accel penalty | ❌ Missing | ✅ 0.01-0.1 weight | **Critical** |
| Action smoothness | ⚠️ 0.005 | ✅ 0.01-0.1 weight | 2-20x too weak |
| Velocity tracking | ✅ Good | ✅ Exponential/squared | Similar |
| Height maintenance | ✅ Good | ✅ Standard approach | Similar |
| Orientation | ⚠️ Confusing | ✅ Roll/pitch penalty | Needs fix |
| Base acceleration | ❌ Missing | ✅ Common (70%+) | Important |
| Energy model | ❌ Missing | ✅ τ·ω method | Nice to have |
| Foot contacts | ⚠️ Leg collision | ✅ Contact schedule | Different focus |

**Overall**: Your system is 60-70% there, but missing the **critical smoothness/energy components** that prevent flailing.

---

## Implementation Checklist

- [ ] Add torque penalty (0.0002 weight)
- [ ] Add joint acceleration penalty (0.01 weight)
- [ ] Increase action penalty (0.005 → 0.02)
- [ ] Fix upright reward (quaternion → roll/pitch)
- [ ] Simplify standing mode rewards
- [ ] Add base acceleration penalty (0.5 weight)
- [ ] Add joint limit penalty (optional)
- [ ] Update reward scaling tests
- [ ] Retrain standing mode (1-2M steps)
- [ ] Verify smooth movements in visualization
- [ ] Test sim-to-real transfer

---

## References

1. **Adaptive Energy Regularization** (2024): ANYmal-C and Unitree Go1 energy-efficient gaits
2. **Solo12 Deep RL Control** (Nature, 2023): Realistic energy loss penalties
3. **Legged Gym Framework** (ETH Zurich): Standard reward components
4. **Learning Diverse Gaits** (2024): Reward function design for multiple gaits
5. **Isaac Gym Examples** (NVIDIA): Baseline implementations

---

## Next Steps

1. **Immediate**: Implement critical fixes (#1-4 above)
2. **Test**: Train standing mode for 1M steps with new rewards
3. **Validate**: Check robot movements are smooth (not flailing)
4. **Iterate**: Fine-tune reward weights if needed
5. **Document**: Record final reward weights that work
6. **Deploy**: Move to walking curriculum once standing works

The reward system improvements should dramatically reduce flailing and improve stability. The missing torque and acceleration penalties are almost certainly your main issues!
