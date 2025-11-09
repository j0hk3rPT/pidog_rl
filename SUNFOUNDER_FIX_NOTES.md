# Sunfounder Gait Extraction - Critical Fixes

## Problems Found

### 1. ❌ **WRONG Actuator Order** (Critical!)

**The Problem:**
- Sunfounder leg order: `FL, FR, RL, RR` (indices 0, 1, 2, 3)
- MuJoCo actuator order: `RR, FR, RL, FL` (back-right, front-right, back-left, front-left)

**Old (Broken) Code:**
```python
# Assumed MuJoCo order was FL, FR, RL, RR
leg_angles.extend([hip_angle, knee_angle])
# Just appended in Sunfounder order → WRONG!
```

**Result:** Front left commands went to back right, etc. **Legs were completely mixed up!**

**New (Fixed) Code:**
```python
# Extract in Sunfounder order
FL_hip, FL_knee = sunfounder_angles[0]
FR_hip, FR_knee = sunfounder_angles[1]
RL_hip, RL_knee = sunfounder_angles[2]
RR_hip, RR_knee = sunfounder_angles[3]

# Reorder to MuJoCo actuator order
mujoco_angles = [
    RR_hip, RR_knee,  # Actuators 0,1
    FR_hip, FR_knee,  # Actuators 2,3
    RL_hip, RL_knee,  # Actuators 4,5
    FL_hip, FL_knee,  # Actuators 6,7
]
```

### 2. ❌ **WRONG Joint Range** (Critical!)

**The Problem:**
- MuJoCo servos: `ctrlrange="0 3.14159"` (0 to π, **NO NEGATIVES!**)
- Old normalization: Used `(-π/2, π)` which includes negative values

**From `pidog_actuators.xml`:**
```xml
<position name="back_right_hip_actuator"
          joint="back_right_hip"
          ctrlrange="0 3.14159"   <!-- 0 to π only! -->
```

Real SF006FM servos are 0-180° (0 to π radians). They **cannot go negative**.

**Old (Broken) Code:**
```python
servo_range = (-np.pi/2, np.pi)  # WRONG! Servos can't go to -π/2
normalized = (angle - servo_range[0]) / (servo_range[1] - servo_range[0]) * 2 - 1
```

**New (Fixed) Code:**
```python
# Shift angles to positive range (add π/2 offset)
shifted_angle = angle + pi/2

# Normalize from [0, π] to [-1, 1]
servo_min = 0.0
servo_max = pi
normalized = (shifted_angle - servo_min) / (servo_max - servo_min) * 2 - 1
```

### 3. ✓ **Left/Right Mirroring** (This was actually correct!)

The mirroring logic was fine:
```python
if i % 2 != 0:  # Right side legs (FR=1, RR=3)
    hip_angle = -hip_angle
    knee_angle = -knee_angle
```

This correctly mirrors right-side legs as in the original Sunfounder code.

## How to Use the Fixed Version

### Extract Demonstrations (Fixed)
```bash
python extract_sunfounder_demos_fixed.py \
    --n-cycles 20 \
    --output-file demonstrations/sunfounder_demos_fixed.pkl
```

### Verify They Work
```bash
python visualize_demonstrations.py \
    --demo-file demonstrations/sunfounder_demos_fixed.pkl \
    --n-cycles 3
```

**What to look for:**
- ✓ All four legs move correctly
- ✓ Left and right sides are symmetric
- ✓ Robot walks forward smoothly
- ✓ No weird jerky motions
- ✓ Body stays level and stable

### Train with Fixed Demonstrations
```bash
python train_pretrain_finetune.py \
    --sunfounder-demos demonstrations/sunfounder_demos_fixed.pkl \
    --total-bc-epochs 300 \
    --total-rl-timesteps 500000
```

## Technical Details

### Actuator Ordering Explained

**MuJoCo XML order (from `pidog_actuators.xml`):**
```xml
1. back_right_hip_actuator    (index 0)
2. back_right_knee_actuator   (index 1)
3. front_right_hip_actuator   (index 2)
4. front_right_knee_actuator  (index 3)
5. back_left_hip_actuator     (index 4)
6. back_left_knee_actuator    (index 5)
7. front_left_hip_actuator    (index 6)
8. front_left_knee_actuator   (index 7)
```

**Sunfounder `legs_angle_calculation` order:**
```python
for i, coord in enumerate(coords):  # coords = [FL, FR, RL, RR]
    # i=0: FL (front left)
    # i=1: FR (front right)
    # i=2: RL (rear left)
    # i=3: RR (rear right)
```

**The mapping:**
| Sunfounder Index | Leg | MuJoCo Indices |
|------------------|-----|----------------|
| 0 | FL | 6, 7 |
| 1 | FR | 2, 3 |
| 2 | RL | 4, 5 |
| 3 | RR | 0, 1 |

### Joint Range Details

**SF006FM Servo Specs:**
- Physical range: 0° to 180° (0 to π radians)
- **Cannot go negative!**
- Neutral standing: ~90° (π/2)

**Why angles can be negative in IK:**
The inverse kinematics outputs angles relative to the leg's local frame, which can be negative. But the servo's absolute position must be 0 to π.

**Solution:**
Add an offset (π/2) to shift the IK output into the valid servo range.

## Comparison: Old vs New

### Old Version (`extract_sunfounder_demos.py`)
```python
# ❌ WRONG: Direct append in Sunfounder order
for i, (y, z) in enumerate(coords):
    hip_angle, knee_angle = cls.coord2angle(y, z)
    if i % 2 != 0:
        hip_angle = -hip_angle
        knee_angle = -knee_angle
    leg_angles.extend([hip_angle, knee_angle])  # ❌ Wrong order!

# ❌ WRONG: Negative servo range
servo_range = (-np.pi/2, np.pi)  # ❌ Servos can't go negative!
```

### New Version (`extract_sunfounder_demos_fixed.py`)
```python
# ✓ CORRECT: Reorder to MuJoCo actuator sequence
FL_hip, FL_knee = sunfounder_angles[0]
FR_hip, FR_knee = sunfounder_angles[1]
RL_hip, RL_knee = sunfounder_angles[2]
RR_hip, RR_knee = sunfounder_angles[3]

mujoco_angles = [
    RR_hip, RR_knee,  # ✓ Back right first
    FR_hip, FR_knee,  # ✓ Front right second
    RL_hip, RL_knee,  # ✓ Back left third
    FL_hip, FL_knee,  # ✓ Front left last
]

# ✓ CORRECT: Shift to positive range, then normalize
shifted_angle = angle + pi/2  # ✓ Offset for servo range
servo_min = 0.0
servo_max = pi  # ✓ Correct servo limits
```

## Why This Matters

The old version had **completely scrambled leg commands**:

| Command For | Old Code Sent To | Result |
|-------------|------------------|--------|
| Front Left | Back Right | ❌ Wrong leg! |
| Front Right | Front Right | ✓ Correct (lucky!) |
| Rear Left | Back Left | ❌ Wrong name (same leg, different convention) |
| Rear Right | Front Left | ❌ Wrong leg! |

Plus, angles were out of range for the servos.

**Result:** Robot couldn't walk properly, legs moved erratically, left/right sides mismatched.

## Files

- `extract_sunfounder_demos.py` - **OLD (BROKEN)** - Do not use!
- `extract_sunfounder_demos_fixed.py` - **NEW (FIXED)** - Use this one!

## Next Steps

1. **Extract fixed demonstrations:**
   ```bash
   python extract_sunfounder_demos_fixed.py --n-cycles 20
   ```

2. **Verify they work:**
   ```bash
   python visualize_demonstrations.py \
       --demo-file demonstrations/sunfounder_demos_fixed.pkl
   ```

   Watch carefully - the robot should walk smoothly with all legs coordinated.

3. **Train with fixed demos:**
   ```bash
   python train_pretrain_finetune.py \
       --sunfounder-demos demonstrations/sunfounder_demos_fixed.pkl \
       --total-bc-epochs 300 \
       --total-rl-timesteps 500000
   ```

## Verification Checklist

When you visualize the fixed demonstrations, you should see:

- ✅ **Diagonal trot pattern**: FL+RR move together, FR+RL move together
- ✅ **Smooth forward motion**: Robot walks forward, not sideways
- ✅ **Symmetric legs**: Left and right sides mirror each other
- ✅ **Stable body**: Body stays level, doesn't tilt excessively
- ✅ **No weird angles**: All leg movements look natural
- ✅ **Coordinated motion**: All legs work together

If you see these, the fix worked! If not, there may be other issues.
