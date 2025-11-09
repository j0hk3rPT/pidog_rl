# Coordinate System Fix: Sunfounder → MuJoCo

## Problem: Robot Walks Backward and Falls

The robot was walking backward instead of forward despite using the correct Sunfounder gait code. This indicates a **coordinate system orientation mismatch** between Sunfounder's frame and MuJoCo's world frame.

## Root Cause Analysis

### 1. MuJoCo World Frame
- **Forward direction**: Positive X axis (red axis in visualization)
- **Right direction**: Positive Y axis
- **Up direction**: Positive Z axis
- Robot base is rotated: `euler="0.0 0.0 -1.57075"` (-90° around Z)

### 2. Sunfounder Coordinate System
In the original Sunfounder PiDog code:
- **Y coordinate**: Forward/backward position of foot relative to hip
- **Z coordinate**: Height of foot relative to hip
- Positive Y = foot moves forward
- Negative Y = foot moves backward

### 3. The Mismatch
When Sunfounder's gait generator produces:
- Positive Y coordinates (foot forward)
- MuJoCo interprets this as backward movement due to frame rotation

Result: Robot walks backward when it should walk forward!

## Evolution of Fixes

### Version 1: extract_sunfounder_demos.py (WRONG)
**Issues found:**
- ❌ Wrong actuator ordering (Sunfounder order instead of MuJoCo order)
- ❌ Wrong joint range (0 to π instead of -π/2 to π)

### Version 2: extract_sunfounder_demos_fixed.py (BETTER)
**Fixed:**
- ✅ Correct actuator ordering: RR, FR, RL, FL
- ✅ Servo range shifted to 0-π

**Still wrong:**
- ❌ Normalization assumed [0, π] range

### Version 3: extract_sunfounder_demos_correct.py (CLOSER)
**Fixed:**
- ✅ Correct action normalization for environment's [-π/2, π] range

**Still wrong:**
- ❌ Forward direction flipped (walks backward)

### Version 4: extract_sunfounder_demos_flipped.py (FINAL FIX)
**The complete fix:**
- ✅ Correct actuator ordering
- ✅ Correct action normalization
- ✅ **FLIPPED Y COORDINATE** to match coordinate frames

## The Final Solution

```python
@staticmethod
def coord2angle(y, z):
    """Convert (y, z) coordinates to joint angles."""
    LEG = 62
    FOOT = 62

    # KEY FIX: Flip Y coordinate
    y = -y  # ← This fixes the backward walking!

    # Rest of IK calculation unchanged
    u = sqrt(y**2 + z**2)
    # ... inverse kinematics ...
    return hip_angle, knee_angle
```

## Why This Works

1. **Sunfounder's positive Y** = foot moves forward in its local frame
2. **MuJoCo's interpretation** = due to base rotation, this becomes backward movement
3. **Solution** = Negate Y coordinate so forward becomes forward

## Complete Transformation Pipeline

```
Sunfounder Gait
    ↓
Foot positions (Y, Z) in mm
    ↓
FLIP Y COORDINATE (y = -y)  ← NEW FIX
    ↓
Inverse Kinematics
    ↓
Joint angles (hip, knee) in radians
    ↓
Mirror right-side legs
    ↓
Reorder: Sunfounder → MuJoCo
    ↓
Normalize: [-π/2, π] → [-1, 1]
    ↓
Actions for MuJoCo environment
```

## Testing the Fix

```bash
# Extract demonstrations with flipped coordinates
python extract_sunfounder_demos_flipped.py --n-cycles 20

# Visualize to verify robot walks FORWARD
python visualize_demonstrations.py --demo-file demonstrations/sunfounder_flipped.pkl
```

## Expected Behavior

**Before fix:**
- Robot walks backward
- Falls forward on face
- Unstable gait

**After fix:**
- ✅ Robot walks forward
- ✅ Stable gait
- ✅ Matches original Sunfounder behavior

## All Critical Fixes Summary

| Issue | File | Fix |
|-------|------|-----|
| Actuator order wrong | `_fixed.py` | Reorder to RR, FR, RL, FL |
| Action normalization wrong | `_correct.py` | Use [-π/2, π] range |
| Forward direction flipped | `_flipped.py` | Negate Y coordinate |
| Height too low | `_v2.py` | Adjustable standing height |

## Recommended Usage

**For imitation learning with correct forward walking:**

```bash
# Use the flipped version
python extract_sunfounder_demos_flipped.py \
    --n-cycles 50 \
    --standing-height 80 \
    --output-file demonstrations/sunfounder_final.pkl

# Verify it works
python visualize_demonstrations.py \
    --demo-file demonstrations/sunfounder_final.pkl
```

## Technical Details

### Actuator Ordering (from pidog_actuators.xml)
```
actuator[0,1] = back_right_hip, back_right_knee   (RR)
actuator[2,3] = front_right_hip, front_right_knee (FR)
actuator[4,5] = back_left_hip, back_left_knee     (RL)
actuator[6,7] = front_left_hip, front_left_knee   (FL)
```

### Environment Servo Range (from pidog_env.py)
```python
servo_specs["range"] = (-np.pi/2, np.pi)  # -90° to 180°
```

### Action Mapping
```python
# Normalized action [-1, 1] maps to servo angle [-π/2, π]
normalized = (angle - (-π/2)) / (π - (-π/2)) * 2 - 1
# Simplified:
normalized = (angle + π/2) / (3π/2) * 2 - 1
```

## Debugging Tools

If you still have issues:

```bash
# Test coordinate system orientation
python test_coordinate_system.py

# Debug specific height settings
python debug_sunfounder_extraction.py --height-offset 20

# Compare with original Sunfounder
python compare_with_sunfounder.py
```

## Next Steps

After verifying the flipped version works:

1. Use it for behavioral cloning pretraining
2. Fine-tune with RL
3. Adjust `standing_height` if needed (try 80-110mm)
4. Train on more diverse gaits (left/right turning)
