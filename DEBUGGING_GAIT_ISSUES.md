# Debugging Gait Extraction Issues

## Problem Statement

The robot is not walking properly when using extracted Sunfounder demonstrations. We've tried several fixes but the robot still doesn't walk forward correctly.

## Systematic Debugging Approach

### Step 1: Verify Basic IK and Standing Position

**First, test if the inverse kinematics calculations produce correct standing angles:**

```bash
docker-compose run --rm pidog_rl python test_standing_position.py --height 80
```

This will:
- Calculate neutral standing angles using IK
- Apply them to all 4 legs (with right-side mirroring)
- Show the robot standing for 10 seconds

**What to check:**
- ✓ Robot stands upright and balanced
- ✓ All 4 legs at roughly same height
- ✓ Robot doesn't fall or tilt
- ✓ Height looks correct (~80mm body clearance)

**If standing FAILS**, test without mirroring:
```bash
docker-compose run --rm pidog_rl python test_standing_position.py --height 80 --no-mirror
```

### Step 2: Test All Gait Versions

**Extract demonstrations using all versions:**

```bash
docker-compose run --rm pidog_rl ./test_all_versions.sh
```

This creates 3 test files:
1. `test_v1_flipped.pkl` - With Y coordinate flipped
2. `test_v2_no_mirror.pkl` - Without right-side angle negation
3. `test_v3_original.pkl` - Original version (baseline)

### Step 3: Visualize Each Version

**Test each version visually:**

```bash
# Version 1: Flipped Y
docker-compose run --rm pidog_rl python visualize_demonstrations.py \
    --demo-file demonstrations/test_v1_flipped.pkl

# Version 2: No mirroring
docker-compose run --rm pidog_rl python visualize_demonstrations.py \
    --demo-file demonstrations/test_v2_no_mirror.pkl

# Version 3: Original
docker-compose run --rm pidog_rl python visualize_demonstrations.py \
    --demo-file demonstrations/test_v3_original.pkl
```

**What to watch for:**
- ✓ Robot walks FORWARD (toward red X-axis)
- ✓ Robot stays upright and balanced
- ✓ Gait looks natural and smooth
- ✓ Forward velocity is positive (>0.1 m/s)
- ✓ Episodes don't terminate early (robot doesn't fall)

### Step 4: Analyze Results

Compare the visualization output for each version:

| Metric | V1 Flipped | V2 No Mirror | V3 Original | Expected |
|--------|-----------|--------------|-------------|----------|
| Velocity | ? | ? | ? | >0.1 m/s |
| Height | ? | ? | ? | ~0.12-0.14m |
| Reward | ? | ? | ? | >-5.0 |
| Falls? | ? | ? | ? | No |
| Direction | ? | ? | ? | Forward (+X) |

## Known Issues and Fixes Applied

### Issue 1: Wrong Actuator Ordering ✅ FIXED
- **Problem**: Sunfounder order (FL, FR, RL, RR) ≠ MuJoCo order (RR, FR, RL, FL)
- **Fix**: Explicit reordering in all versions
- **Status**: ✅ Applied to all versions

### Issue 2: Wrong Action Normalization ✅ FIXED
- **Problem**: Normalizing for [0, π] instead of environment's [-π/2, π]
- **Fix**: Use correct servo range in normalization
- **Status**: ✅ Applied to all versions

### Issue 3: Forward Direction Flipped? 🤔 TESTING
- **Problem**: Robot might walk backward due to coordinate frame mismatch
- **Hypothesis**: Sunfounder's +Y (forward) maps to MuJoCo's -Y (backward)
- **Fix**: Negate Y coordinate before IK (V1)
- **Status**: ⏳ Needs testing

### Issue 4: Right-Side Mirroring Wrong? 🤔 TESTING
- **Problem**: MuJoCo legs are geometrically mirrored, may not need manual negation
- **Observation**: Left legs have `euler="0 3.1415 -1.57075"`, right have `euler="0 0 1.57075"`
- **Hypothesis**: Geometric mirroring means we shouldn't negate angles
- **Fix**: Remove angle negation for right side (V2)
- **Status**: ⏳ Needs testing

## Debugging Checklist

- [ ] Standing position test with mirroring works
- [ ] Standing position test without mirroring works
- [ ] Extracted all 3 gait versions
- [ ] Visualized V1 (flipped Y)
- [ ] Visualized V2 (no mirror)
- [ ] Visualized V3 (original)
- [ ] Identified which version works best
- [ ] Analyzed why the working version works
- [ ] Generated full demonstration set with working version

## Action Diagnostics

When visualizing, check the action statistics:

```
Per-Joint Action Statistics:
Joint          Mean      Std      Min      Max
--------------------------------------------------
FL_hip       -0.669    0.146   -1.000   -0.429    ← Front left
FL_knee      -0.287    0.046   -0.374   -0.163
FR_hip       -0.789    0.105   -1.000   -0.538    ← Front right
FR_knee      -0.315    0.067   -0.485   -0.202
RL_hip        0.002    0.147   -0.238    0.347    ← Rear left
RL_knee      -0.380    0.046   -0.504   -0.293
RR_hip        0.123    0.106   -0.129    0.347    ← Rear right
RR_knee      -0.352    0.067   -0.465   -0.181
```

**Red flags:**
- ❌ Front legs very negative, rear legs near zero (asymmetric)
- ❌ Left/right pairs very different (should be similar with opposite sign)
- ❌ Actions near -1.0 or +1.0 limits (might be clipping)

**Good signs:**
- ✓ Front/rear legs have similar ranges
- ✓ Left/right pairs symmetric (opposite signs)
- ✓ Actions well within [-0.8, 0.8] range (not clipping)

## Coordinate System Reference

### MuJoCo World Frame
- **X-axis (RED)**: Forward direction
- **Y-axis (GREEN)**: Right direction
- **Z-axis (BLUE)**: Up direction

### Robot Base
- Rotated: `euler="0.0 0.0 -1.57075"` (-90° around Z)
- This rotation might affect how Sunfounder's Y coordinate maps to MuJoCo's frame

### Sunfounder Leg Coordinates
- **Y**: Forward/backward position of foot relative to hip
- **Z**: Height of foot from ground
- Positive Y should be forward, but might need flipping

## Next Steps After Identifying Working Version

1. **Extract full dataset:**
```bash
python extract_sunfounder_demos_<WORKING_VERSION>.py \
    --n-cycles 50 \
    --standing-height 80 \
    --output-file demonstrations/sunfounder_working.pkl
```

2. **Verify it still works:**
```bash
python visualize_demonstrations.py \
    --demo-file demonstrations/sunfounder_working.pkl
```

3. **Train with behavioral cloning:**
```bash
python train_pretrain_finetune.py \
    --demo-file demonstrations/sunfounder_working.pkl \
    --total-timesteps 500000
```

## Advanced Debugging

If none of the versions work, try:

1. **Manually test specific joint angles:**
   - Modify `test_standing_position.py` to test specific angles
   - Verify which angles produce desired leg positions

2. **Compare with environment's neutral angles:**
   - Environment uses: `neutral_hip = -π/6`, `neutral_knee = -π/4`
   - See if IK calculations produce similar angles

3. **Test individual legs:**
   - Modify extraction to only move one leg at a time
   - Verify left/right and front/back legs work correctly

4. **Check joint limits:**
   - Verify angles are within servo range [-π/2, π]
   - Check if any angles are being clipped

## Questions to Answer

- [ ] Does the standing position test work?
- [ ] Which mirroring approach is correct?
- [ ] Does Y coordinate need flipping?
- [ ] Are the joint angles within valid ranges?
- [ ] Do the gaits match Sunfounder's original behavior?
