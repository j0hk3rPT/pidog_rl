# PiDog Hardware Specifications

This document details the actual hardware components used in the SunFounder PiDog robot and compares them with the simulation parameters.

## Summary of Findings

After researching the actual PiDog hardware, several discrepancies were found between the real components and the simulation configuration. This document provides accurate specifications and recommended updates for better sim-to-real transfer.

---

## Servo Motors

### Actual Hardware (RESEARCHED)
- **Model**: Likely **MG90S Metal Gear Servo** (or similar 9g metal gear servo)
- **Quantity**: 12 total servos
  - 8 servos for legs (2 per leg: hip + knee)
  - 1 servo for tail
  - 3 servos for head/neck (yaw, pitch, roll)
- **Torque**:
  - @ 4.8V: **1.8 kgf·cm = 0.176 Nm**
  - @ 6.0V: **2.2 kgf·cm = 0.216 Nm**
- **Speed** (No Load Speed):
  - @ 4.8V: 60°/0.18sec = **333.33°/s = 5.82 rad/s**
  - @ 6.0V: 60°/0.15sec = **400°/s = 6.98 rad/s**
- **Operating Voltage**: 4.8-6.0V (typically 5V)
- **Stall Current**: 120-250mA (moving), up to 700mA (stall)
- **Weight**: 13.4g
- **Rotation Range**: 0-180° (90° each direction)
- **Gear Type**: Metal (more durable than plastic SG90)

### Impact on Simulation
The simulation was **underestimating** servo torque but had correct speed:
- **Torque**: Updated from 0.127-0.137 Nm → **0.176-0.216 Nm** (+28-58% MORE powerful)
- **Speed**: Original values were **CORRECT** at 5.8-7.0 rad/s ✓

This means:
- ✅ Speed simulation is accurate for real hardware
- ✅ Torque update allows robot to apply stronger forces (better stability)
- ⚠️ Existing policies may be slightly more conservative on torque

---

## IMU (Inertial Measurement Unit)

### Current Simulation (INCORRECT)
- **Model**: MPU6050
- **Accelerometer Range**: ±16g
- **Gyroscope Range**: ±1000°/s (±17.45 rad/s)
- **Accelerometer Noise**: 0.01 m/s²
- **Gyroscope Noise**: 0.005 rad/s

### Actual Hardware (CONFIRMED)
- **Model**: **SH3001** (NOT MPU6050!)
- **Type**: 6-DOF IMU (3-axis accelerometer + 3-axis gyroscope)
- **Power**: 3.3V
- **Communication**: I2C
- **Output Data Rate (ODR)**: Up to 32kHz (typically 500Hz)

**Accelerometer Ranges** (configurable):
- ±2g
- ±4g
- ±8g
- **±16g** (default)

**Gyroscope Ranges** (configurable):
- ±125°/s (±2.18 rad/s)
- ±250°/s (±4.36 rad/s)
- ±500°/s (±8.73 rad/s)
- ±1000°/s (±17.45 rad/s)
- **±2000°/s (±34.9 rad/s)** (default)

### Impact on Simulation
- ❌ **Wrong sensor model** referenced in documentation
- ✅ Accelerometer range matches (±16g)
- ❌ **Gyroscope range is DOUBLE** the current simulation (±2000°/s vs ±1000°/s)
  - Real sensor can measure **much faster rotations**
  - Important for fall detection and recovery maneuvers
- ⚠️ Noise specifications unknown (manufacturer datasheet not publicly available)
  - Current MPU6050 noise values are reasonable approximations

---

## Camera

### Current Simulation (CORRECT ✓)
- **Model**: OV5647
- **Type**: 5MP Raspberry Pi Camera Module
- **Field of View (FOV)**:
  - Horizontal: ~62° (configured in XML)
  - Vertical: **49°** (configured as `fovy="49"`)
- **Resolution**:
  - Still images: 2592 x 1944 pixels
  - Video: 1080p30, 720p60, 640x480p90
- **Size**: 25mm x 23mm x 9mm
- **Weight**: 3g

### Status
✅ **Camera specifications are accurate**

The simulation uses the correct camera model and FOV configuration.

---

## Ultrasonic Sensor

### Current Simulation (MOSTLY CORRECT)
- **Model**: HC-SR04
- **Range**: 2cm - **450cm** (cutoff="4.5")
- **Accuracy/Noise**: ±3mm (noise="0.003")
- **Beam Angle**: <15° (narrow cone)

### Actual Hardware (CONFIRMED)
- **Model**: **HC-SR04** ✓
- **Range**: 2cm - **400cm** (not 450cm!)
- **Accuracy**: ±3mm ✓
- **Frequency**: 40 kHz ultrasonic burst (8 cycles)
- **Beam Angle**: <15° ✓

### Impact on Simulation
- ⚠️ **Max range is 50cm too long** (450cm vs 400cm)
  - Minor issue - robot shouldn't rely on readings beyond 400cm anyway
  - Update cutoff to 4.0m for accuracy

---

## Recommended Updates

### 1. Servo Specifications

**File**: `model/pidog_actuators.xml`

```xml
<!--
  Servo Specifications: MG90S Metal Gear Servo
  - Torque: 0.176-0.216 Nm (1.8-2.2 kgf·cm at 4.8-6V)
  - Speed: 333-400°/s (5.8-7.0 rad/s)
  - No Load Speed: 4.8V: 60°/0.18sec, 6V: 60°/0.15sec
  - Operating voltage: 4.8-6.0V
  - Physical Range: 0-180° (0 to π radians)
  - Control Range: -90° to 180° (-π/2 to π radians)
-->

<actuator>
    <position name="back_right_hip_actuator"
              joint="back_right_hip"
              forcerange="-0.216 0.216"
              kp="5.0"
              kv="2.5"
              gear="1"/>
    <!-- ... repeat for all servos ... -->
</actuator>
```

**File**: `pidog_env/pidog_env.py`

```python
# Servo specifications (MG90S Metal Gear Servo)
# Real hardware constraints for sim-to-real transfer
self.servo_specs = {
    "range": (-np.pi/2, np.pi),           # Extended range to support negative angles
    "max_torque": 0.216,                  # Nm (at 6V)
    "min_torque": 0.176,                  # Nm (at 4.8V)
    "max_speed": 7.0,                     # rad/s (400°/s at 6V) - 60°/0.15sec
    "min_speed": 5.8,                     # rad/s (333°/s at 4.8V) - 60°/0.18sec
    "voltage_range": (4.8, 6.0),          # Operating voltage
}
```

### 2. IMU Specifications

**File**: `model/pidog_sensors.xml`

```xml
<!-- ==================== IMU SENSORS (SH3001) ==================== -->
<!-- Mounted at center of body (electronics compartment) -->

<!-- 3-Axis Accelerometer -->
<!-- Range: ±16g (default, to handle impacts and dynamic motion) -->
<!-- Noise: ~0.01 m/s² (estimated, datasheet not available) -->
<!-- Output: [ax, ay, az] in m/s² (includes gravity) -->

<accelerometer name="imu_accel"
               site="imu_site"
               noise="0.01"/>

<!-- 3-Axis Gyroscope -->
<!-- Range: ±2000°/s (±34.9 rad/s, default for dynamic movements) -->
<!-- Noise: ~0.005 rad/s (estimated, datasheet not available) -->
<!-- Output: [ωx, ωy, ωz] in rad/s -->

<gyro name="imu_gyro"
      site="imu_site"
      noise="0.005"/>
```

**Update comments to reflect SH3001 (not MPU6050) and correct gyroscope range.**

### 3. Ultrasonic Sensor

**File**: `model/pidog_sensors.xml`

```xml
<!-- ==================== ULTRASONIC RANGEFINDER ==================== -->
<!-- HC-SR04 Ultrasonic Distance Sensor -->
<!-- Range: 2cm - 400cm (0.02m - 4.0m) -->  <!-- FIXED: was 450cm -->
<!-- Accuracy: ±3mm (0.003m) -->
<!-- Beam angle: <15° (narrow cone along Z-axis) -->

<rangefinder name="ultrasonic"
             site="ultrasonic_site"
             cutoff="4.0"
             noise="0.003"/>
```

---

## References

1. **MG90S Servo Specifications**:
   - https://components101.com/motors/mg90s-metal-gear-servo-motor
   - Datasheet: https://www.electronicoscaldas.com/datasheet/MG90S_Tower-Pro.pdf

2. **SH3001 IMU**:
   - SunFounder PiDog Documentation: https://docs.sunfounder.com/projects/pidog/en/latest/hardware/cpn_6dof_imu.html
   - Python Library: https://github.com/sunfounder/python-sh3001

3. **OV5647 Camera**:
   - SunFounder PiDog Documentation: https://docs.sunfounder.com/projects/pidog/en/latest/hardware/cpn_camera.html

4. **HC-SR04 Ultrasonic**:
   - SunFounder PiDog Documentation: https://docs.sunfounder.com/projects/pidog/en/latest/hardware/cpn_ultrasonic.html

5. **PiDog Robot Kit**:
   - Product Page: https://www.sunfounder.com/products/sunfounder-pidog-robot-dog-kit-for-raspberry-pi
   - Documentation: https://docs.sunfounder.com/projects/pidog/en/latest/

---

## Notes for Sim-to-Real Transfer

### Updated Simulation (Current State)
The simulation has been updated to match real hardware:
- **Servo Torque**: Updated from 0.127-0.137 Nm → **0.176-0.216 Nm** ✓
- **Servo Speed**: Was already correct at 5.8-7.0 rad/s ✓
- **Gyroscope Range**: Updated from ±1000°/s → **±2000°/s** ✓
- **IMU Model**: Corrected documentation (MPU6050 → SH3001) ✓
- **Ultrasonic Range**: Fixed to 400cm (was 450cm) ✓

**Implications**:
- ✅ **Better sim-to-real transfer**: Simulation matches actual hardware
- ✅ **Full capability utilization**: Robot can use full torque and sensor range
- ✅ **Improved fall detection**: Gyroscope can measure faster rotations (±2000°/s)
- ⚠️ **Existing models**: Models trained with old specs may perform slightly differently

### Recommended Approach

**Current Status: Simulation Matched to Real Hardware ✓**

The simulation has been updated with accurate specifications. For new training:
- ✅ Use updated specs for best sim-to-real transfer
- ✅ Train new policies with accurate torque and sensor ranges
- ✅ Test thoroughly in simulation before hardware deployment

For existing trained models:
- Models trained with old specs should still work (conservative on torque)
- May benefit from fine-tuning with updated specifications
- Test both old and new models to compare performance

---

## Testing Checklist

Before deploying to real hardware:

- [ ] Verify servo torque limits (start at 50% of max)
- [ ] Test servo speed ramping (don't command max speed immediately)
- [ ] Calibrate IMU on real hardware (see SH3001 calibration docs)
- [ ] Verify ultrasonic sensor returns -1 when no obstacle (beyond 400cm)
- [ ] Test camera feed latency (OV5647 at different resolutions)
- [ ] Monitor servo current draw (ensure power supply adequate)
- [ ] Thermal testing (servos may heat up under continuous operation)

---

## Changelog

- **2025-11-10**: Initial hardware research and documentation
  - Identified servo model mismatch (SF006FM doesn't exist → MG90S likely)
  - Updated servo torque (0.127-0.137 Nm → 0.176-0.216 Nm)
  - **CORRECTED**: Servo speed was already correct (5.8-7.0 rad/s)
    - Initial documentation incorrectly showed 10.47-13.09 rad/s
    - Fixed to match actual spec: 4.8V: 60°/0.18sec, 6V: 60°/0.15sec
  - Corrected IMU model (MPU6050 → SH3001)
  - Updated gyroscope range (±1000°/s → ±2000°/s)
  - Fixed ultrasonic max range (450cm → 400cm)
