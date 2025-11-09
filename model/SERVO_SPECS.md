# PiDog Servo Specifications

## Hardware: SunFounder SF006FM 9g Digital Servo

The PiDog robot uses 12 SunFounder SF006FM servos (8 for legs, 4 for neck/head).

### Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Torque** | 0.127-0.137 Nm | 1.3-1.4 kgf·cm at 4.8-6V |
| **Speed** | 333-400°/s | 5.8-7.0 rad/s |
| **Operating Voltage** | 4.8-6.0V | Optimal at 6V |
| **Range** | 0-180° | 0 to π radians |
| **Neutral Position** | 90° | π/2 radians |
| **Type** | Digital Servo | Position control |

### Implementation in MuJoCo

The servo constraints are implemented in:

1. **`pidog_actuators.xml`**: MuJoCo actuator definitions
   - `ctrlrange`: 0 to π radians (0-180°)
   - `forcerange`: ±0.137 Nm (max torque)
   - `kp`: 5.0 (position gain)

2. **`pidog_env/pidog_env.py`**: Gymnasium environment
   - Velocity limiting: max 7.0 rad/s
   - Position limiting: 0 to π radians
   - Action scaling from [-1, 1] to [0, π]

### Joint Mapping

#### Leg Servos (8 total)
- **Back Right**: hip, knee
- **Front Right**: hip, knee
- **Back Left**: hip, knee
- **Front Left**: hip, knee

#### Control Order
Actions are provided in this order:
1. Back right hip
2. Back right knee
3. Front right hip
4. Front right knee
5. Back left hip
6. Back left knee
7. Front left hip
8. Front left knee

### Realistic Movement Constraints

For sim-to-real transfer, the following constraints are enforced:

1. **Position Limits**: 0° to 180° (0 to π radians)
   - Neutral/home position: 90° (π/2)
   - Full range available in both directions

2. **Velocity Limits**: Maximum 7.0 rad/s (400°/s)
   - Implemented via velocity clipping in environment
   - Prevents unrealistic instantaneous movements

3. **Torque Limits**: Maximum 0.137 Nm
   - Enforced in MuJoCo actuator definitions
   - Realistic force constraints

### Example Gaits

#### Trotting Gait
Diagonal legs move together:
- Back-right + Front-left (pair 1)
- Front-right + Back-left (pair 2)
- Frequency: 1.5 Hz
- Amplitude: ±30° hip, ±45° knee

#### Walking Gait
Sequential leg movement:
- One leg at a time
- More stable, slower
- Frequency: 1.0 Hz
- Amplitude: ±25° hip, ±40° knee

### Testing

Run visualization tests:

```bash
# Test basic visualization
uv run python tests/sit.py

# Test walking with realistic servos
uv run python tests/test_walk.py
```

### References

- [SunFounder PiDog Product Page](https://www.sunfounder.com/products/pidog)
- [SunFounder PiDog Documentation](https://docs.sunfounder.com/projects/pidog/)
- [SunFounder PiDog GitHub](https://github.com/sunfounder/pidog)
