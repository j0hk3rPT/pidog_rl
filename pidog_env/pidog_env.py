"""PiDog Gymnasium Environment using MuJoCo."""

import numpy as np
import mujoco
from mujoco import viewer
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import os
from typing import Dict, Any


class PiDogEnv(gym.Env):
    """
    PiDog Quadruped Robot Environment for Reinforcement Learning.

    This environment simulates the PiDog quadruped robot using MuJoCo physics.
    The goal is to train the robot to walk forward while maintaining balance.

    Observation Space (Always Box, flattened):
        Single flat array containing:
        - Image pixels (84x84x3 = 21168 values, normalized to [0,1])
          - If use_camera=False, this part is zeros
        - Vector sensors (31 values):
          - Joint positions (8 leg joints)
          - Joint velocities (8 leg joints)
          - Body orientation (quaternion) from IMU
          - Body linear velocity from IMU
          - Body angular velocity from IMU gyroscope
          - Body height estimate
          - Ultrasonic distance sensor (HC-SR04)
          - IMU accelerometer (3-axis)
        Total shape: (21199,) = 21168 + 31

    Action Space:
        - Target positions for 8 leg joints (continuous)

    Rewards:
        - Forward velocity
        - Energy efficiency
        - Stability (penalize falling)
        - Height maintenance
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, xml_path=None, use_camera=True, camera_width=84, camera_height=84,
                 domain_randomization=True, curriculum_level=0):
        super().__init__()

        # Set up paths
        if xml_path is None:
            current_dir = Path(__file__).parent.parent
            self.xml_path = str(current_dir / "model" / "pidog.xml")
        else:
            self.xml_path = xml_path

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Domain randomization and curriculum learning
        self.domain_randomization = domain_randomization
        self.curriculum_level = curriculum_level  # 0=easiest, 3=hardest

        # Camera configuration for observations
        self.use_camera = use_camera
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_renderer = None
        if use_camera:
            self.camera_renderer = mujoco.Renderer(self.model, height=camera_height, width=camera_width)

        # Rendering for visualization
        self.render_mode = render_mode
        self.renderer = None
        self.viewer = None
        if render_mode == "human":
            # Launch interactive MuJoCo viewer
            self.viewer = viewer.launch_passive(self.model, self.data)
            print("MuJoCo viewer launched successfully")

        # Initialize sensor IDs
        self._init_sensor_ids()

        # Joint indices (8 leg joints for walking)
        # Format: [front_left, front_right, back_left, back_right]
        # Each leg has 2 DOF (shoulder, knee)
        self.n_joints = 8  # 4 legs × 2 joints per leg

        # Define observation and action spaces
        # Vector observation: joint pos (8) + joint vel (8) + quat (4) + lin_vel (3) +
        #                     ang_vel (3) + height (1) + ultrasonic (1) + accel (3) = 31
        self.vector_obs_dim = 8 + 8 + 4 + 3 + 3 + 1 + 1 + 3  # = 31
        self.image_obs_dim = camera_height * camera_width * 3  # Flattened image size

        # Always use flattened Box observation space
        # Image (flattened, normalized to [0,1]) + vector sensors
        # Total: 84*84*3 + 31 = 21,168 + 31 = 21,199
        # Note: Image part is zeros if use_camera=False
        obs_dim = self.image_obs_dim + self.vector_obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action: target joint positions for 8 leg joints
        # Normalized to [-1, 1], will be scaled to actual joint limits
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_joints,),
            dtype=np.float32
        )

        # Servo specifications (MG90S Metal Gear Servo)
        # Real hardware constraints for sim-to-real transfer
        self.servo_specs = {
            "range": (-np.pi/2, np.pi),       # Extended range to support negative angles
            "max_torque": 0.216,              # Nm (at 6V)
            "min_torque": 0.176,              # Nm (at 4.8V)
            "max_speed": 13.09,               # rad/s (750°/s at 6V)
            "min_speed": 10.47,               # rad/s (600°/s at 4.8V)
            "voltage_range": (4.8, 6.0),      # Operating voltage
        }

        # Joint limits (extended to support found neutral angles)
        # Hip neutral: -30° (-π/6), Knee neutral: -45° (-π/4)
        self.joint_limits = {
            "hip": (-np.pi/2, np.pi),         # -90° to 180° range
            "knee": (-np.pi/2, np.pi),        # -90° to 180° range
        }

        # Neutral standing angles (found by systematic search)
        self.neutral_hip = -np.pi / 6     # -30°
        self.neutral_knee = -np.pi / 4    # -45°

        # Previous action for velocity limiting
        self.prev_action = None

        # Simulation parameters
        self.dt = self.model.opt.timestep
        self.frame_skip = 5

        # Target velocity
        self.target_forward_velocity = 0.5  # m/s

        # Episode tracking
        self.step_count = 0
        self.max_steps = 5000  # 100 seconds at 50Hz - gives more time to learn walking

        # Velocity history for stationary detection
        # Track forward velocity over ~1 second (depends on frame_skip and dt)
        # With dt=0.002 and frame_skip=5, each step is 0.01s, so 100 steps = 1 second
        self.velocity_history_size = 100  # ~1 second of history
        self.velocity_history = []

        # Obstacle avoidance parameters
        self.obstacle_danger_distance = 0.5  # Start penalizing below 50cm
        self.obstacle_critical_distance = 0.2  # Heavy penalty below 20cm

        # Self-collision detection
        # Get geom IDs for leg collision detection
        self._init_leg_geom_ids()

    def _init_sensor_ids(self):
        """Initialize sensor IDs for efficient data access."""
        self.sensor_ids = {}

        # Find sensor indices by name
        sensor_names = ['ultrasonic', 'imu_accel', 'imu_gyro']

        for name in sensor_names:
            try:
                sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                sensor_adr = self.model.sensor_adr[sensor_id]
                sensor_dim = self.model.sensor_dim[sensor_id]
                self.sensor_ids[name] = {
                    'id': sensor_id,
                    'adr': sensor_adr,
                    'dim': sensor_dim
                }
            except Exception as e:
                print(f"Warning: Could not find sensor '{name}': {e}")
                # Set default values if sensor not found
                self.sensor_ids[name] = {'id': -1, 'adr': 0, 'dim': 0}

    def _init_leg_geom_ids(self):
        """Initialize geom IDs for leg self-collision detection."""
        self.leg_geom_ids = []

        # Find all geoms with "_leg_" in their name (leg collision geoms)
        # These include: leg_b_c0, leg_b_c1, leg_a_c0, leg_a_sc, etc.
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and '_leg_' in geom_name:
                self.leg_geom_ids.append(i)

    def _get_camera_obs(self) -> np.ndarray:
        """Get RGB camera observation from pidog_camera."""
        if not self.use_camera or self.camera_renderer is None:
            return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)

        # Update scene and render
        self.camera_renderer.update_scene(self.data, camera="pidog_camera")
        rgb_array = self.camera_renderer.render()

        return rgb_array

    def _get_sensor_data(self, sensor_name: str) -> np.ndarray:
        """Get sensor data by name."""
        if sensor_name not in self.sensor_ids or self.sensor_ids[sensor_name]['id'] == -1:
            # Return zeros if sensor not found
            default_dims = {'ultrasonic': 1, 'imu_accel': 3, 'imu_gyro': 3}
            return np.zeros(default_dims.get(sensor_name, 1), dtype=np.float32)

        sensor_info = self.sensor_ids[sensor_name]
        adr = sensor_info['adr']
        dim = sensor_info['dim']

        return self.data.sensordata[adr:adr+dim].copy()

    def _detect_leg_collisions(self) -> int:
        """
        Detect self-collisions between legs.

        Returns:
            Number of leg-to-leg collision contacts
        """
        leg_collision_count = 0

        # Check all active contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            geom1 = contact.geom1
            geom2 = contact.geom2

            # Check if both geoms are leg geoms (self-collision)
            if geom1 in self.leg_geom_ids and geom2 in self.leg_geom_ids:
                # Get the body IDs to ensure they're from different legs
                body1 = self.model.geom_bodyid[geom1]
                body2 = self.model.geom_bodyid[geom2]

                # Only count if bodies are different (different legs colliding)
                if body1 != body2:
                    leg_collision_count += 1

        return leg_collision_count

    def _detect_belly_touch(self) -> bool:
        """
        Detect if belly/chest touches the ground.

        Returns:
            True if belly is touching ground, False otherwise
        """
        # Get ground geom ID
        ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ground')

        # Body geom names that indicate belly/chest contact
        belly_geom_names = ['chest_c0', 'chest_c1', 'chest_c2',
                            'torso_left_c0', 'torso_left_c1', 'torso_left_c2',
                            'torso_right_c0', 'torso_right_c1', 'torso_right_c2']

        # Get belly geom IDs
        belly_geom_ids = []
        for name in belly_geom_names:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:
                belly_geom_ids.append(geom_id)

        # Check all active contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Check if belly geom is touching ground
            if (geom1 == ground_id and geom2 in belly_geom_ids) or \
               (geom2 == ground_id and geom1 in belly_geom_ids):
                return True

        return False

    def _detect_torso_fall(self) -> bool:
        """
        Detect actual fall: torso/body touching ground.

        Simple and reliable: If torso touches ground = fall.
        No complex force calculations needed.

        Returns:
            True if fallen, False otherwise
        """
        # Get ground geom ID
        ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ground')

        # Torso/body geom names that should NOT touch ground (indicates fall)
        torso_geom_names = [
            'chest_c0', 'chest_c1', 'chest_c2',
            'torso_left_c0', 'torso_left_c1', 'torso_left_c2',
            'torso_right_c0', 'torso_right_c1', 'torso_right_c2',
            'body_c0', 'body_c1', 'body_c2',  # Main body collision geoms
        ]

        # Get torso geom IDs
        torso_geom_ids = []
        for name in torso_geom_names:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:
                torso_geom_ids.append(geom_id)

        # Check all active contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Check if torso geom is touching ground
            if (geom1 == ground_id and geom2 in torso_geom_ids) or \
               (geom2 == ground_id and geom1 in torso_geom_ids):
                # Torso touching ground = FALL
                return True

        return False

    def _get_obs(self):
        """Get current observation with camera and all sensors.

        Returns:
            np.ndarray: Flattened observation array of shape (21199,)
                       First 21168 values are image pixels (normalized to [0,1], zeros if no camera)
                       Last 31 values are sensor readings
        """
        # Joint positions (first 8 actuated joints)
        joint_pos = self.data.qpos[7:15].copy()  # Skip free joint (first 7 DOF)

        # Joint velocities
        joint_vel = self.data.qvel[6:14].copy()  # Skip free joint (first 6 DOF)

        # Body orientation (quaternion) - in real robot, from IMU sensor fusion
        body_quat = self.data.qpos[3:7].copy()

        # Body linear velocity - in real robot, from IMU integration or visual odometry
        body_lin_vel = self.data.qvel[0:3].copy()

        # Body angular velocity - in real robot, from IMU gyroscope
        body_ang_vel = self.data.qvel[3:6].copy()

        # Body height (z-position) - in real robot, estimated from kinematics or ultrasonic
        body_height = np.array([self.data.qpos[2]])

        # Sensor readings
        ultrasonic_dist = self._get_sensor_data('ultrasonic')  # [1] - distance in meters
        imu_accel = self._get_sensor_data('imu_accel')  # [3] - ax, ay, az in m/s²
        # imu_gyro already captured in body_ang_vel

        # Concatenate vector observations
        # Total: 8 + 8 + 4 + 3 + 3 + 1 + 1 + 3 = 31 dimensions
        vector_obs = np.concatenate([
            joint_pos,        # [0:8]
            joint_vel,        # [8:16]
            body_quat,        # [16:20]
            body_lin_vel,     # [20:23]
            body_ang_vel,     # [23:26]
            body_height,      # [26]
            ultrasonic_dist,  # [27]
            imu_accel,        # [28:31]
        ]).astype(np.float32)

        # Get camera image (or zeros if camera disabled)
        camera_obs = self._get_camera_obs()

        # Always return flattened Box observation
        # Flatten image and concatenate with vector, all as float32
        image_flat = camera_obs.flatten().astype(np.float32) / 255.0  # Normalize to [0, 1]
        flattened_obs = np.concatenate([image_flat, vector_obs])
        return flattened_obs

    def _scale_action(self, action):
        """
        Scale normalized action [-1, 1] to actual joint limits with servo constraints.

        Applies realistic servo limitations:
        - Range: -90° to 180° (-π/2 to π radians)
        - Max speed: 13.09 rad/s (750°/s at 6V)
        """
        scaled_action = np.zeros_like(action)

        # For each joint, scale to its limits
        for i in range(self.n_joints):
            low, high = self.servo_specs["range"]
            # Map from [-1, 1] to [-π/2, π]
            scaled_action[i] = low + (action[i] + 1.0) * 0.5 * (high - low)

        # Apply velocity limiting (simulate servo speed constraints)
        if self.prev_action is not None:
            max_delta = self.servo_specs["max_speed"] * self.dt * self.frame_skip
            delta = scaled_action - self.prev_action

            # Clip to max speed
            delta = np.clip(delta, -max_delta, max_delta)
            scaled_action = self.prev_action + delta

        # Store for next step
        self.prev_action = scaled_action.copy()

        return scaled_action

    def _compute_reward(self):
        """
        Compute reward based on current state.

        REWARD STRUCTURE (varies by curriculum):
        - Level -1 (Standing): Focus on balance, no forward motion required
        - Level 0+ (Walking): Progressive forward velocity requirements

        Components:
        1. Forward velocity / Standing reward - curriculum dependent
        2. Obstacle avoidance using ultrasonic - critical for safety
        3. Upright stability - important for not falling
        4. Fall penalty - ONLY for actual falls (torso touching ground)
        5. Stationary penalty (disabled for standing mode)
        6. Energy efficiency - smooth movements
        7. Lateral stability - walk straight
        8. Leg collision penalty - avoid unnatural gaits
        """
        forward_vel = self.data.qvel[0]

        # ============= 1. FORWARD VELOCITY / STANDING REWARD =============
        if self.curriculum_level < 0:
            # STANDING MODE: Reward staying still and balanced
            # Penalize movement, reward stillness
            velocity_reward = -2.0 * abs(forward_vel)  # Penalize any movement

            # Bonus for staying nearly still
            if abs(forward_vel) < 0.05:  # Less than 5 cm/s
                velocity_reward += 2.0  # Big reward for standing still
        else:
            # WALKING MODE: Reward forward velocity
            velocity_reward = forward_vel

            # Bonus for reaching target speed (adjusted by curriculum)
            target_vel = self.target_forward_velocity * (1.0 + 0.2 * self.curriculum_level)
            if forward_vel >= target_vel:
                velocity_reward += 1.0


        # ============= 2. OBSTACLE AVOIDANCE (ULTRASONIC SENSOR) =============
        ultrasonic_dist = self._get_sensor_data('ultrasonic')[0]

        obstacle_penalty = 0.0
        if ultrasonic_dist > 0:  # Valid reading (sensor returns -1 if no obstacle)
            if ultrasonic_dist < self.obstacle_critical_distance:  # < 20cm
                # CRITICAL: Very close to obstacle - heavy penalty
                obstacle_penalty = -10.0 * (self.obstacle_critical_distance - ultrasonic_dist)
            elif ultrasonic_dist < self.obstacle_danger_distance:  # < 50cm
                # WARNING: Getting close to obstacle - moderate penalty
                obstacle_penalty = -2.0 * (self.obstacle_danger_distance - ultrasonic_dist)


        # ============= 3. UPRIGHT STABILITY =============
        body_quat = self.data.qpos[3:7]  # MuJoCo format: [w, x, y, z]
        # w component (1 = upright, 0 = sideways/tilted)
        upright_reward = 2.0 * (body_quat[0] - 0.7)  # Bonus for staying upright


        # ============= 4. FALL PENALTY (ONLY FOR ACTUAL FALLS) =============
        # Check for actual fall: torso touching ground
        is_fallen = self._detect_torso_fall()
        fall_penalty = 0.0
        if is_fallen:
            # Large penalty for actual fall
            fall_penalty = -20.0
            # Store fall state for termination check
            self._has_fallen = True
        else:
            self._has_fallen = False


        # ============= 5. STATIONARY PENALTY (DISABLED IN STANDING MODE) =============
        # Track velocity history to detect if robot is stuck
        self.velocity_history.append(abs(forward_vel))
        if len(self.velocity_history) > self.velocity_history_size:
            self.velocity_history.pop(0)  # Remove oldest

        stationary_penalty = 0.0
        # Only penalize being stationary in walking mode (curriculum >= 0)
        if self.curriculum_level >= 0 and len(self.velocity_history) >= self.velocity_history_size:
            # Check average velocity over last ~1 second
            avg_velocity = np.mean(self.velocity_history)
            if avg_velocity < 0.05:  # Moving less than 5 cm/s for 1 second
                stationary_penalty = -5.0  # Heavy penalty for being stuck


        # ============= 6. ENERGY EFFICIENCY =============
        # Encourage smooth, efficient movements
        action_penalty = -0.005 * np.sum(np.square(self.data.ctrl))


        # ============= 7. LATERAL STABILITY =============
        # Penalize sideways drift - should walk straight
        lateral_vel = abs(self.data.qvel[1])
        lateral_penalty = -0.5 * lateral_vel


        # ============= 8. LEG SELF-COLLISION PENALTY =============
        # Penalize when legs collide with each other (unnatural gait)
        leg_collisions = self._detect_leg_collisions()
        collision_penalty = -2.0 * leg_collisions  # -2.0 per collision


        # ============= 9. SURVIVAL BONUS =============
        # Reward for staying alive and not falling
        # At 50Hz, this adds +5.0 per second of stable operation
        survival_bonus = 0.1


        # ============= 10. GAIT QUALITY / STANDING POSTURE REWARD =============
        joint_velocities = self.data.qvel[6:14]  # 8 leg joints
        leg_vel_variance = np.var(joint_velocities)

        if self.curriculum_level < 0:
            # STANDING MODE: Reward stable, neutral posture
            # Low joint velocities = good standing
            gait_quality = 0.0
            if leg_vel_variance < 0.2:  # Very still = good
                gait_quality = 2.0

            # Bonus: reward being near neutral joint positions
            joint_pos = self.data.qpos[7:15]
            neutral_positions = np.array([self.neutral_hip, self.neutral_knee] * 4)
            position_error = np.mean(np.abs(joint_pos - neutral_positions))
            posture_reward = max(0, 1.0 - position_error * 2.0)  # 0-1 based on error
        else:
            # WALKING MODE: Reward rhythmic leg movement
            # Good gait has moderate variance (legs moving at different speeds rhythmically)
            # Too low = frozen/stationary, too high = flailing/chaotic
            gait_quality = 0.0
            if 0.5 < leg_vel_variance < 3.0:  # Sweet spot for coordinated movement
                gait_quality = 1.0
            elif 0.2 < leg_vel_variance < 5.0:  # Acceptable range
                gait_quality = 0.5
            posture_reward = 0.0


        # ============= 11. HEIGHT MAINTENANCE (ESPECIALLY FOR STANDING) =============
        body_height = self.data.qpos[2]
        target_height = 0.14  # Target standing height
        height_error = abs(body_height - target_height)

        if self.curriculum_level < 0:
            # STANDING MODE: Strong reward for maintaining height
            height_reward = max(0, 2.0 - height_error * 20.0)  # 0-2 based on error
        else:
            # WALKING MODE: Moderate reward for height
            height_reward = max(0, 1.0 - height_error * 10.0)  # 0-1 based on error


        # ============= COMBINE ALL REWARDS =============
        reward = (
            2.0 * velocity_reward +      # Forward speed / standing still
            1.0 * obstacle_penalty +     # Avoid obstacles
            2.0 * upright_reward +       # Stay balanced
            1.0 * fall_penalty +         # Penalize actual falls
            1.0 * stationary_penalty +   # Don't get stuck (disabled in standing)
            1.0 * survival_bonus +       # Reward staying alive
            1.0 * gait_quality +         # Gait quality / standing stillness
            1.0 * posture_reward +       # Neutral posture (standing only)
            1.0 * height_reward +        # Height maintenance
            action_penalty +             # Be efficient
            lateral_penalty +            # Walk straight / no lateral drift
            collision_penalty            # Avoid leg collisions
        )

        return reward

    def _is_terminated(self):
        """
        Check if episode should terminate (fall detection).

        NEW APPROACH: Only terminate on ACTUAL FALLS (torso touching ground).
        Simple contact detection - if torso hits ground, it's a fall.

        This allows the robot to:
        - Recover from tilted positions (no early termination)
        - Learn from near-fall situations
        - Develop robust behaviors through curriculum learning
        """
        # Only terminate if we detected an actual fall (torso touching ground)
        # Fall detection is done in _compute_reward() and stored in self._has_fallen
        if hasattr(self, '_has_fallen') and self._has_fallen:
            return True

        return False

    def _is_truncated(self):
        """Check if episode should be truncated (max steps)."""
        return self.step_count >= self.max_steps

    def _apply_domain_randomization(self):
        """
        Apply domain randomization to physics parameters.

        Randomizes:
        - Body mass (±20%)
        - Ground friction (±30%)
        - Joint damping (±25%)
        - Actuator gains (±15%)
        - Initial body height (±10%)

        This helps sim-to-real transfer by training on diverse physics.
        """
        if not self.domain_randomization:
            return

        # 1. Randomize body mass (±20%)
        # Find body ID for main body
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'body')
            original_mass = 0.5  # kg (default PiDog mass)
            self.model.body_mass[body_id] = original_mass * np.random.uniform(0.8, 1.2)
        except:
            pass  # Body not found, skip

        # 2. Randomize ground friction (±30%)
        try:
            ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ground')
            original_friction = 1.0
            self.model.geom_friction[ground_id, 0] = original_friction * np.random.uniform(0.7, 1.3)
        except:
            pass

        # 3. Randomize joint damping (±25%)
        # Apply to all actuated joints
        for i in range(8):  # 8 leg joints
            original_damping = 0.1
            self.model.dof_damping[i + 6] = original_damping * np.random.uniform(0.75, 1.25)

        # 4. Randomize actuator gains (±15%)
        # This simulates servo performance variation
        for i in range(self.model.nu):
            # Store original gain if not already stored
            if not hasattr(self, '_original_actuator_gain'):
                self._original_actuator_gain = self.model.actuator_gainprm.copy()

            gain_variation = np.random.uniform(0.85, 1.15)
            self.model.actuator_gainprm[i, 0] = self._original_actuator_gain[i, 0] * gain_variation

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)

        # Add small random perturbation to initial state
        if seed is not None:
            np.random.seed(seed)

        # Apply domain randomization
        self._apply_domain_randomization()

        # Randomize initial joint positions slightly around neutral
        # Use correct neutral angles: hip=-30°, knee=-45°
        # Variation increases with curriculum level
        joint_noise = 0.1 * (1.0 + 0.5 * self.curriculum_level)
        for i in range(4):  # 4 legs
            # Hip joints (even indices)
            self.data.qpos[7 + i*2] = self.neutral_hip + np.random.uniform(-joint_noise, joint_noise)
            # Knee joints (odd indices)
            self.data.qpos[7 + i*2 + 1] = self.neutral_knee + np.random.uniform(-joint_noise, joint_noise)

        # Set body to target height with randomization (±10%)
        # Higher curriculum = more variation
        height_base = 0.14
        height_variation = 0.02 * (1.0 + 0.5 * self.curriculum_level)
        self.data.qpos[2] = height_base + np.random.uniform(-height_variation, height_variation)

        # Randomize initial orientation slightly (curriculum-dependent)
        if self.curriculum_level >= 1:
            # Add small initial tilt for higher difficulty
            angle_noise = 0.05 * self.curriculum_level  # Up to 0.15 rad (~8.5°) at level 3
            self.data.qpos[3:7] = np.array([
                1.0,  # w
                np.random.uniform(-angle_noise, angle_noise),  # x (roll)
                np.random.uniform(-angle_noise, angle_noise),  # y (pitch)
                0.0   # z (yaw)
            ])
            # Normalize quaternion
            self.data.qpos[3:7] /= np.linalg.norm(self.data.qpos[3:7])

        # Forward the simulation to settle
        mujoco.mj_forward(self.model, self.data)

        # Reset action tracking for velocity limiting
        self.prev_action = None

        # Reset velocity history for stationary detection
        self.velocity_history = []

        # Reset fall flag
        self._has_fallen = False

        self.step_count = 0

        obs = self._get_obs()
        info = {
            'curriculum_level': self.curriculum_level,
            'domain_randomization': self.domain_randomization
        }

        return obs, info

    def step(self, action):
        """Execute one step in the environment."""
        # Scale action to joint limits
        scaled_action = self._scale_action(action)

        # Set control
        self.data.ctrl[:self.n_joints] = scaled_action

        # Step simulation (with frame skip)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

            # Enforce realistic servo velocity limits (13.09 rad/s for MG90S servos)
            # Clamp joint velocities to hardware specifications
            joint_velocities = self.data.qvel[6:14]  # 8 leg joints
            np.clip(joint_velocities, -self.servo_specs['max_speed'],
                   self.servo_specs['max_speed'], out=joint_velocities)

        # Get observation
        obs = self._get_obs()

        # Compute reward
        reward = self._compute_reward()

        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        # Additional info
        info = {
            "forward_velocity": self.data.qvel[0],
            "body_height": self.data.qpos[2],
            "step": self.step_count,
            "has_fallen": self._has_fallen if hasattr(self, '_has_fallen') else False,
            "curriculum_level": self.curriculum_level,
        }

        self.step_count += 1

        # Render if needed
        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            # Create renderer if not exists
            if self.renderer is None:
                from mujoco import Renderer
                self.renderer = Renderer(self.model, height=480, width=640)

            self.renderer.update_scene(self.data)
            return self.renderer.render()

        return None

    def close(self):
        """Clean up resources."""
        # Close viewer - handle gracefully as it may already be closed
        if self.viewer is not None:
            try:
                self.viewer.close()
            except:
                pass  # Viewer may already be closed
            self.viewer = None

        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

        if self.camera_renderer is not None:
            self.camera_renderer.close()
            self.camera_renderer = None
