"""PiDog Gymnasium Environment using MuJoCo."""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import os


class PiDogEnv(gym.Env):
    """
    PiDog Quadruped Robot Environment for Reinforcement Learning.

    This environment simulates the PiDog quadruped robot using MuJoCo physics.
    The goal is to train the robot to walk forward while maintaining balance.

    Observation Space:
        - Joint positions (8 leg joints + 3 neck joints)
        - Joint velocities (8 leg joints + 3 neck joints)
        - Body orientation (quaternion)
        - Body linear velocity
        - Body angular velocity

    Action Space:
        - Target positions for 8 leg joints (continuous)

    Rewards:
        - Forward velocity
        - Energy efficiency
        - Stability (penalize falling)
        - Height maintenance
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, xml_path=None):
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

        # Rendering
        self.render_mode = render_mode
        self.renderer = None
        if render_mode == "human":
            self.renderer = mujoco.viewer.launch_passive(self.model, self.data)

        # Joint indices (8 leg joints for walking)
        # Format: [front_left, front_right, back_left, back_right]
        # Each leg has 2 DOF (shoulder, knee)
        self.n_joints = 8  # 4 legs × 2 joints per leg

        # Define observation and action spaces
        # Observation: joint pos (8) + joint vel (8) + quat (4) + lin_vel (3) + ang_vel (3) + height (1)
        obs_dim = 8 + 8 + 4 + 3 + 3 + 1  # = 27
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

        # Servo specifications (SunFounder SF006FM 9g Digital Servo)
        # Real hardware constraints for sim-to-real transfer
        self.servo_specs = {
            "range": (-np.pi/2, np.pi),       # Extended range to support negative angles
            "max_torque": 0.137,              # Nm (at 6V)
            "min_torque": 0.127,              # Nm (at 4.8V)
            "max_speed": 7.0,                 # rad/s (400°/s)
            "min_speed": 5.8,                 # rad/s (333°/s)
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
        self.max_steps = 1000

    def _get_obs(self):
        """Get current observation."""
        # Joint positions (first 8 actuated joints)
        joint_pos = self.data.qpos[7:15]  # Skip free joint (first 7 DOF)

        # Joint velocities
        joint_vel = self.data.qvel[6:14]  # Skip free joint (first 6 DOF)

        # Body orientation (quaternion)
        body_quat = self.data.qpos[3:7]

        # Body linear velocity
        body_lin_vel = self.data.qvel[0:3]

        # Body angular velocity
        body_ang_vel = self.data.qvel[3:6]

        # Body height (z-position)
        body_height = np.array([self.data.qpos[2]])

        # Concatenate all observations
        obs = np.concatenate([
            joint_pos,
            joint_vel,
            body_quat,
            body_lin_vel,
            body_ang_vel,
            body_height
        ])

        return obs.astype(np.float32)

    def _scale_action(self, action):
        """
        Scale normalized action [-1, 1] to actual joint limits with servo constraints.

        Applies realistic servo limitations:
        - Range: -90° to 180° (-π/2 to π radians)
        - Max speed: 7.0 rad/s (400°/s)
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
        """Compute reward based on current state."""
        # Forward velocity reward - maximize forward speed!
        forward_vel = self.data.qvel[0]
        # Reward positive forward velocity, penalize backward
        velocity_reward = forward_vel

        # Bonus for reaching target speed
        if forward_vel >= self.target_forward_velocity:
            velocity_reward += 1.0

        # Height reward (penalize if body is too low/high)
        body_height = self.data.qpos[2]
        target_height = 0.14  # Target height in meters (based on neutral standing height)
        height_penalty = -2.0 * abs(body_height - target_height)

        # Stability reward (penalize large roll/pitch)
        body_quat = self.data.qpos[3:7]
        # w component (1 = upright, 0 = sideways)
        upright_reward = 2.0 * (body_quat[3] - 0.7)  # Bonus for staying upright

        # Energy penalty (encourage efficiency, penalize excessive movement)
        action_penalty = -0.005 * np.sum(np.square(self.data.ctrl))

        # Lateral movement penalty (penalize sideways drift)
        lateral_vel = abs(self.data.qvel[1])
        lateral_penalty = -0.5 * lateral_vel

        # Combine rewards
        reward = (
            3.0 * velocity_reward +      # Prioritize forward speed
            1.0 * height_penalty +
            1.5 * upright_reward +
            action_penalty +
            lateral_penalty
        )

        return reward

    def _is_terminated(self):
        """Check if episode should terminate (fall detection)."""
        body_height = self.data.qpos[2]
        body_quat = self.data.qpos[3:7]

        # Terminate if robot falls (too low or flipped)
        if body_height < 0.05:  # Body touching ground
            return True

        if body_quat[3] < 0.5:  # Severely tilted
            return True

        return False

    def _is_truncated(self):
        """Check if episode should be truncated (max steps)."""
        return self.step_count >= self.max_steps

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)

        # Add small random perturbation to initial state
        if seed is not None:
            np.random.seed(seed)

        # Randomize initial joint positions slightly around neutral
        # Use correct neutral angles: hip=-30°, knee=-45°
        for i in range(4):  # 4 legs
            # Hip joints (even indices)
            self.data.qpos[7 + i*2] = self.neutral_hip + np.random.uniform(-0.1, 0.1)
            # Knee joints (odd indices)
            self.data.qpos[7 + i*2 + 1] = self.neutral_knee + np.random.uniform(-0.1, 0.1)

        # Set body to target height (based on neutral standing height)
        self.data.qpos[2] = 0.14

        # Forward the simulation to settle
        mujoco.mj_forward(self.model, self.data)

        # Reset action tracking for velocity limiting
        self.prev_action = None

        self.step_count = 0

        obs = self._get_obs()
        info = {}

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
        }

        self.step_count += 1

        # Render if needed
        if self.render_mode == "human" and self.renderer is not None:
            self.renderer.sync()

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
        if self.renderer is not None:
            if self.render_mode == "human":
                self.renderer.close()
            self.renderer = None
