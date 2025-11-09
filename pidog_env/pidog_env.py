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
            "range": (0, np.pi),              # 0-180° in radians
            "max_torque": 0.137,              # Nm (at 6V)
            "min_torque": 0.127,              # Nm (at 4.8V)
            "max_speed": 7.0,                 # rad/s (400°/s)
            "min_speed": 5.8,                 # rad/s (333°/s)
            "voltage_range": (4.8, 6.0),      # Operating voltage
        }

        # Joint limits (based on real servo: 0-180°)
        # Centered at π/2 (90°), with ±π/2 range
        self.joint_limits = {
            "shoulder": (0, np.pi),           # 0-180° full range
            "knee": (0, np.pi),               # 0-180° full range
        }

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
        - Range: 0-180° (0 to π radians)
        - Max speed: 7.0 rad/s (400°/s)
        """
        scaled_action = np.zeros_like(action)

        # For each joint, scale to its limits (all servos have 0-180° range)
        for i in range(self.n_joints):
            low, high = self.servo_specs["range"]
            # Map from [-1, 1] to [0, π]
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
        # Forward velocity reward
        forward_vel = self.data.qvel[0]
        velocity_reward = -abs(forward_vel - self.target_forward_velocity)

        # Height reward (penalize if body is too low)
        body_height = self.data.qpos[2]
        target_height = 0.15  # Target height in meters
        height_reward = -abs(body_height - target_height)

        # Stability reward (penalize large roll/pitch)
        body_quat = self.data.qpos[3:7]
        # Convert quaternion to roll, pitch
        # Simple approximation: penalize non-upright orientation
        upright_reward = body_quat[3]  # w component (1 = upright, 0 = sideways)

        # Energy penalty (penalize large actions)
        action_penalty = -0.01 * np.sum(np.square(self.data.ctrl))

        # Combine rewards
        reward = (
            2.0 * velocity_reward +
            1.0 * height_reward +
            1.0 * upright_reward +
            action_penalty
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

        # Randomize initial joint positions slightly around neutral (π/2)
        # Servos start at mid-range (90°)
        neutral_pos = np.pi / 2
        self.data.qpos[7:15] = neutral_pos + np.random.uniform(-0.1, 0.1, 8)

        # Set body to target height
        self.data.qpos[2] = 0.15

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
