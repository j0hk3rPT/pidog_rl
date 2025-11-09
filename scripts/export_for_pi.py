"""
Export trained policy for deployment on Raspberry Pi PiDog.

This script exports a trained Stable-Baselines3 model to a lightweight format
that can run on Raspberry Pi with minimal dependencies.
"""

import argparse
from pathlib import Path
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Export policy for Raspberry Pi")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained SB3 model (.zip)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pi_deployment",
        help="Output directory for exported files",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "sac", "td3"],
        help="Algorithm used",
    )
    return parser.parse_args()


def export_policy(model_path, output_dir, algorithm):
    """Export policy to PyTorch format for Raspberry Pi."""
    from stable_baselines3 import PPO, SAC, TD3

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}...")

    # Load the trained model
    if algorithm == "ppo":
        model = PPO.load(model_path)
    elif algorithm == "sac":
        model = SAC.load(model_path)
    elif algorithm == "td3":
        model = TD3.load(model_path)

    # Extract the policy network
    policy = model.policy

    # Save the policy network as PyTorch model
    policy_path = output_dir / "policy.pth"
    torch.save(policy.state_dict(), policy_path)
    print(f"Saved policy weights to: {policy_path}")

    # Save observation normalization stats if available
    if hasattr(model, "obs_rms") and model.obs_rms is not None:
        obs_rms = {
            "mean": model.obs_rms.mean,
            "var": model.obs_rms.var,
            "count": model.obs_rms.count,
        }
        np.savez(output_dir / "obs_stats.npz", **obs_rms)
        print(f"Saved observation stats to: {output_dir / 'obs_stats.npz'}")

    # Create a simple inference script for Raspberry Pi
    inference_script = """
'''
Simple inference script for Raspberry Pi PiDog.
Deploy this file along with policy.pth to your Raspberry Pi.
'''

import torch
import numpy as np

class PiDogPolicy:
    def __init__(self, policy_path="policy.pth", obs_stats_path=None):
        # Load policy
        self.policy = torch.load(policy_path, map_location="cpu")
        self.policy.eval()

        # Load normalization stats if available
        self.obs_mean = None
        self.obs_std = None
        if obs_stats_path:
            stats = np.load(obs_stats_path)
            self.obs_mean = stats["mean"]
            self.obs_std = np.sqrt(stats["var"])

    def predict(self, observation):
        '''Get action from observation.'''
        # Normalize observation
        if self.obs_mean is not None:
            obs = (observation - self.obs_mean) / (self.obs_std + 1e-8)
        else:
            obs = observation

        # Get action from policy
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_tensor = self.policy(obs_tensor)
            action = action_tensor.cpu().numpy()[0]

        return action

# Example usage:
if __name__ == "__main__":
    policy = PiDogPolicy("policy.pth")

    # Mock observation (replace with real sensor data)
    obs = np.zeros(27)  # 27-dim observation

    # Get action
    action = policy.predict(obs)
    print(f"Action: {action}")
"""

    with open(output_dir / "pi_inference.py", "w") as f:
        f.write(inference_script)
    print(f"Created inference script: {output_dir / 'pi_inference.py'}")

    # Create deployment instructions
    readme = f"""
# PiDog Policy Deployment

## Files
- `policy.pth`: Trained neural network weights
- `obs_stats.npz`: Observation normalization statistics (if available)
- `pi_inference.py`: Inference script for Raspberry Pi

## Installation on Raspberry Pi

```bash
# Install PyTorch for Raspberry Pi
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install NumPy
pip3 install numpy
```

## Usage

```python
from pi_inference import PiDogPolicy

# Load policy
policy = PiDogPolicy("policy.pth", "obs_stats.npz")

# Get observation from sensors (27-dimensional)
observation = get_sensor_data()  # Implement this

# Predict action (8-dimensional joint positions)
action = policy.predict(observation)

# Send action to servos
send_to_servos(action)  # Implement this
```

## Observation Format (27-dim)
1. Joint positions (8): back_right_hip, back_right_knee, front_right_hip, front_right_knee,
                        back_left_hip, back_left_knee, front_left_hip, front_left_knee
2. Joint velocities (8): same order as positions
3. Body orientation (4): quaternion [x, y, z, w]
4. Body linear velocity (3): [vx, vy, vz]
5. Body angular velocity (3): [wx, wy, wz]
6. Body height (1): z-position

## Action Format (8-dim)
Normalized joint positions [-1, 1] for 8 leg joints (same order as above).
Scale to servo range: 0-180° (0 to π radians).
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(readme)
    print(f"Created deployment README: {output_dir / 'README.md'}")

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"Deploy the '{output_dir}' directory to your Raspberry Pi")
    print("=" * 60)


def main():
    args = parse_args()
    export_policy(args.model_path, args.output_dir, args.algorithm)


if __name__ == "__main__":
    main()
