"""
Custom Feature Extractors for PiDog Multi-Sensor RL Training

This module provides custom neural network architectures that combine
CNN (for camera images) and MLP (for proprioceptive/sensor vectors)
for use with Stable-Baselines3's MultiInputPolicy.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict


class PiDogCombinedExtractor(BaseFeaturesExtractor):
    """
    Combined CNN-MLP feature extractor for PiDog observations.

    Architecture:
    - CNN branch: Processes RGB camera images (84x84x3)
      - 3 conv layers with ReLU activations
      - Outputs 512-dim feature vector

    - MLP branch: Processes proprioceptive/sensor vector (31-dim)
      - 2 fully connected layers with ReLU
      - Outputs 128-dim feature vector

    - Fusion: Concatenates CNN + MLP features → 640-dim combined features

    This architecture is inspired by:
    - DQN (Mnih et al., 2015) for vision processing
    - Actor-Critic methods for locomotion (Schulman et al., 2017)
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        """
        Initialize the combined feature extractor.

        Args:
            observation_space: Dictionary observation space with 'image' and 'vector' keys
            features_dim: Dimension of final output features (default: 256)
        """
        # Initialize with the final feature dimension
        super().__init__(observation_space, features_dim)

        # Extract image and vector dimensions from observation space
        image_space = observation_space.spaces['image']
        vector_space = observation_space.spaces['vector']

        # Handle both HWC (height, width, channels) and CHW (channels, height, width) formats
        # After VecTransposeImage wrapping, images are in CHW format for PyTorch
        if image_space.shape[0] in [1, 3, 4]:  # Likely CHW format (channels first)
            n_input_channels = image_space.shape[0]
            height = image_space.shape[1]
            width = image_space.shape[2]
        else:  # HWC format (channels last)
            height = image_space.shape[0]
            width = image_space.shape[1]
            n_input_channels = image_space.shape[2]

        vector_dim = vector_space.shape[0]  # 31 dimensions

        # =============== CNN Branch for Image Processing ===============
        # Input: (batch, 3, 84, 84)
        self.cnn = nn.Sequential(
            # Conv layer 1: 3x84x84 → 32x20x20
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),

            # Conv layer 2: 32x20x20 → 64x9x9
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),

            # Conv layer 3: 64x9x9 → 64x7x7
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),

            # Flatten: 64x7x7 → 3136
            nn.Flatten(),
        )

        # Calculate CNN output dimension
        with torch.no_grad():
            sample_image = torch.zeros(1, n_input_channels, height, width)
            cnn_output_dim = self.cnn(sample_image).shape[1]

        # CNN feature compression: 3136 → 512
        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_output_dim, 512),
            nn.ReLU(),
        )

        # =============== MLP Branch for Vector Processing ===============
        # Input: (batch, 31) - joint pos/vel, IMU, ultrasonic, etc.
        self.mlp = nn.Sequential(
            # FC layer 1: 31 → 128
            nn.Linear(vector_dim, 128),
            nn.ReLU(),

            # FC layer 2: 128 → 128
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # =============== Fusion Layer ===============
        # Combines CNN (512) + MLP (128) → features_dim
        combined_dim = 512 + 128  # 640
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the combined extractor.

        Args:
            observations: Dictionary with 'image' (uint8, HWC) and 'vector' (float32) tensors

        Returns:
            Combined feature tensor of shape (batch, features_dim)
        """
        # Extract image and vector from observations dict
        image = observations['image']
        vector = observations['vector']

        # Preprocess image: convert to float and normalize to [0, 1]
        image = image.float() / 255.0

        # Check if we need to permute (HWC → CHW)
        # After VecTransposeImage, images are already in CHW format
        if image.shape[1] != 3:  # If not already CHW (channels not at dim 1)
            image = image.permute(0, 3, 1, 2)  # HWC → CHW

        # Process image through CNN
        cnn_features = self.cnn(image)
        cnn_features = self.cnn_fc(cnn_features)  # → (batch, 512)

        # Process vector through MLP
        mlp_features = self.mlp(vector)  # → (batch, 128)

        # Concatenate features
        combined = torch.cat([cnn_features, mlp_features], dim=1)  # → (batch, 640)

        # Final fusion
        output = self.fusion(combined)  # → (batch, features_dim)

        return output


class PiDogVisionExtractor(BaseFeaturesExtractor):
    """
    Vision-only CNN feature extractor for PiDog (no proprioception).

    Use this when you want pure end-to-end vision-based learning.
    Architecture follows DQN from Mnih et al., 2015.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # Handle both HWC and CHW formats
        if observation_space.shape[0] in [1, 3, 4]:  # CHW format
            n_input_channels = observation_space.shape[0]
            height = observation_space.shape[1]
            width = observation_space.shape[2]
        else:  # HWC format
            height = observation_space.shape[0]
            width = observation_space.shape[1]
            n_input_channels = observation_space.shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate output dimension
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, height, width)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalize and convert HWC → CHW
        observations = observations.float() / 255.0
        observations = observations.permute(0, 3, 1, 2)

        features = self.cnn(observations)
        return self.linear(features)


class PiDogProprioceptionExtractor(BaseFeaturesExtractor):
    """
    MLP-only feature extractor for proprioceptive/sensor data (no vision).

    Use this for faster training without camera or for baseline comparisons.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]  # 31 dimensions

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)


# ================== Advanced Architectures ==================

class PiDogNatureCNNExtractor(BaseFeaturesExtractor):
    """
    Combined extractor using Nature DQN architecture (deeper CNN).

    This is a more powerful but slower architecture suitable for
    complex visual tasks or obstacle-rich environments.

    Reference: "Human-level control through deep reinforcement learning"
               Mnih et al., Nature 2015
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        image_space = observation_space.spaces['image']
        vector_space = observation_space.spaces['vector']

        # Handle both HWC and CHW formats
        if image_space.shape[0] in [1, 3, 4]:  # CHW format
            n_input_channels = image_space.shape[0]
            height = image_space.shape[1]
            width = image_space.shape[2]
        else:  # HWC format
            height = image_space.shape[0]
            width = image_space.shape[1]
            n_input_channels = image_space.shape[2]

        vector_dim = vector_space.shape[0]

        # Deeper CNN for complex vision tasks
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, height, width)
            cnn_output_dim = self.cnn(sample).shape[1]

        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # Deeper MLP for proprioception
        self.mlp = nn.Sequential(
            nn.Linear(vector_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Fusion with dropout for regularization
        combined_dim = 512 + 256
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        image = observations['image'].float() / 255.0

        # Check if we need to permute (HWC → CHW)
        if image.shape[1] != 3:  # If not already CHW
            image = image.permute(0, 3, 1, 2)

        vector = observations['vector']

        cnn_features = self.cnn(image)
        cnn_features = self.cnn_fc(cnn_features)

        mlp_features = self.mlp(vector)

        combined = torch.cat([cnn_features, mlp_features], dim=1)
        return self.fusion(combined)
