# Use ROCm PyTorch as base image (includes Python, PyTorch, and ROCm support)
FROM rocm/pytorch:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=osmesa

# Install system dependencies for MuJoCo, rendering, and development
RUN apt-get update && apt-get install -y \
    # MuJoCo dependencies
    libglew-dev \
    libosmesa6-dev \
    libgl1 \
    libglfw3 \
    libglfw3-dev \
    patchelf \
    # X11 and GUI support
    libx11-6 \
    libxext6 \
    libxrender1 \
    # Build tools
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # Additional utilities
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (PyTorch already included in base image)
RUN pip install --no-cache-dir \
    mujoco \
    gymnasium \
    "stable-baselines3[extra]" \
    tensorboard \
    matplotlib \
    numpy \
    imitation \
    "sb3-extra-buffers[fast,extra]"

# Set working directory (repo will be mounted here via docker-compose volume)
WORKDIR /workspace/pidog_rl

# Set up MuJoCo license path (if needed)
ENV MUJOCO_LICENSE_PATH=/workspace/pidog_rl/.mujoco/mjkey.txt

# Expose TensorBoard port
EXPOSE 6006

# Default command
CMD ["/bin/bash"]
