# Use ROCm PyTorch as base image
FROM rocm/pytorch:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=osmesa \
    UV_SYSTEM_PYTHON=1

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

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /workspace/pidog_rl

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY model/ ./model/
COPY pidog_env/ ./pidog_env/
COPY training/ ./training/
COPY configs/ ./configs/
COPY build/ ./build/
COPY tools/ ./tools/
COPY tests/ ./tests/
COPY res/ ./res/
COPY scripts/ ./scripts/

# Install Python dependencies
RUN uv sync

# Create directories for outputs
RUN mkdir -p /workspace/pidog_rl/outputs \
    /workspace/pidog_rl/logs \
    /workspace/pidog_rl/checkpoints \
    /workspace/pidog_rl/datasets

# Set up MuJoCo license path (if needed)
ENV MUJOCO_LICENSE_PATH=/workspace/pidog_rl/.mujoco/mjkey.txt

# Expose TensorBoard port
EXPOSE 6006

# Default command
CMD ["/bin/bash"]
