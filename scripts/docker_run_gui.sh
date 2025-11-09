#!/bin/bash
# Helper script to run GUI applications from Docker container

set -e

echo "=================================="
echo "PiDog Docker GUI Helper"
echo "=================================="

# Allow X server connections from Docker
echo "Allowing X server connections from localhost..."
xhost +local:docker

# Set MUJOCO_GL to use GLFW for GUI
export MUJOCO_GL=glfw

echo ""
echo "Starting Docker container with GUI support..."
echo "Container: pidog_rl_training"
echo "Display: $DISPLAY"
echo ""

# Run docker-compose
docker-compose up -d pidog_rl

echo ""
echo "Container started! Now you can:"
echo "  1. Enter the container:"
echo "     docker exec -it pidog_rl_training bash"
echo ""
echo "  2. Run visualization tests:"
echo "     uv run python tests/test_walk.py"
echo "     uv run python tests/sit.py"
echo ""
echo "To stop the container:"
echo "  docker-compose down"
echo ""
