#!/bin/bash
echo "Stopping and removing old containers..."
docker stop pidog_rl_training 2>/dev/null || true
docker rm pidog_rl_training 2>/dev/null || true
docker stop pidog_tensorboard 2>/dev/null || true
docker rm pidog_tensorboard 2>/dev/null || true
echo "Cleanup complete!"
echo ""
echo "Now run: ./scripts/docker_run_gui.sh"
