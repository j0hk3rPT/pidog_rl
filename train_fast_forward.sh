#!/bin/bash
# Quick-start script for training PiDog to move fast forward using RL

set -e

echo "============================================================"
echo "PiDog RL Training - Fast Forward Movement"
echo "============================================================"
echo ""
echo "This will train the robot to move forward as fast as possible"
echo "while maintaining balance and stability."
echo ""
echo "Training configuration:"
echo "  - Algorithm: PPO (Proximal Policy Optimization)"
echo "  - Parallel envs: 4"
echo "  - Total timesteps: 1,000,000"
echo "  - Device: auto (will use GPU if available)"
echo ""
echo "Results will be saved in: outputs/"
echo "Monitor training with: tensorboard --logdir outputs/[experiment]/logs"
echo ""
echo "============================================================"
echo ""

# Run training inside Docker container
docker-compose run --rm pidog_rl python training/train_rl.py \
    --algorithm ppo \
    --total-timesteps 1000000 \
    --n-envs 4 \
    --learning-rate 3e-4 \
    --n-steps 2048 \
    --batch-size 64 \
    --save-freq 10000 \
    --eval-freq 5000 \
    --seed 42 \
    --device auto \
    --experiment-name pidog_fast_forward

echo ""
echo "============================================================"
echo "Training completed!"
echo "============================================================"
