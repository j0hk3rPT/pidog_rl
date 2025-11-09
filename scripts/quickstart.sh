#!/bin/bash
# Quick start script for PiDog training

set -e

echo "=================================="
echo "PiDog Training Quick Start"
echo "=================================="

# Check if in Docker
if [ -f /.dockerenv ]; then
    echo "Running inside Docker container"
else
    echo "Running on host system"
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p outputs logs checkpoints datasets

# Collect expert demonstrations (if not exists)
if [ ! -f "datasets/expert_demos.pkl" ]; then
    echo ""
    echo "Collecting expert demonstrations..."
    uv run python training/collect_demos.py \
        --n-episodes 20 \
        --output-path datasets/expert_demos.pkl
else
    echo ""
    echo "Expert demonstrations already exist at datasets/expert_demos.pkl"
fi

# Ask user what they want to do
echo ""
echo "What would you like to do?"
echo "1) Train with RL (PPO)"
echo "2) Train with Imitation Learning (Behavioral Cloning)"
echo "3) Train with Imitation Learning (GAIL)"
echo "4) Evaluate a trained model"
echo "5) Test environment"
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "Starting RL training with PPO..."
        uv run python training/train_rl.py \
            --algorithm ppo \
            --total-timesteps 100000 \
            --n-envs 4
        ;;
    2)
        echo ""
        echo "Starting Behavioral Cloning training..."
        uv run python training/train_imitation.py \
            --method bc \
            --expert-data datasets/expert_demos.pkl \
            --n-epochs 100
        ;;
    3)
        echo ""
        echo "Starting GAIL training..."
        uv run python training/train_imitation.py \
            --method gail \
            --expert-data datasets/expert_demos.pkl \
            --total-timesteps 100000
        ;;
    4)
        read -p "Enter path to model: " model_path
        read -p "Enter algorithm (ppo/sac/td3): " algo
        echo ""
        echo "Evaluating model..."
        uv run python training/evaluate.py \
            --model-path "$model_path" \
            --algorithm "$algo" \
            --n-episodes 10 \
            --render
        ;;
    5)
        echo ""
        echo "Testing environment..."
        uv run python tests/sit.py
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Done!"
