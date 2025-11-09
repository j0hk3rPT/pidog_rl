#!/bin/bash
# Monitor training progress and GPU usage

echo "============================================================"
echo "PiDog RL Training Monitor"
echo "============================================================"
echo ""

# Detect GPU type
if command -v rocm-smi &> /dev/null; then
    echo "✓ AMD GPU detected (ROCm)"
    GPU_CMD="rocm-smi"
    GPU_TYPE="AMD"
elif command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    GPU_CMD="nvidia-smi"
    GPU_TYPE="NVIDIA"
else
    echo "⚠ No GPU monitoring tool found"
    echo "  Install rocm-smi (AMD) or nvidia-smi (NVIDIA)"
    GPU_CMD=""
    GPU_TYPE="CPU"
fi

echo ""
echo "Monitoring mode: $GPU_TYPE"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo "============================================================"
echo ""

# Monitor GPU usage
if [ -n "$GPU_CMD" ]; then
    watch -n 1 -c "$GPU_CMD"
else
    echo "CPU-only mode - monitoring system resources:"
    watch -n 1 'top -bn1 | head -n 20'
fi
