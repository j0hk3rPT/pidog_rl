#!/bin/bash
# Quick test of the flipped coordinate fix

set -e

echo "========================================================================"
echo " TESTING FLIPPED COORDINATE FIX"
echo "========================================================================"
echo ""
echo "This will:"
echo "  1. Extract 5 cycles of demonstrations with FLIPPED Y coordinates"
echo "  2. Visualize them to see if robot walks FORWARD (not backward)"
echo ""
echo "If the robot walks forward and stays stable, the fix worked!"
echo "========================================================================"
echo ""

# Extract demonstrations
echo "Step 1: Extracting demonstrations with flipped coordinates..."
python extract_sunfounder_demos_flipped.py \
    --n-cycles 5 \
    --standing-height 80 \
    --output-file demonstrations/test_flipped.pkl

echo ""
echo "Step 2: Visualizing..."
echo ""
echo "WATCH CAREFULLY:"
echo "  ✓ Robot should walk FORWARD (in +X direction, toward red axis)"
echo "  ✓ Robot should stay upright and balanced"
echo "  ✓ Gait should look natural"
echo ""
echo "If robot still walks backward, the problem is elsewhere."
echo "Press Ctrl+C to stop visualization when done."
echo ""

python visualize_demonstrations.py \
    --demo-file demonstrations/test_flipped.pkl \
    --fps 30

echo ""
echo "========================================================================"
echo " TEST COMPLETE"
echo "========================================================================"
echo ""
echo "Did the robot walk FORWARD?"
echo "  YES → Use extract_sunfounder_demos_flipped.py for training"
echo "  NO  → Need to investigate further"
echo ""
