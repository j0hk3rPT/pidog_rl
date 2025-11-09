#!/bin/bash
# Test all gait extraction versions to find which one works

set -e

echo "========================================================================"
echo " TESTING ALL GAIT EXTRACTION VERSIONS"
echo "========================================================================"
echo ""
echo "This will test 3 versions:"
echo "  1. FLIPPED Y (negates Y coordinate)"
echo "  2. NO MIRROR (no right-side angle negation)"
echo "  3. BOTH (flipped Y + no mirror)"
echo ""
echo "========================================================================"
echo ""

# Version 1: Flipped Y (already tested, didn't work well)
echo "VERSION 1: FLIPPED Y COORDINATE"
echo "----------------------------------------"
python extract_sunfounder_demos_flipped.py \
    --n-cycles 3 \
    --output-file demonstrations/test_v1_flipped.pkl
echo ""

# Version 2: No mirroring
echo "VERSION 2: NO MIRRORING"
echo "----------------------------------------"
python extract_sunfounder_demos_no_mirror.py \
    --n-cycles 3 \
    --output-file demonstrations/test_v2_no_mirror.pkl
echo ""

# Version 3: Original (for baseline)
echo "VERSION 3: ORIGINAL (baseline)"
echo "----------------------------------------"
python extract_sunfounder_demos_correct.py \
    --n-cycles 3 \
    --output-file demonstrations/test_v3_original.pkl
echo ""

echo "========================================================================"
echo " ALL VERSIONS EXTRACTED"
echo "========================================================================"
echo ""
echo "Now test each one visually:"
echo ""
echo "  python visualize_demonstrations.py --demo-file demonstrations/test_v1_flipped.pkl"
echo "  python visualize_demonstrations.py --demo-file demonstrations/test_v2_no_mirror.pkl"
echo "  python visualize_demonstrations.py --demo-file demonstrations/test_v3_original.pkl"
echo ""
echo "Watch for which one walks forward and stays balanced!"
echo ""
