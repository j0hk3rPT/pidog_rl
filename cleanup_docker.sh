#!/bin/bash
echo "============================================"
echo "PiDog Docker Cleanup Script"
echo "============================================"
echo ""
echo "Step 1: Stopping docker-compose services..."
docker-compose down 2>/dev/null || true
echo ""
echo "Step 2: Force removing all related containers..."
docker ps -a | grep pidog | awk '{print $1}' | xargs -r docker rm -f 2>/dev/null || true
docker ps -a | grep 565a849592ef | awk '{print $1}' | xargs -r docker rm -f 2>/dev/null || true
echo ""
echo "Step 3: Cleaning up volumes (optional)..."
docker volume ls | grep pidog | awk '{print $2}' | xargs -r docker volume rm 2>/dev/null || true
echo ""
echo "============================================"
echo "Cleanup complete!"
echo "============================================"
echo ""
echo "Now run: ./scripts/docker_run_gui.sh"
