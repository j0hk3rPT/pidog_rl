
uv run tools/scale_meshes.py
echo "[1/3] Meshes scaled to SI units"

uv run obj2mjcf --obj-dir model/assets/meshes  --decompose --no-verbose --coacd-args.threshold 0.1 
echo "[2/3] Meshes decomposed in convex meshes"

uv run tools/create_mesh_file.py
echo "[3/3] Assets XML updated with new meshes"