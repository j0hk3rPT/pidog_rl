import trimesh
import os, glob

meshes = ["acrylic", "back", "chest", "eletronics", "leg_a", "leg_b", "motor", "torso_left", "torso_right"]

meshes = [os.path.basename(f) for f in glob.glob("model/*.obj")]

for m in meshes:
    mesh = trimesh.load(f'model/{m}')
    mesh.apply_scale(0.001)  # convert mm → m
    mesh.export(f'model/assets/meshes/{m}')
