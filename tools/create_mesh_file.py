import xml.etree.ElementTree as ET
import os

tree = ET.parse("res/meshes_template.xml")
root = tree.getroot()

asset = root.find("asset")

meshes = [x[0] for x in os.walk("model/assets/meshes")][1:]

for m in meshes:
    mesh = m.split("/")[-1]
    for f in os.listdir(m):
        if "collision" in f:
            components = f[:-4].split("_") 
            name = mesh + "_c" + components[-1]
            new_texture = ET.SubElement(asset, "mesh", {
                "name": name,
                "file": f"assets/meshes/{mesh}/{f}",
            })
        else:
            components = f[:-4].split("_") 
            name = mesh
            new_texture = ET.SubElement(asset, "mesh", {
                "name":name,
                "file": f"assets/meshes/{mesh}/{f}",
            })

tree.write("model/meshes.xml", encoding="utf-8", xml_declaration=True)