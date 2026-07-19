import os
import subprocess
import json

img_dir = "data/images"
scales = {}

for f in os.listdir(img_dir):
    if f.endswith("_HE.tif"):
        img_path = os.path.join(img_dir, f)
        
        # Get original width
        out = subprocess.check_output(["sips", "-g", "pixelWidth", img_path]).decode()
        width = int(out.strip().split()[-1])
        
        out_h = subprocess.check_output(["sips", "-g", "pixelHeight", img_path]).decode()
        height = int(out_h.strip().split()[-1])
        
        print(f"{f}: {width}x{height}")
        
        scale_x = 2000.0 / width
        scale_y = 2000.0 / height
        
        # Resize to max 2000
        subprocess.run(["sips", "-Z", "2000", img_path])
        
        scales[f.replace("_HE.tif", "")] = {"scale_x": scale_x, "scale_y": scale_y}

with open("data/scales.json", "w") as f:
    json.dump(scales, f)

print("Done resizing!")
