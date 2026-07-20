import os
import json
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

data_dir = "data"
scales_path = os.path.join(data_dir, "scales.json")
with open(scales_path, "r") as f:
    scales = json.load(f)

for slide_id in ["P16357_1001", "P16357_1005", "P16357_1013", "P16357_1017", "P16357_1025", "P16357_1037"]:
    img_path = os.path.join(data_dir, "images", f"{slide_id}_HE.tif")
    if os.path.exists(img_path):
        size_bytes = os.path.getsize(img_path)
        if size_bytes > 5 * 1024 * 1024:  # If > 5MB, it's the raw image
            print(f"Resizing {img_path} ({size_bytes/1024/1024:.1f} MB)...")
            img = Image.open(img_path)
            orig_w, orig_h = img.size
            if slide_id in scales:
                new_w = int(orig_w * scales[slide_id]["scale_x"])
                new_h = int(orig_h * scales[slide_id]["scale_y"])
            else:
                new_w = 2000
                new_h = 2000
                
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img.close()
            img_resized.save(img_path, format="JPEG") # Save as JPEG to match size
            print(f"  -> Resized to {new_w}x{new_h}")
