import os
from PIL import Image

def resize_images():
    images_dir = "data/images"
    if not os.path.exists(images_dir):
        print(f"Directory {images_dir} does not exist.")
        return

    for img_file in os.listdir(images_dir):
        if img_file.endswith(".tif") or img_file.endswith(".jpg"):
            img_path = os.path.join(images_dir, img_file)
            try:
                with Image.open(img_path) as img:
                    if max(img.size) > 2000:
                        img.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
                        img.save(img_path)
                        print(f"Resized {img_file} to {img.size}")
                    else:
                        print(f"Skipped {img_file}, already {img.size}")
            except Exception as e:
                print(f"Error resizing {img_file}: {e}")

if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = None
    resize_images()
