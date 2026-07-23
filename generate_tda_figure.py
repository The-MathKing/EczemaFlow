"""
Generate the Topology Pipeline Diagram (Figure 1 in new.tex).
Shows: H&E patch -> StarDist masks -> Point Cloud -> Persistence Diagram.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os
import scanpy as sc
from csbdeep.utils import normalize
from stardist.models import StarDist2D
import ripser
import persim

# Load test slide to find a good patch
print("Loading H&E slide...")
slide_id = "P15509_1001" 
adata = sc.read_h5ad("data/GSE206391/GSE206391_Preprocessed_data.h5")
adata = adata[adata.obs['sample'] == slide_id]
img = Image.open(f"data/images/{slide_id}_HE.tif")

# Find a spot with high density by searching 20 random spots
import json
with open("data/scales.json", "r") as f:
    scales = json.load(f)
scale = scales[slide_id]

patch_size = 224
print("Running StarDist to find high density spot...")
model = StarDist2D.from_pretrained('2D_versatile_he')

best_idx = 0
max_nuclei = 0
best_patch = None
best_labels = None
best_details = None

np.random.seed(42)
indices_to_check = np.random.choice(len(adata), 30, replace=False)

for idx in indices_to_check:
    px, py = adata.obsm['spatial'][idx]
    px = px * scale['scale_x']
    py = py * scale['scale_y']
    
    left = max(0, int(px - patch_size // 2))
    top = max(0, int(py - patch_size // 2))
    right = left + patch_size
    bottom = top + patch_size
    
    # ensure we don't go out of bounds
    if right > img.width or bottom > img.height:
        continue
        
    patch = img.crop((left, top, right, bottom))
    img_np = np.array(patch)
    img_norm = normalize(img_np, 1, 99.8, axis=(0,1))
    
    labels, details = model.predict_instances(img_norm, prob_thresh=0.692, nms_thresh=0.3)
    num_nuclei = len(details['points'])
    
    if num_nuclei > max_nuclei:
        max_nuclei = num_nuclei
        best_idx = idx
        best_patch = img_np
        best_labels = labels
        best_details = details

print(f"Found best spot at index {best_idx} with {max_nuclei} nuclei.")
img_np = best_patch
labels = best_labels
details = best_details

print("Running StarDist...")
model = StarDist2D.from_pretrained('2D_versatile_he')
img_norm = normalize(img_np, 1, 99.8, axis=(0,1))
labels, details = model.predict_instances(img_norm, prob_thresh=0.692, nms_thresh=0.3)
centroids = details['coord']  # (N, 2, num_rays) we need centroids (N, 2)
# The 'points' key holds (y, x) centroids in StarDist
centroids = details['points']

print("Computing Persistent Homology...")
# Flip to (x, y) for plotting
points = np.stack([centroids[:, 1], centroids[:, 0]], axis=1)

# Ripser computes VR filtration
dgms = ripser.ripser(points, maxdim=1)['dgms']

print("Generating figure...")
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
plt.subplots_adjust(wspace=0.3)

# 1. H&E Patch
axs[0].imshow(img_np)
axs[0].set_title("H&E Patch", fontsize=14, fontweight='bold')
axs[0].axis('off')

# 2. Nuclei Masks
# Create a nice overlay
from skimage.color import label2rgb
overlay = label2rgb(labels, image=img_np, bg_label=0, alpha=0.5)
axs[1].imshow(overlay)
axs[1].set_title("StarDist Segmentation", fontsize=14, fontweight='bold')
axs[1].axis('off')

# 3. Point Cloud
axs[2].scatter(points[:, 0], points[:, 1], s=20, c='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.5)
axs[2].set_xlim(0, 224)
axs[2].set_ylim(224, 0) # flip y axis to match image
axs[2].set_title("Nuclei Point Cloud", fontsize=14, fontweight='bold')
axs[2].set_aspect('equal')
axs[2].grid(True, linestyle='--', alpha=0.5)

# 4. Persistence Diagram
# persim plot
persim.plot_diagrams(dgms, ax=axs[3], legend=True)
axs[3].set_title("Persistence Diagram", fontsize=14, fontweight='bold')

os.makedirs("paper/figures", exist_ok=True)
out_path = "paper/figures/topology_pipeline.pdf"
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"Figure saved to {out_path}")
