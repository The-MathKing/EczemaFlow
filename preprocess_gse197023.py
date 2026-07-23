"""
Preprocess GSE197023 (Mitamura et al. 2023) - External Validation Cohort.

Extracts all AD-only .tar.gz files from data/GSE197023/,
reads the Visium data with scanpy, ALIGNS genes to GSE206391 reference, 
extracts spot features using frozen ViT-B/16 and TDA, and saves
precomputed .pt files to data/precomputed_gse197023/.
"""
import os
import tarfile
import torch
import numpy as np
import scanpy as sc
import sys
from PIL import Image, ImageFile
import torchvision.transforms.v2 as v2
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, '/Volumes/2TB/EczemaFlow-1')
from eczema_flow.attention import MorphologyEncoder
from eczema_flow.tda import TDAFeatureExtractor

GEO_DIR = "data/GSE197023"
PRECOMP_DIR = "data/precomputed_gse197023"
os.makedirs(PRECOMP_DIR, exist_ok=True)

# 0. Load the reference genes from GSE206391 to align external data
print("Loading reference genes from GSE206391...")
ref_adata = sc.read_h5ad("data/GSE206391/GSE206391_Preprocessed_data.h5")
ref_genes = list(ref_adata.var_names)
print(f"Reference has {len(ref_genes)} genes.")
del ref_adata

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
cnn_encoder = MorphologyEncoder(embed_dim=256, use_pretrained=True).to(device)
cnn_encoder.eval()
tda_extractor = TDAFeatureExtractor(output_dim=256).to(device)
tda_extractor.eval()

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

AD_TARBALLS = [f for f in os.listdir(GEO_DIR)
               if f.endswith(".tar.gz") and ("_AD_" in f or "_HE_" not in f)
               and f != "GSE197023_RAW.tar" and "_HE_" not in f]

sample_dirs = {}
for tarball in sorted(AD_TARBALLS):
    gsm_id = tarball.split("_")[0]
    sample_label = tarball.replace(".tar.gz", "").replace(f"{gsm_id}_", "")
    sample_dir = os.path.join(GEO_DIR, gsm_id)
    sample_dirs[gsm_id] = (sample_dir, sample_label)

print("\nLoading Visium samples and extracting features...")
all_features = []
all_counts = []
all_coords = []
all_gsm = []

for gsm_id, (sdir, label) in sample_dirs.items():
    visium_dir = sdir
    for root, dirs, fnames in os.walk(sdir):
        if "filtered_feature_bc_matrix.h5" in fnames or "barcodes.tsv.gz" in fnames:
            visium_dir = root
            break
        if "spatial" in dirs and any(f.endswith(".h5") for f in fnames):
            visium_dir = root
            break
    
    try:
        adata = sc.read_visium(visium_dir)
        adata.var_names_make_unique()
        
        # ALIGN GENES to reference
        # Create a zero matrix of the reference size
        aligned_X = np.zeros((adata.n_obs, len(ref_genes)), dtype=np.float32)
        
        # Find which genes match
        curr_genes = list(adata.var_names)
        curr_gene_to_idx = {g: i for i, g in enumerate(curr_genes)}
        
        matched = 0
        original_X = adata.X
        if hasattr(original_X, 'toarray'):
            original_X = original_X.toarray()
            
        for ref_i, ref_g in enumerate(ref_genes):
            if ref_g in curr_gene_to_idx:
                curr_i = curr_gene_to_idx[ref_g]
                aligned_X[:, ref_i] = original_X[:, curr_i]
                matched += 1
                
        print(f"\nProcessing {gsm_id} ({label}): {adata.n_obs} spots, matched {matched}/{len(ref_genes)} ref genes")
        
        # Replace adata.X with the aligned matrix
        adata = sc.AnnData(X=aligned_X, obs=adata.obs)
        adata.var_names = ref_genes
        
        # Find H&E image
        img_path = None
        for root, dirs, fnames in os.walk(visium_dir):
            for fn in fnames:
                if fn.endswith('.tif') or fn.endswith('.png') or fn.endswith('.jpg'):
                    if 'hires' in fn or 'tissue' in fn:
                        img_path = os.path.join(root, fn)
                        break
            if img_path: break
            
        if img_path is None:
            print(f"  Warning: No image found for {gsm_id}")
            continue
            
        img = Image.open(img_path)
        img_array = np.array(img)
        img.close()
        
        # Process each spot
        batch_features = []
        batch_counts = []
        batch_coords = []
        
        with torch.no_grad():
            for idx in tqdm(range(adata.n_obs), desc=f"Extracting {gsm_id}"):
                obs = adata.obs.iloc[idx]
                if 'px_x' in obs and 'px_y' in obs:
                    px, py = obs['px_x'], obs['px_y']
                elif 'spatial' in adata.obsm:
                    px, py = adata.obsm['spatial'][idx]
                else:
                    px = obs.get('array_col', 50) * 10
                    py = obs.get('array_row', 50) * 10
                    
                x_val = adata.X[idx]
                counts = torch.tensor(x_val, dtype=torch.float32)
                    
                img_p = Image.fromarray(img_array)
                left = max(0, int(px - 112))
                top = max(0, int(py - 112))
                right = left + 224
                bottom = top + 224
                patch = img_p.crop((left, top, right, bottom))
                
                patch_tensor = transform(patch)
                patches = patch_tensor.unsqueeze(0).expand(4, -1, -1, -1).unsqueeze(0).to(device) # (1, 4, C, H, W)
                
                b, n, c, h, w = patches.shape
                patches_flat = patches.view(b * n, c, h, w)
                patch_features = cnn_encoder(patches_flat)
                
                tda_features = tda_extractor(patches)
                tda_features_flat = tda_features.view(b * n, -1)
                
                combined_features = torch.cat([patch_features, tda_features_flat], dim=-1)
                combined_features = combined_features.view(b, n, -1)
                
                batch_features.append(combined_features.cpu())
                batch_counts.append(counts.cpu().unsqueeze(0))
                batch_coords.append(torch.tensor([[px, py]], dtype=torch.float32))
                
        all_features.append(torch.cat(batch_features, dim=0))
        all_counts.append(torch.cat(batch_counts, dim=0))
        all_coords.append(torch.cat(batch_coords, dim=0))
        all_gsm.extend([gsm_id] * len(batch_features))
        
    except Exception as e:
        print(f"  {gsm_id}: FAILED - {e}")

if not all_features:
    print("ERROR: No features extracted.")
    sys.exit(1)

print("\nSaving precomputed external cohort...")
features_cat = torch.cat(all_features, dim=0)
counts_cat = torch.cat(all_counts, dim=0)
coords_cat = torch.cat(all_coords, dim=0)

out_path = os.path.join(PRECOMP_DIR, "external_cohort.pt")
torch.save({
    'features': features_cat,
    'counts': counts_cat,
    'coords': coords_cat,
    'gsm_ids': all_gsm,
}, out_path)

print(f"Saved external cohort to {out_path}")
print(f"  Total spots: {features_cat.shape[0]}")
print(f"  Feature dim: {features_cat.shape[1:]}")
print(f"  Count genes: {counts_cat.shape[1]}")
