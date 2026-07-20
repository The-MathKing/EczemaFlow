import torch
from torch.utils.data import Dataset
import numpy as np
import os
import scanpy as sc
import pandas as pd
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VisiumDataset(Dataset):
    """
    Real Visium spatial transcriptomics loader for EczemaFlow.
    Expects H5AD or SpatialData formats containing:
    1. Spot-level raw counts.
    2. Registered H&E image.
    3. Spatial coordinates for patch extraction.
    
    Implements grouped leave-patient-out splits to avoid spatial autocorrelation.
    """
    def __init__(self, data_path, split_manifest, fold='train', num_genes=500, patch_size=224, adata=None):
        super().__init__()
        self.data_path = data_path
        self.fold = fold
        self.patch_size = patch_size
        self.num_genes = num_genes
        h5ad_path = os.path.join(data_path, "GSE206391", "GSE206391_Preprocessed_data.h5")
        if adata is not None:
            self.adata = adata
        else:
            self.adata = sc.read_h5ad(h5ad_path)
        
        # 2. Parse patient-level fold assignments
        # Ensures that a patient's entire biological replicate set is kept in the same fold
        manifest = pd.read_csv(split_manifest)
        if isinstance(fold, list):
            fold_patients = manifest[manifest['fold_assignment'].isin(fold)]['patient_id'].unique()
        else:
            fold_patients = manifest[manifest['fold_assignment'] == fold]['patient_id'].unique()
        
        # Filter to the requested fold's patients
        patient_col = 'patient_id' if 'patient_id' in self.adata.obs else 'patient'
        fold_patients_str = [str(p) for p in fold_patients]
        self.adata = self.adata[self.adata.obs[patient_col].astype(str).isin(fold_patients_str)].copy()
        
        slide_col = 'slide_id' if 'slide_id' in self.adata.obs else 'sample'
        self.slides = self.adata.obs[slide_col].unique()
        self.num_samples = len(self.adata)
        print(f"Loaded VisiumDataset ({fold}) with {self.num_samples} spatial spots across {len(fold_patients)} patients.")
        
        # Setup lazy-loading image references
        self._load_image_references()
        
    def _load_image_references(self):
        self.slide_images = {}
        self.scales = {}
        
        scales_path = os.path.join(self.data_path, 'scales.json')
        if os.path.exists(scales_path):
            import json
            with open(scales_path, 'r') as f:
                self.scales = json.load(f)
                
        slide_col = 'slide_id' if 'slide_id' in self.adata.obs else 'sample'
        for slide_id in self.adata.obs[slide_col].unique():
            img_path = os.path.join(self.data_path, 'images', f"{slide_id}_HE.tif")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                self.slide_images[slide_id] = np.array(img)
                img.close()
            else:
                self.slide_images[slide_id] = None
                print(f"Warning: Missing H&E image for slide {slide_id}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Extract spot metadata
        obs = self.adata.obs.iloc[idx]
        slide_col = 'slide_id' if 'slide_id' in self.adata.obs else 'sample'
        slide_id = obs[slide_col]
        
        # High-res pixel coordinates
        if 'px_x' in obs and 'px_y' in obs:
            px, py = obs['px_x'], obs['px_y']
        elif 'spatial' in self.adata.obsm:
            px, py = self.adata.obsm['spatial'][idx]
        else:
            # Fallback to dummy or array_col/row
            px = obs.get('array_col', 50) * 10
            py = obs.get('array_row', 50) * 10
            
        # Scale coords down to match resized 2000px images
        if slide_id in self.scales:
            scale = self.scales[slide_id]
            px = px * scale['scale_x']
            py = py * scale['scale_y']
        
        # 2. Extract transcriptomics counts
        x_val = self.adata.X[idx]
        if hasattr(x_val, 'toarray'):
            counts = torch.tensor(x_val.toarray()[0], dtype=torch.float32)
        else:
            counts = torch.tensor(x_val, dtype=torch.float32)
        
        # 3. Extract high-resolution H&E patch
        img_array = self.slide_images.get(slide_id)
        if img_array is not None:
            # Recreate PIL Image from numpy array safely
            img = Image.fromarray(img_array)
            left = max(0, int(px - self.patch_size // 2))
            top = max(0, int(py - self.patch_size // 2))
            right = left + self.patch_size
            bottom = top + self.patch_size
            patch = img.crop((left, top, right, bottom))
        else:
            patch = Image.new('RGB', (self.patch_size, self.patch_size))
            
        # 4. Convert patch to normalized tensor
        patch_tensor = torch.from_numpy(np.array(patch)).permute(2, 0, 1).float() / 255.0
        
        # Expand to num_patches dimension (e.g. 4 patches per spot for ViT)
        # In a full pipeline, we would extract 4 adjacent tiles
        patches = patch_tensor.unsqueeze(0).expand(4, -1, -1, -1)
        coords = torch.tensor([px, py], dtype=torch.float32)
        
        return patches, counts, coords

class PrecomputedVisiumDataset(Dataset):
    """
    Lightweight dataset that loads pre-computed dense embeddings 
    instead of massive .tif images. Used for rapid Apple Silicon training.
    """
    def __init__(self, pt_path):
        super().__init__()
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Missing precomputed features at {pt_path}. Run scripts/precompute_features.py first.")
        
        data = torch.load(pt_path, map_location='cpu')
        self.features = data['features']
        self.counts = data['counts']
        self.coords = data['coords']
        self.num_samples = self.features.shape[0]
        print(f"Loaded PrecomputedVisiumDataset with {self.num_samples} spots from {pt_path}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.features[idx], self.counts[idx], self.coords[idx]
