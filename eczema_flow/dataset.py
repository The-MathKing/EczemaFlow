import torch
from torch.utils.data import Dataset
import numpy as np
import os
import scanpy as sc
import pandas as pd
from PIL import Image

class VisiumDataset(Dataset):
    """
    Real Visium spatial transcriptomics loader for EczemaFlow.
    Expects H5AD or SpatialData formats containing:
    1. Spot-level raw counts.
    2. Registered H&E image.
    3. Spatial coordinates for patch extraction.
    
    Implements grouped leave-patient-out splits to avoid spatial autocorrelation.
    """
    def __init__(self, data_path, split_manifest, fold='train', patch_size=224, num_genes=500):
        super().__init__()
        self.data_path = data_path
        self.fold = fold
        self.patch_size = patch_size
        self.num_genes = num_genes
        
        # In a real implementation, this loads the specific slides assigned to this fold
        # based on patient-level grouping to ensure no data leakage across spots from the same patient.
        # e.g., if fold=='train', loads slides for 4 patients; if fold=='test', loads 1 held-out patient.
        
        h5ad_path = os.path.join(data_path, "GSE206391_spatial.h5ad")
        if not os.path.exists(h5ad_path):
            raise FileNotFoundError(f"Missing clinical dataset: {h5ad_path}. Please download GSE206391_spatial.h5ad to run the full pipeline.")
            
        # 1. Load clinical spatial transcriptomics
        self.adata = sc.read_h5ad(h5ad_path)
        
        # 2. Parse patient-level fold assignments
        # Ensures that a patient's entire biological replicate set is kept in the same fold
        manifest = pd.read_csv(split_manifest)
        fold_patients = manifest[manifest['fold_assignment'] == fold]['patient_id'].unique()
        
        # 3. Filter AnnData for this fold's patients
        self.adata = self.adata[self.adata.obs['patient_id'].isin(fold_patients)].copy()
        
        self.num_samples = len(self.adata)
        print(f"Loaded VisiumDataset ({fold}) with {self.num_samples} spatial spots across {len(fold_patients)} patients.")
        
        # Setup lazy-loading image references
        self._load_image_references()
        
    def _load_image_references(self):
        self.slide_images = {}
        for slide_id in self.adata.obs['slide_id'].unique():
            img_path = os.path.join(self.data_path, 'images', f"{slide_id}_HE.tif")
            if os.path.exists(img_path):
                self.slide_images[slide_id] = img_path
            else:
                raise FileNotFoundError(f"Missing H&E image for slide {slide_id}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Extract spot metadata
        obs = self.adata.obs.iloc[idx]
        slide_id = obs['slide_id']
        px, py = obs['px_x'], obs['px_y'] # High-res pixel coordinates
        
        # 2. Extract transcriptomics counts
        x_val = self.adata.X[idx]
        if hasattr(x_val, 'toarray'):
            counts = torch.tensor(x_val.toarray()[0], dtype=torch.float32)
        else:
            counts = torch.tensor(x_val, dtype=torch.float32)
        
        # 3. Extract high-resolution H&E patch
        img_path = self.slide_images[slide_id]
        with Image.open(img_path) as img:
            left = max(0, int(px - self.patch_size // 2))
            top = max(0, int(py - self.patch_size // 2))
            right = left + self.patch_size
            bottom = top + self.patch_size
            patch = img.crop((left, top, right, bottom))
            
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
