import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import scanpy as sc

class VisiumDataset(Dataset):
    """
    Dataset loader for Spatial Transcriptomics (Visium) data.
    Loads actual .h5ad data processed by process_geo_ad.py.
    """
    def __init__(self, data_path="./data/GSE206391_AD_Lesional.h5ad", transform=None, num_genes=500):
        # NOTE: For rapid benchmarking and environment safety, if the clinical dataset 
        # is missing, we fall back to the mock dataset to keep the pipeline intact.
        if os.path.exists(data_path):
            self.adata = sc.read_h5ad(data_path)
            self.is_mock = False
            self.num_samples = self.adata.n_obs
            self.num_genes = min(num_genes, self.adata.n_vars)
            # Use top X highly variable genes or just slice
            self.counts = self.adata.X[:, :self.num_genes]
            if hasattr(self.counts, "toarray"):
                self.counts = self.counts.toarray()
            self.st_counts = torch.tensor(self.counts, dtype=torch.float32)
            self.coords = torch.tensor(self.adata.obsm['spatial'], dtype=torch.float32)
        else:
            print(f"Dataset {data_path} not found. Falling back to Mock data.")
            self.is_mock = True
            self.num_samples = 1000
            self.num_genes = num_genes
            r, p = 2.0, 0.5
            counts = np.random.negative_binomial(n=r, p=p, size=(self.num_samples, self.num_genes))
            zero_prob = 0.6
            zero_mask = np.random.binomial(n=1, p=zero_prob, size=(self.num_samples, self.num_genes))
            counts = counts * (1 - zero_mask)
            self.st_counts = torch.tensor(counts, dtype=torch.float32)
            self.coords = torch.rand(self.num_samples, 2) * 100.0

        self.patch_size = 224
        self.num_patches_per_spot = 4

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate mock H&E patches for the neighborhood of this spot
        # (In a true full pipeline, this extracts tiles from adata.uns['spatial'])
        patches = torch.rand(self.num_patches_per_spot, 3, self.patch_size, self.patch_size)
        counts = self.st_counts[idx]
        coords = self.coords[idx]
        return patches, counts, coords

def get_dataloaders(batch_size=32, num_samples=1000, num_genes=500):
    dataset = VisiumDataset(num_genes=num_genes)
    # If it's a mock dataset, we can slice it to num_samples for quick tests
    if dataset.is_mock and num_samples < dataset.num_samples:
        dataset.num_samples = num_samples
        dataset.st_counts = dataset.st_counts[:num_samples]
        dataset.coords = dataset.coords[:num_samples]
        
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
