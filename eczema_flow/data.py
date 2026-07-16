import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.stats as stats

class MockSpatialDataset(Dataset):
    """
    Mock dataset generating paired H&E patches and Spatial Transcriptomics (ST) counts.
    Simulates the zero-inflated, overdispersed nature of scRNA/ST data.
    """
    def __init__(self, num_samples=1000, num_genes=500, patch_size=224, num_patches_per_spot=4):
        super().__init__()
        self.num_samples = num_samples
        self.num_genes = num_genes
        self.patch_size = patch_size
        self.num_patches_per_spot = num_patches_per_spot
        
        # Simulate ST data using Negative Binomial
        # r (dispersion) and p (probability)
        r = 2.0
        p = 0.5
        counts = np.random.negative_binomial(n=r, p=p, size=(num_samples, num_genes))
        
        # Zero inflation
        zero_prob = 0.6
        zero_mask = np.random.binomial(n=1, p=zero_prob, size=(num_samples, num_genes))
        counts = counts * (1 - zero_mask)
        
        self.st_counts = torch.tensor(counts, dtype=torch.float32)
        
        # Generate some mock spatial coordinates (x, y)
        self.coords = torch.rand(num_samples, 2) * 100.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate mock H&E patches for the neighborhood of this spot
        # Shape: (num_patches, channels, height, width)
        # In a real scenario, these would be loaded from a WSI using self.coords[idx]
        patches = torch.rand(self.num_patches_per_spot, 3, self.patch_size, self.patch_size)
        counts = self.st_counts[idx]
        coords = self.coords[idx]
        
        return patches, counts, coords

def get_dataloaders(batch_size=32, num_samples=1000, num_genes=500):
    dataset = MockSpatialDataset(num_samples=num_samples, num_genes=num_genes)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=7)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=7)
    
    return train_loader, val_loader
