import torch
import torch.nn as nn
from .attention import MorphologyEncoder

class CNNRegressor(nn.Module):
    """
    Baseline 1: Deterministic CNN Regressor (similar to ST-Net).
    Directly maps WSI patches to Spatial Transcriptomics counts.
    """
    def __init__(self, num_genes=500, cond_dim=256):
        super().__init__()
        self.encoder = MorphologyEncoder(embed_dim=cond_dim)
        # Directly regress the gene expression values
        self.regressor = nn.Sequential(
            nn.Linear(cond_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_genes)
        )

    def forward(self, patches):
        b, n, c, h, w = patches.shape
        # Flatten and extract features
        patches_flat = patches.view(b * n, c, h, w)
        features = self.encoder(patches_flat) # (b*n, cond_dim)
        
        # Aggregate features across the n patches (mean pooling instead of ViT Contextual Encoder)
        features = features.view(b, n, -1).mean(dim=1)
        
        # Regress
        preds = self.regressor(features)
        return preds

class GaussianPrior(nn.Module):
    """
    Standard Gaussian noise prior for the ablation study.
    """
    def __init__(self, num_genes, device='cpu'):
        super().__init__()
        self.num_genes = num_genes
        self.device = device
        
    def sample(self, batch_size):
        return torch.randn(batch_size, self.num_genes, device=self.device)

# We will monkey-patch or subclass EczemaFlowModel in the benchmark script 
# to swap the prior for GaussianFlowModel.
