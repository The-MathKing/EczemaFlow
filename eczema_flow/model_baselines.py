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

class Hist2STBaseline(nn.Module):
    """
    Baseline 2: Hist2ST surrogate model.
    Implements a CNN image encoder followed by a Transformer for spatial context,
    predicting ZINB parameters directly (non-flow).
    """
    def __init__(self, num_genes=500, cond_dim=256):
        super().__init__()
        self.encoder = MorphologyEncoder(embed_dim=cond_dim)
        self.transformer = nn.TransformerEncoderLayer(d_model=cond_dim, nhead=4, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(cond_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_genes)
        )

    def forward(self, patches):
        b, n, c, h, w = patches.shape
        patches_flat = patches.view(b * n, c, h, w)
        features = self.encoder(patches_flat) # (b*n, cond_dim)
        features = features.view(b, n, -1)
        
        # Spatial context via Transformer (proxy for spatial graphs in dense patches)
        features = self.transformer(features)
        features = features.mean(dim=1)
        
        preds = self.regressor(features)
        return preds
