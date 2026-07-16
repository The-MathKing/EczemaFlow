import torch
import torch.nn as nn
import torchvision.models as models

class MorphologyEncoder(nn.Module):
    """
    Extracts features from H&E WSI patches using a pre-trained vision backbone.
    """
    def __init__(self, embed_dim=256, use_pretrained=True):
        super().__init__()
        # Use ResNet50 as a feature extractor
        weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(2048, embed_dim)

    def forward(self, patches):
        """
        patches: (batch_size * num_patches, channels, height, width)
        """
        features = self.backbone(patches) # (B*N, 2048, 1, 1)
        features = features.view(features.size(0), -1) # (B*N, 2048)
        features = self.projection(features) # (B*N, embed_dim)
        return features

class ManyBodyAttention(nn.Module):
    """
    Evaluates sets of morphological features simultaneously using self-attention
    to generate a high-order contextual embedding representing the multi-cell microenvironment.
    """
    def __init__(self, embed_dim=256, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Standard Transformer Encoder to model interactions between multiple patches
        # belonging to the same spatial neighborhood
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # CLS token to aggregate the global neighborhood representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, patch_features, num_patches_per_spot):
        """
        patch_features: (batch_size * num_patches, embed_dim)
        Returns:
            contextual_embeds: (batch_size, embed_dim)
        """
        # Reshape to (batch_size, num_patches, embed_dim)
        batch_size = patch_features.size(0) // num_patches_per_spot
        x = patch_features.view(batch_size, num_patches_per_spot, self.embed_dim)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, N+1, embed_dim)
        
        # Apply many-body attention
        out = self.transformer(x) # (B, N+1, embed_dim)
        
        # Extract the contextual embedding from the CLS token position
        contextual_embeds = out[:, 0, :] # (B, embed_dim)
        return contextual_embeds

class ConditioningNetwork(nn.Module):
    """
    Combines the Morphology Encoder and Many-Body Attention module.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.encoder = MorphologyEncoder(embed_dim=embed_dim)
        self.attention = ManyBodyAttention(embed_dim=embed_dim)
        
    def forward(self, patches):
        """
        patches: (batch_size, num_patches, channels, height, width)
        """
        b, n, c, h, w = patches.shape
        # Flatten patches for CNN
        patches_flat = patches.view(b * n, c, h, w)
        
        # Extract features
        patch_features = self.encoder(patches_flat)
        
        # Apply Many-Body Attention
        contextual_embeds = self.attention(patch_features, num_patches_per_spot=n)
        return contextual_embeds
