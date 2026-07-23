import torch
import torch.nn as nn
import torchvision.models as models

class MorphologyEncoder(nn.Module):
    """
    Extracts features from H&E WSI patches using a pre-trained vision backbone.
    """
    def __init__(self, embed_dim=256, use_pretrained=True):
        super().__init__()
        # Use ViT_B_16 as the image encoder, as claimed in the manuscript
        weights = models.ViT_B_16_Weights.DEFAULT if use_pretrained else None
        self.vit = models.vit_b_16(weights=weights)
        
        # We override the final classification head to output our desired embedding dimension
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, embed_dim)
        
        # Freeze the transformer blocks to save compute time and memory, but leave the head trainable
        for name, param in self.vit.named_parameters():
            if not name.startswith("heads.head"):
                param.requires_grad = False

    def forward(self, patches):
        """
        patches: (batch_size * num_patches, channels, height, width)
        """
        # vit_b_16 requires 224x224 input
        features = self.vit(patches) # (B*N, embed_dim)
        return features

class ViTContextualEncoder(nn.Module):
    """
    Evaluates sets of morphological features simultaneously using self-attention
    to generate a high-order contextual embedding representing the multi-cell microenvironment.
    """
    def __init__(self, embed_dim=256, num_heads=4, num_layers=1, dropout=0.1):
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
        
        # Apply ViT self-attention
        out = self.transformer(x) # (B, N+1, embed_dim)
        
        # Extract the contextual embedding from the CLS token position
        contextual_embeds = out[:, 0, :] # (B, embed_dim)
        return contextual_embeds

class SpatialFourierEncoder(nn.Module):
    """
    Maps 2D spatial coordinates (x, y) to a high-dimensional continuous feature space 
    using Fourier Features to allow the flow-matching model to become spatially-aware.
    """
    def __init__(self, embed_dim=32, scale=1.0):
        super().__init__()
        # We project 2D coords to embed_dim // 2 dimensions
        self.B = nn.Parameter(torch.randn(2, embed_dim // 2) * scale, requires_grad=False)
        
    def forward(self, coords):
        """
        coords: (batch_size, 2)
        Returns: (batch_size, embed_dim)
        """
        # Normalize coordinates to roughly [0, 1] assuming max spatial size ~2000
        coords_norm = coords / 2000.0
        # (batch_size, 2) x (2, embed_dim // 2) -> (batch_size, embed_dim // 2)
        x_proj = (2 * torch.pi * coords_norm) @ self.B
        # Concat sin and cos
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out

class ConditioningNetwork(nn.Module):
    """
    Combines the Morphology Encoder, TDA Feature Extractor, Spatial Fourier Encoder,
    and ViT Contextual Encoder.
    """
    def __init__(self, embed_dim=256, tda_dim=256, spatial_dim=32):
        super().__init__()
        self.encoder = MorphologyEncoder(embed_dim=embed_dim)
        # We import TDA locally to avoid circular import if needed, or import at top
        from .tda import TDAFeatureExtractor
        self.tda = TDAFeatureExtractor(output_dim=tda_dim)
        self.spatial_encoder = SpatialFourierEncoder(embed_dim=spatial_dim)
        self.spatial_dim = spatial_dim
        
        # The ViT now takes the concatenated embed_dim + tda_dim + spatial_dim
        self.attention = ViTContextualEncoder(embed_dim=embed_dim + tda_dim + spatial_dim)
        
    def forward(self, patches, coords):
        """
        patches: (batch_size, num_patches, channels, height, width)
        coords: (batch_size, 2)
        """
        b, n, c, h, w = patches.shape
        # Flatten patches for CNN
        patches_flat = patches.view(b * n, c, h, w)
        
        # Extract features
        patch_features = self.encoder(patches_flat) # (b*n, embed_dim)
        
        # Extract TDA features
        tda_features = self.tda(patches) # (b, n, tda_dim)
        tda_features_flat = tda_features.view(b * n, -1)
        
        # Concatenate Morphology and TDA features
        combined_features = torch.cat([patch_features, tda_features_flat], dim=-1) # (b*n, embed_dim + tda_dim)
        
        # Extract Spatial Features
        spatial_features = self.spatial_encoder(coords) # (b, spatial_dim)
        # Expand spatial features to match each patch
        spatial_features = spatial_features.unsqueeze(1).expand(b, n, -1) # (b, n, spatial_dim)
        spatial_features_flat = spatial_features.reshape(b * n, -1)
        
        combined_features = torch.cat([combined_features, spatial_features_flat], dim=-1)
        
        # Apply ViT Contextual Encoder
        contextual_embeds = self.attention(combined_features, num_patches_per_spot=n)
        return contextual_embeds

    def forward_precomputed(self, combined_features, coords):
        """
        Bypasses the CNN and TDA extraction. Directly pipes the pre-computed 
        dense embeddings (b, n, embed_dim + tda_dim) + newly computed spatial features into the ViT.
        """
        b, n, d = combined_features.shape
        
        # Backward compatibility for precomputed features (which had tda_dim=64)
        # If the model is instantiated with tda_dim=256 (d=512) but precomputed is 320
        target_d = self.attention.embed_dim
        if d < target_d:
            cnn_feats = combined_features[:, :, :256]
            tda_feats = combined_features[:, :, 256:]
            
            # Project the 64-dim TDA features to the new tda_dim to maintain dimensionality
            if not hasattr(self, 'tda_adapter'):
                self.tda_adapter = nn.Linear(tda_feats.shape[-1], target_d - 256 - self.spatial_dim).to(combined_features.device)
            tda_feats = self.tda_adapter(tda_feats)
            combined_features = torch.cat([cnn_feats, tda_feats], dim=-1)
            d = target_d - self.spatial_dim
            
        # Add Spatial Features
        spatial_features = self.spatial_encoder(coords) # (b, spatial_dim)
        spatial_features = spatial_features.unsqueeze(1).expand(b, n, -1)
        
        # combine_features is (b, n, d)
        combined_features = torch.cat([combined_features, spatial_features], dim=-1)
        
        d = combined_features.shape[-1]
        combined_features_flat = combined_features.view(b * n, d)
        contextual_embeds = self.attention(combined_features_flat, num_patches_per_spot=n)
        return contextual_embeds

