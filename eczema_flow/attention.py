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
        
        # Apply ViT self-attention
        out = self.transformer(x) # (B, N+1, embed_dim)
        
        # Extract the contextual embedding from the CLS token position
        contextual_embeds = out[:, 0, :] # (B, embed_dim)
        return contextual_embeds

class ConditioningNetwork(nn.Module):
    """
    Combines the Morphology Encoder, TDA Feature Extractor, and ViT Contextual Encoder.
    """
    def __init__(self, embed_dim=256, tda_dim=64):
        super().__init__()
        self.encoder = MorphologyEncoder(embed_dim=embed_dim)
        # We import TDA locally to avoid circular import if needed, or import at top
        from .tda import TDAFeatureExtractor
        self.tda = TDAFeatureExtractor(output_dim=tda_dim)
        
        # The ViT now takes the concatenated embed_dim + tda_dim
        self.attention = ViTContextualEncoder(embed_dim=embed_dim + tda_dim)
        
    def forward(self, patches):
        """
        patches: (batch_size, num_patches, channels, height, width)
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
        combined_features = torch.cat([patch_features, tda_features_flat], dim=-1)
        
        # Apply ViT Contextual Encoder
        contextual_embeds = self.attention(combined_features, num_patches_per_spot=n)
        return contextual_embeds

    def forward_precomputed(self, combined_features):
        """
        Bypasses the CNN and TDA extraction. Directly pipes the pre-computed 
        dense embeddings (b, n, embed_dim + tda_dim) into the ViT.
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
                self.tda_adapter = nn.Linear(tda_feats.shape[-1], target_d - 256).to(combined_features.device)
            tda_feats = self.tda_adapter(tda_feats)
            combined_features = torch.cat([cnn_feats, tda_feats], dim=-1)
            d = target_d
            
        combined_features_flat = combined_features.view(b * n, d)
        contextual_embeds = self.attention(combined_features_flat, num_patches_per_spot=n)
        return contextual_embeds

