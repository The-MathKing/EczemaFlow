import torch
import torch.nn as nn
import numpy as np
from ripser import ripser
from persim import PersistenceImager

class TDAFeatureExtractor(nn.Module):
    """
    Topological Data Analysis (TDA) Feature Extractor.
    Extracts cell nuclei coordinates from H&E patches by intensity thresholding,
    constructs a Vietoris-Rips complex, and computes persistence landscapes (images).
    """
    def __init__(self, output_dim=64, pixels_per_patch=224):
        super().__init__()
        self.output_dim = output_dim
        # We use PersistenceImager to convert persistence diagrams to fixed-size vectors
        self.pimgr = PersistenceImager(pixel_size=0.1)
        self.pimgr.fit([np.array([[0, 1]])]) # Dummy fit to initialize bounds
        
        # Linear layer to project the flattened persistence image into embedding space
        # Persim defaults to 20x20 images for Betti-1 (400 dims) + Betti-0 (400 dims) = 800
        self.projection = nn.Linear(800, output_dim)
        
    def _extract_nuclei_points(self, patch_np, threshold=0.4):
        """ Extract mock nuclei coordinates from a single patch using intensity thresholding. """
        # patch_np: (C, H, W). Convert to grayscale.
        gray = np.mean(patch_np, axis=0)
        # Nuclei are typically dark (low intensity)
        y, x = np.where(gray < threshold)
        points = np.column_stack((x, y))
        return points

    def _compute_persistence_image(self, points):
        """ Compute Vietoris-Rips persistence diagram and convert to image. """
        if len(points) < 3:
            return np.zeros(800)
            
        try:
            diagrams = ripser(points, maxdim=1)['dgms']
            # diagrams[0] is Betti-0, diagrams[1] is Betti-1
            dgm0 = diagrams[0][:-1] # Remove the infinite point
            dgm1 = diagrams[1]
            
            img0 = self.pimgr.transform(dgm0) if len(dgm0) > 0 else np.zeros((20, 20))
            img1 = self.pimgr.transform(dgm1) if len(dgm1) > 0 else np.zeros((20, 20))
            
            return np.concatenate([img0.flatten(), img1.flatten()])
        except Exception:
            return np.zeros(800)

    def forward(self, patches):
        """
        patches: (batch_size, num_patches, channels, height, width)
        Returns:
            tda_features: (batch_size, num_patches, output_dim)
        """
        b, n, c, h, w = patches.shape
        device = patches.device
        
        # This is a non-differentiable CPU extraction step
        patches_np = patches.detach().cpu().numpy()
        
        all_features = []
        for i in range(b):
            batch_features = []
            for j in range(n):
                pts = self._extract_nuclei_points(patches_np[i, j])
                p_img = self._compute_persistence_image(pts)
                batch_features.append(p_img)
            all_features.append(batch_features)
            
        tda_tensor = torch.tensor(np.array(all_features), dtype=torch.float32, device=device) # (b, n, 800)
        
        # Project differentiable
        tda_embedded = self.projection(tda_tensor) # (b, n, output_dim)
        return tda_embedded
