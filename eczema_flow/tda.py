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
        self.pimgr = PersistenceImager(pixel_size=10.0)
        self.pimgr.birth_range = (0.0, 200.0)
        self.pimgr.pers_range = (0.0, 200.0)
        
        # Linear layer to project the flattened persistence image into embedding space
        # Persim defaults to 20x20 images for Betti-1 (400 dims) + Betti-0 (400 dims) = 800
        self.projection = nn.Linear(800, output_dim)
        
        self._stardist_initialized = False
        self.stardist_model = None
        
    def _extract_nuclei_points(self, patch_np):
        """ Extract nuclei coordinates from a single patch using pre-trained StarDist2D. """
        if not self._stardist_initialized:
            try:
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force TF to CPU to avoid MPS conflict
                from stardist.models import StarDist2D
                self.stardist_model = StarDist2D.from_pretrained('2D_versatile_he')
            except ImportError:
                self.stardist_model = None
                print("Warning: stardist not installed. TDA extraction will fail.")
            self._stardist_initialized = True
            
        if self.stardist_model is None:
            return np.zeros((0, 2))
            
        # Ensure patch is (H, W, C) for StarDist
        if patch_np.shape[0] == 3:
            patch_hwc = np.transpose(patch_np, (1, 2, 0))
        else:
            patch_hwc = patch_np
            
        try:
            labels, details = self.stardist_model.predict_instances(patch_hwc)
            points_yx = details['points']
            if len(points_yx) == 0:
                return np.zeros((0, 2))
            # Return as (x, y)
            return np.array([[p[1], p[0]] for p in points_yx])
        except Exception:
            return np.zeros((0, 2))

    def _compute_persistence_image(self, points):
        """ Compute Vietoris-Rips persistence diagram and convert to image. """
        if len(points) < 3:
            return np.zeros(800)
            
        try:
            diagrams = ripser(points, maxdim=1)['dgms']
            dgm0 = diagrams[0][:-1]
            dgm1 = diagrams[1]
            
            img0 = self.pimgr.transform(dgm0) if len(dgm0) > 0 else np.zeros((20, 20))
            img1 = self.pimgr.transform(dgm1) if len(dgm1) > 0 else np.zeros((20, 20))
            
            res = np.concatenate([np.array(img0).flatten(), np.array(img1).flatten()])
            if res.shape != (800,):
                return np.zeros(800)
            return res
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
