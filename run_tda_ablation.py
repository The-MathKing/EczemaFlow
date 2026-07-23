import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr

import sys
sys.path.insert(0, '/Volumes/2TB/EczemaFlow-1')
from run_full_benchmarks import PrecomputedFoldDataset, train_and_eval_loocv
from eczema_flow.model import EczemaFlowModel

# -------------------------------------------------------
# Ablation: Density Only
# Zeros out all topological features except a single 1D feature 
# representing simple point density/count.
# -------------------------------------------------------
class EczemaFlowDensityOnly(EczemaFlowModel):
    def _modify(self, patches):
        p = patches.clone()
        density_proxy = p[:, :, 256:257]
        p[:, :, 256:] = 0.0
        p[:, :, 256:257] = density_proxy
        return p

    def compute_loss(self, patches, target_counts, coords=None, is_precomputed=True):
        return super().compute_loss(self._modify(patches), target_counts, coords, is_precomputed)
        
    def sample(self, patches, coords=None, num_steps=20, is_precomputed=True):
        return super().sample(self._modify(patches), coords, num_steps, is_precomputed)

# -------------------------------------------------------
# Ablation: H0 Only
# Zeros out H1 loops.
# -------------------------------------------------------
class EczemaFlowH0Only(EczemaFlowModel):
    def _modify(self, patches):
        p = patches.clone()
        p[:, :, 256+128:] = 0.0
        return p

    def compute_loss(self, patches, target_counts, coords=None, is_precomputed=True):
        return super().compute_loss(self._modify(patches), target_counts, coords, is_precomputed)
        
    def sample(self, patches, coords=None, num_steps=20, is_precomputed=True):
        return super().sample(self._modify(patches), coords, num_steps, is_precomputed)

# -------------------------------------------------------
# Ablation: H1 Only
# Zeros out H0 components.
# -------------------------------------------------------
class EczemaFlowH1Only(EczemaFlowModel):
    def _modify(self, patches):
        p = patches.clone()
        p[:, :, 256:256+128] = 0.0
        return p

    def compute_loss(self, patches, target_counts, coords=None, is_precomputed=True):
        return super().compute_loss(self._modify(patches), target_counts, coords, is_precomputed)
        
    def sample(self, patches, coords=None, num_steps=20, is_precomputed=True):
        return super().sample(self._modify(patches), coords, num_steps, is_precomputed)

def main():
    device = torch.device('cpu')
    torch.set_num_threads(8)
    print("Executing Topology Ablation Ladder...", flush=True)
    
    num_genes = 500
    epochs = 5
    results = {}
    
    kwargs = {'num_genes': num_genes, 'num_experts': 4, 'device': device}
    
    results['Density Only'] = train_and_eval_loocv("Density Only", EczemaFlowDensityOnly, kwargs, device, epochs)
    results['H0 Only'] = train_and_eval_loocv("H0 Only", EczemaFlowH0Only, kwargs, device, epochs)
    results['H1 Only'] = train_and_eval_loocv("H1 Only", EczemaFlowH1Only, kwargs, device, epochs)
    
    os.makedirs("results", exist_ok=True)
    with open("results/tda_ablation.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Ablation run complete. Results saved to results/tda_ablation.json")

if __name__ == "__main__":
    main()
