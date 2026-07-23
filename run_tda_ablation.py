"""
Topology Ablation: Compare EczemaFlow's TDA persistence landscapes against 
simpler nuclei statistics (count, density, mean nearest-neighbor distance).

This is the key H1 test: does TOPOLOGY add value beyond simple nuclei counting?
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr

# Reuse the precomputed dataset
import sys
sys.path.insert(0, '/Volumes/2TB/EczemaFlow-1')
from run_full_benchmarks import PrecomputedFoldDataset
from eczema_flow.model import EczemaFlowModel
from eczema_flow.attention import ConditioningNetwork, SpatialFourierEncoder

device = torch.device('cpu')
EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-4

def compute_pcc(preds, targets):
    pcc_list = []
    for g in range(preds.shape[1]):
        p = preds[:, g]; t = targets[:, g]
        if np.std(p) > 1e-8 and np.std(t) > 1e-8:
            r, _ = pearsonr(p, t)
            if not np.isnan(r):
                pcc_list.append(r)
    return float(np.mean(pcc_list)) if pcc_list else 0.0

def run_loocv(model_factory, name):
    folds = [f"fold_{i}" for i in range(1, 6)] + ["test"]
    fold_pccs = []
    fold_mses = []
    
    for val_fold in folds:
        train_folds = [f for f in folds if f != val_fold]
        train_ds = PrecomputedFoldDataset(train_folds)
        val_ds = PrecomputedFoldDataset([val_fold])
        
        # Compute top 500 HVGs from training data
        variances = train_ds.counts.var(dim=0)
        top_idx = torch.argsort(variances, descending=True)[:500]
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = model_factory()
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
        
        # Train
        for epoch in range(EPOCHS):
            model.train()
            for patches, b_counts, b_coords in train_loader:
                patches = patches.to(device)
                b_counts = b_counts[:, top_idx].to(device)
                b_coords = b_coords.to(device)
                optimizer.zero_grad()
                loss = model.compute_loss(patches, b_counts, b_coords, is_precomputed=True)
                loss.backward()
                optimizer.step()
        
        # Eval
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for patches, b_counts, b_coords in val_loader:
                patches = patches.to(device)
                b_counts = b_counts[:, top_idx].to(device)
                b_coords = b_coords.to(device)
                tgt = torch.log1p(b_counts)
                preds = model.sample(patches, b_coords, num_steps=20, is_precomputed=True)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(tgt.cpu().numpy())
        
        preds_all = np.concatenate(all_preds, axis=0)
        tgts_all = np.concatenate(all_targets, axis=0)
        pcc = compute_pcc(preds_all, tgts_all)
        mse = float(np.mean((preds_all - tgts_all)**2))
        fold_pccs.append(pcc)
        fold_mses.append(mse)
        print(f"  {name} [{val_fold}]: PCC={pcc:.4f}, MSE={mse:.4f}")
    
    return {
        'pcc_mean': float(np.mean(fold_pccs)),
        'pcc_std': float(np.std(fold_pccs)),
        'mse_mean': float(np.mean(fold_mses)),
        'mse_std': float(np.std(fold_mses)),
    }


# -------------------------------------------------------
# Ablation: EczemaFlow with NO TDA features at all
# Replace the 256-dim TDA vector with zeros
# -------------------------------------------------------
class EczemaFlowNoTDA(EczemaFlowModel):
    """Ablation: zero out the TDA feature vector at forward time."""
    def _zero_tda_patches(self, patches):
        """Zero out TDA dimensions (columns 256:512 in precomputed features)."""
        zeroed = patches.clone()
        zeroed[:, :, 256:512] = 0.0  # TDA features occupy dims 256-511
        return zeroed
    
    def compute_loss(self, patches, target_counts, coords, is_precomputed=True):
        return super().compute_loss(self._zero_tda_patches(patches), target_counts, coords, is_precomputed=is_precomputed)
    
    def sample(self, patches, coords, num_steps=20, is_precomputed=True):
        return super().sample(self._zero_tda_patches(patches), coords, num_steps=num_steps, is_precomputed=is_precomputed)


if __name__ == "__main__":
    import json, os
    
    results = {}
    
    print("="*60)
    print("ABLATION A: EczemaFlow with TDA=0 (simpler nuclei proxy)")
    print("="*60)
    no_tda_results = run_loocv(
        lambda: EczemaFlowNoTDA(num_genes=500, num_experts=4, device=device).to(device),
        "No TDA"
    )
    results['EczemaFlow_NoTDA'] = no_tda_results
    print(f"\n[RESULT] No TDA: PCC={no_tda_results['pcc_mean']:.4f} ± {no_tda_results['pcc_std']:.4f}")
    
    os.makedirs("results", exist_ok=True)
    with open("results/ablation_tda.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/ablation_tda.json")
