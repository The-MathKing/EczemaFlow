import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr

from eczema_flow.model import EczemaFlowModel

# Ablation 1: No Context (Simple Mean Pooling instead of ViT+TDA)
class EczemaFlowNoTopology(EczemaFlowModel):
    def compute_loss(self, patches, target_counts, coords=None, is_precomputed=True):
        if is_precomputed:
            patches_no_topo = patches.clone()
            patches_no_topo[:, :, 256:] = 0.0
            return super().compute_loss(patches_no_topo, target_counts, coords, is_precomputed)
        else:
            return super().compute_loss(patches, target_counts, coords, is_precomputed)
            
    def sample(self, patches, coords=None, num_steps=20, is_precomputed=True):
        if is_precomputed:
            patches_no_topo = patches.clone()
            patches_no_topo[:, :, 256:] = 0.0
            return super().sample(patches_no_topo, coords, num_steps, is_precomputed)
        else:
            return super().sample(patches, coords, num_steps, is_precomputed)

class PrecomputedExternalDataset(Dataset):
    def __init__(self):
        path = "data/precomputed_gse197023/external_cohort.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing precomputed file: {path}")
        data = torch.load(path, map_location='cpu', weights_only=True)
        self.features = data['features']
        self.counts = data['counts']
        self.coords = data['coords']
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.counts[idx], self.coords[idx]

def main():
    device = torch.device('cpu')
    torch.set_num_threads(8)
    print(f"Executing ZERO-SHOT EXTERNAL VALIDATION on GSE197023...", flush=True)
    
    num_genes = 500
    
    test_ds = PrecomputedExternalDataset()
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    model = EczemaFlowNoTopology(num_genes=num_genes, num_experts=4, device=device).to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for b_feats, b_counts, b_coords in test_loader:
            b_feats = b_feats.to(device)
            b_counts = b_counts.to(device)
            b_coords = b_coords.to(device)
            
            target_log = torch.log1p(b_counts)
            preds = model.sample(b_feats, coords=b_coords, num_steps=20, is_precomputed=True)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target_log.cpu().numpy())
            
    if len(all_preds) > 0:
        f_preds = np.concatenate(all_preds, axis=0)
        f_targets = np.concatenate(all_targets, axis=0)[:, :500]
        
        mse = np.mean((f_preds - f_targets)**2)
        
        f_pcc_list = []
        for g in range(f_preds.shape[1]):
            pred_g = f_preds[:, g]
            targ_g = f_targets[:, g]
            if np.std(pred_g) > 1e-8 and np.std(targ_g) > 1e-8:
                r, _ = pearsonr(pred_g, targ_g)
                if not np.isnan(r): f_pcc_list.append(r)
        
        pcc = np.mean(f_pcc_list) if f_pcc_list else 0.0
        
        print(f"External Validation Results (GSE197023) - No Topology Flow:")
        print(f"MSE: {mse:.3f}")
        print(f"PCC: {pcc:.3f}")

if __name__ == "__main__":
    main()
