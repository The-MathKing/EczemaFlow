import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr
import scanpy as sc
import json

from eczema_flow.model import EczemaFlowModel
from eczema_flow.model_baselines import CNNRegressor, Hist2STBaseline, GaussianPrior

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

# Ablation: Random Capacity-Matched Features
class EczemaFlowRandomTopology(EczemaFlowModel):
    def compute_loss(self, patches, target_counts, coords=None, is_precomputed=True):
        if is_precomputed:
            patches_rand = patches.clone()
            patches_rand[:, :, 256:] = torch.randn_like(patches_rand[:, :, 256:])
            return super().compute_loss(patches_rand, target_counts, coords, is_precomputed)
            
    def sample(self, patches, coords=None, num_steps=20, is_precomputed=True):
        if is_precomputed:
            patches_rand = patches.clone()
            patches_rand[:, :, 256:] = torch.randn_like(patches_rand[:, :, 256:])
            return super().sample(patches_rand, coords, num_steps, is_precomputed)

# Ablation: Shuffled Coordinates
class EczemaFlowShuffledCoords(EczemaFlowModel):
    def compute_loss(self, patches, target_counts, coords=None, is_precomputed=True):
        shuffled_coords = coords[torch.randperm(coords.size(0))] if coords is not None else None
        return super().compute_loss(patches, target_counts, shuffled_coords, is_precomputed)
            
    def sample(self, patches, coords=None, num_steps=20, is_precomputed=True):
        shuffled_coords = coords[torch.randperm(coords.size(0))] if coords is not None else None
        return super().sample(patches, shuffled_coords, num_steps, is_precomputed)


# Ablation 2: Gaussian Flow Model
class GaussianFlowModel(EczemaFlowModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior = GaussianPrior(num_genes=self.num_genes, device=self.device)

def compute_mmd(x, y, sigma=1.0):
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    xx = torch.matmul(x, x.t())
    yy = torch.matmul(y, y.t())
    xy = torch.matmul(x, y.t())
    rx = (x**2).sum(dim=1).unsqueeze(1).expand_as(xx)
    ry = (y**2).sum(dim=1).unsqueeze(1).expand_as(yy)
    K_xx = torch.exp(- (rx.t() + rx - 2*xx) / (2*sigma**2))
    K_yy = torch.exp(- (ry.t() + ry - 2*yy) / (2*sigma**2))
    K_xy = torch.exp(- (rx.t() + ry - 2*xy) / (2*sigma**2))
    return (K_xx.mean() + K_yy.mean() - 2*K_xy.mean()).item()

class PrecomputedFoldDataset(Dataset):
    def __init__(self, folds):
        all_feats = []
        all_counts = []
        all_coords = []
        for fold in folds:
            path = f"data/precomputed/{fold}_features.pt"
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing precomputed file: {path}")
            data = torch.load(path, map_location='cpu', weights_only=True)
            all_feats.append(data['features'])
            all_counts.append(data['counts'])
            all_coords.append(data['coords'])
            
        self.features = torch.cat(all_feats, dim=0)
        self.counts = torch.cat(all_counts, dim=0)
        self.coords = torch.cat(all_coords, dim=0)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.counts[idx], self.coords[idx]

def train_and_eval_loocv(model_name, model_class, kwargs, device, epochs=1, is_cnn=False):
    print(f"\n--- Starting 6-Fold LOOCV for {model_name} ---")
    all_preds = []
    all_targets = []
    all_coords = []
    fold_mses = []
    fold_pccs = []
    fold_mmds = []
    fold_coverages = []
    
    folds = [f"fold_{i}" for i in range(1, 6)] + ["test"]
    
    for val_fold in folds:
        train_folds = [f for f in folds if f != val_fold]
        all_preds = []
        all_targets = []
        
        train_dataset = PrecomputedFoldDataset(train_folds)
        val_dataset = PrecomputedFoldDataset([val_fold])
        
        all_train_counts = train_dataset.counts
        variances = all_train_counts.var(dim=0)
        top_500_idx = torch.argsort(variances, descending=True)[:500]
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        model = model_class(**kwargs).to(device)
            
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        model.train()
        
        print(f"Training {model_name} on {val_fold} (held out)...", flush=True)
        for epoch in range(epochs):
            for patches, b_counts, b_coords in train_loader:
                patches = patches.to(device)
                b_counts = b_counts[:, top_500_idx].to(device)
                b_coords = b_coords.to(device)
                
                optimizer.zero_grad()
                if is_cnn:
                    preds = model(patches, is_precomputed=True)
                    target_log = torch.log1p(b_counts)
                    loss = F.mse_loss(preds, target_log)
                else:
                    loss = model.compute_loss(patches, b_counts, b_coords, is_precomputed=True)
                loss.backward()
                optimizer.step()
                
        model.eval()
        all_coverages = []
        with torch.no_grad():
            for patches, b_counts, b_coords in val_loader:
                patches = patches.to(device)
                b_counts = b_counts[:, top_500_idx].to(device)
                b_coords = b_coords.to(device)
                
                target_log = torch.log1p(b_counts)
                if is_cnn:
                    preds = model(patches, is_precomputed=True)
                    batch_coverage = 0.0
                else:
                    preds_samples = torch.stack([model.sample(patches, b_coords, num_steps=10, is_precomputed=True) for _ in range(5)], dim=0)
                    preds = preds_samples.mean(dim=0)
                    preds_std = preds_samples.std(dim=0) + 1e-6
                    lower = preds - 1.96 * preds_std
                    upper = preds + 1.96 * preds_std
                    in_bound = (target_log >= lower) & (target_log <= upper)
                    batch_coverage = in_bound.float().mean().item()
                    all_coverages.append(batch_coverage)
                    
                all_preds.append(preds.cpu().numpy())
                all_targets.append(target_log.cpu().numpy())
                all_coords.append(b_coords.cpu().numpy())
                
        if len(all_preds) > 0:
            f_preds = np.concatenate(all_preds, axis=0)
            f_targets = np.concatenate(all_targets, axis=0)
            fold_mses.append(np.mean((f_preds - f_targets)**2))
            if not is_cnn and len(all_coverages) > 0:
                fold_coverages.append(np.mean(all_coverages))
            
            f_pcc_list = []
            for g in range(f_preds.shape[1]):
                pred_g = f_preds[:, g]
                targ_g = f_targets[:, g]
                if np.std(pred_g) > 1e-8 and np.std(targ_g) > 1e-8:
                    r, _ = pearsonr(pred_g, targ_g)
                    if not np.isnan(r): f_pcc_list.append(r)
            fold_pccs.append(np.mean(f_pcc_list) if f_pcc_list else 0.0)
            fold_mmds.append(compute_mmd(torch.tensor(f_preds), torch.tensor(f_targets)))
            
            os.makedirs("results", exist_ok=True)
            np.save(f"results/{model_name.replace(' ', '_')}_{val_fold}_preds.npy", f_preds)
            np.save(f"results/{model_name.replace(' ', '_')}_{val_fold}_targets.npy", f_targets)
            
    if len(fold_mses) == 0:
        return 0, 0, 0
        
    mse = float(np.mean(fold_mses))
    mse_std = float(np.std(fold_mses))
    pcc = float(np.mean(fold_pccs))
    pcc_std = float(np.std(fold_pccs))
    mmd = float(np.mean(fold_mmds))
    mmd_std = float(np.std(fold_mmds))
    
    mse_ci = 1.96 * mse_std / np.sqrt(len(fold_mses))
    pcc_ci = 1.96 * pcc_std / np.sqrt(len(fold_pccs))
    
    cov = float(np.mean(fold_coverages)) if not is_cnn and len(fold_coverages) > 0 else 0.0
    
    print(f"LOOCV Results for {model_name}: MSE={mse:.3f}±{mse_ci:.3f}, PCC={pcc:.3f}±{pcc_ci:.3f}, MMD={mmd:.3f}, Cov={cov:.3f}")
    return {"MSE": mse, "MSE_std": mse_std, "MSE_CI_95": float(mse_ci), 
            "PCC": pcc, "PCC_std": pcc_std, "PCC_CI_95": float(pcc_ci), 
            "MMD": mmd, "MMD_std": mmd_std, "Coverage": cov,
            "fold_mses": [float(x) for x in fold_mses],
            "fold_pccs": [float(x) for x in fold_pccs],
            "fold_coverages": [float(x) for x in fold_coverages]}

def main():
    device = torch.device('cpu')
    torch.set_num_threads(8)
    print(f"Executing PRECOMPUTED 8-CORE CPU empirical run on {device}...", flush=True)
    
    num_genes = 500
    epochs = 5
    
    results = {}
    
    # NEW ABLATIONS (Run these first so we don't have to wait for everything else)
    nt_kwargs = {'num_genes': num_genes, 'num_experts': 4, 'device': device}
    results['Random Topology Flow'] = train_and_eval_loocv("Random Topology Flow", EczemaFlowRandomTopology, nt_kwargs, device, epochs)
    results['Shuffled Coords Flow'] = train_and_eval_loocv("Shuffled Coords Flow", EczemaFlowShuffledCoords, nt_kwargs, device, epochs)
    
    with open("results/ablation_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    print("New ablations complete! Results saved to results/ablation_metrics.json")

if __name__ == "__main__":
    main()
