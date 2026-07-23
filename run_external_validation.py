import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr

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

# Ablation 2: Gaussian Flow Model
class GaussianFlowModel(EczemaFlowModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior = GaussianPrior(num_genes=self.num_genes, device=self.device)

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

def evaluate_model(model_name, model_class, kwargs, test_loader, device, is_cnn=False):
    model = model_class(**kwargs).to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_coverages = []
    
    print(f"Evaluating {model_name} on external dataset GSE197023...")
    with torch.no_grad():
        for b_feats, b_counts, b_coords in test_loader:
            b_feats = b_feats.to(device)
            b_counts = b_counts[:, :500].to(device)
            b_coords = b_coords.to(device)
            
            target_log = torch.log1p(b_counts)
            
            if is_cnn:
                preds = model(b_feats, is_precomputed=True)
            else:
                preds_samples = torch.stack([model.sample(b_feats, coords=b_coords, num_steps=20, is_precomputed=True) for _ in range(5)], dim=0)
                preds = preds_samples.mean(dim=0)
                preds_std = preds_samples.std(dim=0) + 1e-6
                lower = preds - 1.96 * preds_std
                upper = preds + 1.96 * preds_std
                in_bound = (target_log >= lower) & (target_log <= upper)
                batch_coverage = in_bound.float().mean().item()
                all_coverages.append(batch_coverage)
                
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
        cov = np.mean(all_coverages) if not is_cnn and all_coverages else 0.0
        
        print(f"Results - {model_name}: MSE={mse:.3f}, PCC={pcc:.3f}, Cov={cov:.3f}")
        return {"MSE": float(mse), "PCC": float(pcc), "Coverage": float(cov)}
    return {"MSE": 0.0, "PCC": 0.0, "Coverage": 0.0}

def main():
    device = torch.device('cpu')
    torch.set_num_threads(8)
    print(f"Executing ZERO-SHOT EXTERNAL VALIDATION on GSE197023...", flush=True)
    
    num_genes = 500
    
    test_ds = PrecomputedExternalDataset()
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    results = {}
    
    cnn_kwargs = {'num_genes': num_genes}
    results['CNN Regressor'] = evaluate_model("CNN Regressor", CNNRegressor, cnn_kwargs, test_loader, device, is_cnn=True)
    
    hist2st_kwargs = {'num_genes': num_genes}
    results['Hist2ST'] = evaluate_model("Hist2ST", Hist2STBaseline, hist2st_kwargs, test_loader, device, is_cnn=True)
    
    gf_kwargs = {'num_genes': num_genes, 'num_experts': 4, 'device': device}
    results['Gaussian Flow'] = evaluate_model("Gaussian Flow", GaussianFlowModel, gf_kwargs, test_loader, device)
    
    nt_kwargs = {'num_genes': num_genes, 'num_experts': 4, 'device': device}
    results['No Topology Flow'] = evaluate_model("No Topology Flow", EczemaFlowNoTopology, nt_kwargs, test_loader, device)
    
    full_kwargs = {'num_genes': num_genes, 'num_experts': 4, 'device': device}
    results['EczemaFlow (Full)'] = evaluate_model("EczemaFlow (Full)", EczemaFlowModel, full_kwargs, test_loader, device)
    
    os.makedirs("results", exist_ok=True)
    with open("results/external_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    print("External validation complete. Results saved to results/external_metrics.json")

if __name__ == "__main__":
    main()
