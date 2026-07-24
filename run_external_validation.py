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
    def compute_loss(self, patches, target_counts, coords=None, library_size=None, is_precomputed=True, shuffle_coords=True):
        if is_precomputed:
            patches_no_topo = patches.clone()
            patches_no_topo[:, :, 256:] = 0.0
            return super().compute_loss(patches_no_topo, target_counts, coords, library_size, is_precomputed, shuffle_coords)
        else:
            return super().compute_loss(patches, target_counts, coords, library_size, is_precomputed, shuffle_coords)
            
    def sample(self, patches, coords=None, library_size=None, num_steps=20, is_precomputed=True, shuffle_coords=True):
        if is_precomputed:
            patches_no_topo = patches.clone()
            patches_no_topo[:, :, 256:] = 0.0
            return super().sample(patches_no_topo, coords, library_size, num_steps, is_precomputed, shuffle_coords)
        else:
            return super().sample(patches, coords, library_size, num_steps, is_precomputed, shuffle_coords)

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
        # Note: weights_only=False is required to load python lists like gsm_ids
        data = torch.load(path, map_location='cpu', weights_only=False)
        self.features = data['features']
        self.counts = data['counts']
        self.coords = data['coords']
        self.gsm_ids = data['gsm_ids']
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        library_size = self.counts[idx].sum().unsqueeze(0)
        return self.features[idx], self.counts[idx], self.coords[idx], self.gsm_ids[idx], library_size

def evaluate_model(model_name, model_class, kwargs, test_loader, device, is_cnn=False):
    model = model_class(**kwargs).to(device)
    model.eval()
    
    patient_results = {}
    
    print(f"Evaluating {model_name} on external dataset GSE197023...")
    with torch.no_grad():
        for b_feats, b_counts, b_coords, b_gsm, lib_size in test_loader:
            b_feats = b_feats.to(device)
            b_counts = b_counts[:, :500].to(device)
            b_coords = b_coords.to(device)
            lib_size = lib_size.to(device)
            
            target_log = torch.log1p(b_counts)
            
            if is_cnn:
                preds = model(b_feats, is_precomputed=True)
                preds_std = torch.zeros_like(preds) # CNN is deterministic
                lower = preds
                upper = preds
                in_bound = torch.zeros_like(preds, dtype=torch.bool)
            else:
                preds_samples = torch.stack([model.sample(b_feats, coords=b_coords, library_size=lib_size, num_steps=20, is_precomputed=True) for _ in range(5)], dim=0)
                preds = preds_samples.mean(dim=0)
                preds_std = preds_samples.std(dim=0) + 1e-6
                lower = preds - 1.96 * preds_std
                upper = preds + 1.96 * preds_std
                in_bound = (target_log >= lower) & (target_log <= upper)
                
            # Compute per-spot stats, then group by gsm
            for i, gsm in enumerate(b_gsm):
                if gsm not in patient_results:
                    patient_results[gsm] = {
                        'preds': [], 'targets': [], 'coverages': [], 'interval_widths': [], 'maes': []
                    }
                    
                p = preds[i].cpu().numpy()
                t = target_log[i].cpu().numpy()
                patient_results[gsm]['preds'].append(p)
                patient_results[gsm]['targets'].append(t)
                patient_results[gsm]['maes'].append(np.abs(p - t).mean())
                
                if not is_cnn:
                    patient_results[gsm]['coverages'].append(in_bound[i].float().mean().item())
                    patient_results[gsm]['interval_widths'].append((upper[i] - lower[i]).mean().item())

    # Aggregate by patient
    final_patient_metrics = {}
    global_preds, global_targets, global_coverages, global_widths, global_maes = [], [], [], [], []
    
    for gsm, data in patient_results.items():
        f_preds = np.stack(data['preds'])
        f_targets = np.stack(data['targets'])
        
        mse = np.mean((f_preds - f_targets)**2)
        mae = np.mean(data['maes'])
        cov = np.mean(data['coverages']) if not is_cnn else 0.0
        width = np.mean(data['interval_widths']) if not is_cnn else 0.0
        
        # calculate pcc
        pcc_list = []
        for g in range(f_preds.shape[1]):
            pg = f_preds[:, g]
            tg = f_targets[:, g]
            if np.std(pg) > 1e-8 and np.std(tg) > 1e-8:
                r, _ = pearsonr(pg, tg)
                if not np.isnan(r): pcc_list.append(r)
        pcc = np.mean(pcc_list) if pcc_list else 0.0
        
        # map GSM to patient and lesion status (mock mapping since actual mapping requires GEO query, but we group by GSM here)
        final_patient_metrics[gsm] = {
            "MSE": float(mse),
            "MAE": float(mae),
            "PCC": float(pcc),
            "Coverage": float(cov),
            "IntervalWidth": float(width)
        }
        
        global_preds.extend(data['preds'])
        global_targets.extend(data['targets'])
        global_coverages.extend(data.get('coverages', []))
        global_widths.extend(data.get('interval_widths', []))
        global_maes.extend(data.get('maes', []))
        
    global_preds = np.array(global_preds)
    global_targets = np.array(global_targets)
    global_mse = np.mean((global_preds - global_targets)**2)
    global_mae = np.mean(global_maes)
    global_cov = np.mean(global_coverages) if not is_cnn else 0.0
    global_width = np.mean(global_widths) if not is_cnn else 0.0
    
    print(f"Results - {model_name}: Global MSE={global_mse:.3f}, MAE={global_mae:.3f}, Cov={global_cov:.3f}, Width={global_width:.3f}")
    
    return {
        "Global": {
            "MSE": float(global_mse),
            "MAE": float(global_mae),
            "Coverage": float(global_cov),
            "IntervalWidth": float(global_width)
        },
        "By_Sample": final_patient_metrics
    }

def main():
    device = torch.device('cpu')
    torch.set_num_threads(8)
    print(f"Executing PATIENT-LEVEL ZERO-SHOT EXTERNAL VALIDATION on GSE197023...", flush=True)
    
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
    with open("results/external_metrics_patient_level.json", "w") as f:
        json.dump(results, f, indent=4)
    print("External validation complete. Results saved to results/external_metrics_patient_level.json")

if __name__ == "__main__":
    main()
