import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import scanpy as sc

from eczema_flow.model import EczemaFlowModel
from eczema_flow.model_baselines import CNNRegressor, Hist2STBaseline, GaussianPrior
from eczema_flow.dataset import VisiumDataset

# Ablation 1: No Context (Simple Mean Pooling instead of ViT+TDA)
class EczemaFlowNoAttention(EczemaFlowModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from eczema_flow.attention import MorphologyEncoder
        class MeanPoolConditioner(torch.nn.Module):
            def __init__(self, embed_dim):
                super().__init__()
                self.encoder = MorphologyEncoder(embed_dim=embed_dim)
            def forward(self, patches):
                b, n, c, h, w = patches.shape
                patches_flat = patches.view(b * n, c, h, w)
                features = self.encoder(patches_flat)
                return features.view(b, n, -1).mean(dim=1)
        self.conditioner = MeanPoolConditioner(embed_dim=kwargs.get('cond_dim', 256))

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

def train_and_eval_loocv(model_name, model_class, kwargs, adata, num_genes, device, epochs=1, is_cnn=False):
    print(f"\n--- Starting 6-Fold LOOCV for {model_name} ---")
    all_preds = []
    all_targets = []
    all_coords = []
    fold_mses = []
    fold_pccs = []
    fold_mmds = []
    
    folds = [f"fold_{i}" for i in range(1, 6)] + ["test"]
    
    for val_fold in folds:
        train_folds = [f for f in folds if f != val_fold]
        all_preds = []
        all_targets = []
        
        train_dataset = VisiumDataset(data_path="data", split_manifest="data/splits.csv", fold=train_folds, num_genes=num_genes, adata=adata)
        val_dataset = VisiumDataset(data_path="data", split_manifest="data/splits.csv", fold=[val_fold], num_genes=num_genes, adata=adata)
        
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        
        import gc
        gc.collect()
        
        model = model_class(**kwargs).to(device)
        
        if hasattr(model, 'prior') and hasattr(model.prior, 'fit_to_data'):
            model.prior.fit_to_data(train_dataset.adata)
            
        optimizer = optim.AdamW(model.parameters(), lr=2e-3)
        model.train()
        
        print(f"Training {model_name} on {val_fold} (held out)...")
        for epoch in range(epochs):
            for i, (patches, counts, coords) in enumerate(train_loader):
                patches = patches.to(device)
                counts = counts.to(device)
                
                optimizer.zero_grad()
                if is_cnn:
                    preds = model(patches)
                    target_log = torch.log1p(counts)
                    loss = F.mse_loss(preds, target_log)
                else:
                    loss = model.compute_loss(patches, counts)
                loss.backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            for i, (patches, counts, coords) in enumerate(val_loader):
                patches = patches.to(device)
                counts = counts.to(device)
                
                target_log = torch.log1p(counts)
                if is_cnn:
                    preds = model(patches)
                else:
                    preds = model.sample(patches, num_steps=20)
                    
                all_preds.append(preds.cpu().numpy())
                all_targets.append(target_log.cpu().numpy())
                all_coords.append(coords.cpu().numpy())
                
            del model
            del optimizer
            gc.collect()
            
        # Calculate per-fold metrics
        if len(all_preds) > 0:
            f_preds = np.concatenate(all_preds, axis=0)
            f_targets = np.concatenate(all_targets, axis=0)
            fold_mses.append(np.mean((f_preds - f_targets)**2))
            
            f_pcc_list = []
            for g in range(f_preds.shape[1]):
                pred_g = f_preds[:, g]
                targ_g = f_targets[:, g]
                if np.std(pred_g) > 1e-8 and np.std(targ_g) > 1e-8:
                    r, _ = pearsonr(pred_g, targ_g)
                    if not np.isnan(r): f_pcc_list.append(r)
            fold_pccs.append(np.mean(f_pcc_list) if f_pcc_list else 0.0)
            fold_mmds.append(compute_mmd(torch.tensor(f_preds), torch.tensor(f_targets)))
            
            # Save predictions for this fold
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
    
    # 95% Confidence Interval (approx)
    mse_ci = 1.96 * mse_std / np.sqrt(len(fold_mses))
    pcc_ci = 1.96 * pcc_std / np.sqrt(len(fold_pccs))
    
    print(f"LOOCV Results for {model_name}: MSE={mse:.3f}±{mse_ci:.3f}, PCC={pcc:.3f}±{pcc_ci:.3f}, MMD={mmd:.3f}")
    return {"MSE": mse, "MSE_std": mse_std, "MSE_CI_95": float(mse_ci), 
            "PCC": pcc, "PCC_std": pcc_std, "PCC_CI_95": float(pcc_ci), 
            "MMD": mmd, "MMD_std": mmd_std}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Executing traceable empirical run on {device}...")
    
    num_genes = 500
    epochs = 50
    
    print("Loading transcriptomic data matrix...")
    adata = sc.read_h5ad("data/GSE206391_spatial.h5ad")
    
    results = {}
    
    cnn_kwargs = {'num_genes': num_genes}
    results['CNN Regressor'] = train_and_eval_loocv("CNN Regressor", CNNRegressor, cnn_kwargs, adata, num_genes, device, epochs, is_cnn=True)
    
    hist2st_kwargs = {'num_genes': num_genes}
    results['Hist2ST'] = train_and_eval_loocv("Hist2ST", Hist2STBaseline, hist2st_kwargs, adata, num_genes, device, epochs, is_cnn=True)
    
    gf_kwargs = {'num_genes': num_genes, 'num_experts': 4, 'device': device}
    results['Gaussian Flow'] = train_and_eval_loocv("Gaussian Flow", GaussianFlowModel, gf_kwargs, adata, num_genes, device, epochs)
    
    na_kwargs = {'num_genes': num_genes, 'num_experts': 4, 'device': device}
    results['No Context Flow'] = train_and_eval_loocv("No Context Flow", EczemaFlowNoAttention, na_kwargs, adata, num_genes, device, epochs)
    
    full_kwargs = {'num_genes': num_genes, 'num_experts': 4, 'device': device}
    results['EczemaFlow (Full)'] = train_and_eval_loocv("EczemaFlow", EczemaFlowModel, full_kwargs, adata, num_genes, device, epochs)
    
    import json
    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Traceable prediction run complete! Results saved to results/metrics.json")

if __name__ == "__main__":
    main()
