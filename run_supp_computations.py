"""
Generate QC table and cohort statistics.
Also generates gene prediction spatial maps and the uncertainty/generative evaluation.
"""
import torch
import torch.optim as optim
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '/Volumes/2TB/EczemaFlow-1')
from run_full_benchmarks import PrecomputedFoldDataset
from eczema_flow.model import EczemaFlowModel

os.makedirs("results", exist_ok=True)
os.makedirs("paper/figures", exist_ok=True)

device = torch.device('cpu')

# ===========================================================
# A1: Cohort QC Table
# ===========================================================
print("="*60)
print("A1: Computing Cohort QC Statistics")
print("="*60)

folds = [f"fold_{i}" for i in range(1, 6)] + ["test"]
cohort_rows = []
for fold in folds:
    try:
        ds = PrecomputedFoldDataset([fold])
        n_spots = len(ds)
        n_genes = ds.counts.shape[1]
        mean_umi = float(ds.counts.sum(dim=1).mean().item())
        cohort_rows.append({
            'fold': fold,
            'n_spots': n_spots,
            'n_genes': n_genes,
            'mean_umi_per_spot': mean_umi,
        })
        print(f"  {fold}: {n_spots} spots, {n_genes} genes, mean UMI={mean_umi:.0f}")
    except Exception as e:
        print(f"  {fold}: ERROR - {e}")

total_spots = sum(r['n_spots'] for r in cohort_rows)
print(f"\nTotal spots across all folds: {total_spots}")

qc_report = {
    'folds': cohort_rows,
    'total_spots': total_spots,
    'total_genes_raw': cohort_rows[0]['n_genes'] if cohort_rows else 0,
    'hvg_panel': 500,
}
with open("results/cohort_qc.json", "w") as f:
    json.dump(qc_report, f, indent=2)
print("QC report saved to results/cohort_qc.json")

# ===========================================================
# A3: Generative Evaluation (multi-sample uncertainty)
# ===========================================================
print("\n" + "="*60)
print("A3: Generative Uncertainty Evaluation (N=10 samples)")
print("="*60)

train_ds = PrecomputedFoldDataset([f"fold_{i}" for i in range(1, 6)])
val_ds = PrecomputedFoldDataset(["test"])
variances = train_ds.counts.var(dim=0)
top_idx = torch.argsort(variances, descending=True)[:500]
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

# Train a quick model for this evaluation
model = EczemaFlowModel(num_genes=500, num_experts=4, device=device).to(device)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

print("  Training model for generative eval (30 epochs)...")
for epoch in range(30):
    model.train()
    for patches, b_counts, b_coords in train_loader:
        patches = patches.to(device)
        b_counts = b_counts[:, top_idx].to(device)
        b_coords = b_coords.to(device)
        optimizer.zero_grad()
        loss = model.compute_loss(patches, b_counts, b_coords, is_precomputed=True)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"    Epoch {epoch+1}/30")

# Sample N=10 draws per spot, compute mean and std
model.eval()
N_SAMPLES = 10
all_means, all_stds, all_targets = [], [], []

with torch.no_grad():
    for patches, b_counts, b_coords in val_loader:
        patches = patches.to(device)
        b_counts = b_counts[:, top_idx].to(device)
        b_coords = b_coords.to(device)
        target_log = torch.log1p(b_counts)
        
        draws = []
        for _ in range(N_SAMPLES):
            preds = model.sample(patches, b_coords, num_steps=20, is_precomputed=True)
            draws.append(preds.cpu().numpy())
        
        draws = np.stack(draws, axis=0)  # (N_SAMPLES, batch, G)
        batch_mean = draws.mean(axis=0)   # (batch, G)
        batch_std = draws.std(axis=0)     # (batch, G)
        
        all_means.append(batch_mean)
        all_stds.append(batch_std)
        all_targets.append(target_log.cpu().numpy())

all_means = np.concatenate(all_means, axis=0)
all_stds = np.concatenate(all_stds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# Compute calibration: do higher-uncertainty spots have higher errors?
spot_errors = np.mean((all_means - all_targets)**2, axis=1)
spot_uncertainty = np.mean(all_stds, axis=1)
corr_unc_err, _ = pearsonr(spot_uncertainty, spot_errors)

# Compute sample diversity
mean_std = float(all_stds.mean())

# Compute PCC of the mean prediction
pcc_list = []
for g in range(all_means.shape[1]):
    p = all_means[:, g]; t = all_targets[:, g]
    if np.std(p) > 1e-8 and np.std(t) > 1e-8:
        r, _ = pearsonr(p, t)
        if not np.isnan(r):
            pcc_list.append(r)
mean_pcc = float(np.mean(pcc_list)) if pcc_list else 0.0

gen_results = {
    'n_samples_per_spot': N_SAMPLES,
    'mean_pcc_of_ensemble': mean_pcc,
    'mean_predictive_std': mean_std,
    'uncertainty_error_correlation': float(corr_unc_err),
    'description': 'Positive correlation means higher uncertainty -> higher error (calibrated)'
}
with open("results/generative_eval.json", "w") as f:
    json.dump(gen_results, f, indent=2)

print(f"  Mean ensemble PCC: {mean_pcc:.4f}")
print(f"  Mean predictive std: {mean_std:.4f}")
print(f"  Uncertainty-Error correlation: {corr_unc_err:.4f} {'(calibrated ✓)' if corr_unc_err > 0 else '(miscalibrated)'}")
print("  Generative eval saved to results/generative_eval.json")

# ===========================================================
# B2: Gene Prediction Spatial Maps
# ===========================================================
print("\n" + "="*60)
print("B2: Generating Gene Prediction Spatial Maps")
print("="*60)

# Use coords from the test fold
test_ds = PrecomputedFoldDataset(["test"])
coords = test_ds.coords.numpy()  # (N, 2)
test_counts = test_ds.counts[:, top_idx]
target_log_all = torch.log1p(test_counts).numpy()

# Get full predictions for test fold
test_loader_seq = DataLoader(test_ds, batch_size=256, shuffle=False)
preds_all = []
with torch.no_grad():
    for patches, b_counts, b_coords in test_loader_seq:
        patches = patches.to(device)
        b_coords = b_coords.to(device)
        preds = model.sample(patches, b_coords, num_steps=20, is_precomputed=True)
        preds_all.append(preds.cpu().numpy())
preds_all = np.concatenate(preds_all, axis=0)

# Find top-3 genes by PCC for visualization
top_gene_pccs = []
for g in range(preds_all.shape[1]):
    p = preds_all[:, g]; t = target_log_all[:, g]
    if np.std(p) > 1e-8 and np.std(t) > 1e-8:
        r, _ = pearsonr(p, t)
        if not np.isnan(r):
            top_gene_pccs.append((r, g))

top_gene_pccs.sort(reverse=True)
best_genes = top_gene_pccs[:3]  # Top 3 genes by PCC

fig, axes = plt.subplots(len(best_genes), 2, figsize=(10, 4*len(best_genes)))
fig.suptitle("EczemaFlow: Observed vs. Predicted Gene Expression\n(Test Patient 8, Top 3 Genes by PCC)", fontsize=12, fontweight='bold')

for row, (pcc, gene_idx) in enumerate(best_genes):
    ax_obs = axes[row, 0]
    ax_pred = axes[row, 1]
    
    obs_vals = target_log_all[:, gene_idx]
    pred_vals = preds_all[:, gene_idx]
    
    vmin = min(obs_vals.min(), pred_vals.min())
    vmax = max(obs_vals.max(), pred_vals.max())
    
    sc1 = ax_obs.scatter(coords[:, 0], coords[:, 1], c=obs_vals, cmap='RdYlBu_r', s=5,
                          vmin=vmin, vmax=vmax)
    ax_obs.set_title(f"Gene {gene_idx} — Observed", fontsize=9)
    ax_obs.set_aspect('equal'); ax_obs.axis('off')
    plt.colorbar(sc1, ax=ax_obs, shrink=0.7)
    
    sc2 = ax_pred.scatter(coords[:, 0], coords[:, 1], c=pred_vals, cmap='RdYlBu_r', s=5,
                           vmin=vmin, vmax=vmax)
    ax_pred.set_title(f"Gene {gene_idx} — Predicted (PCC={pcc:.3f})", fontsize=9)
    ax_pred.set_aspect('equal'); ax_pred.axis('off')
    plt.colorbar(sc2, ax=ax_pred, shrink=0.7)

plt.tight_layout()
out_path = "paper/figures/gene_prediction_maps.pdf"
plt.savefig(out_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"  Gene prediction maps saved to {out_path}")

print("\n" + "="*60)
print("All supplementary computations complete!")
print("  results/cohort_qc.json")
print("  results/generative_eval.json")
print("  paper/figures/gene_prediction_maps.pdf")
print("="*60)
