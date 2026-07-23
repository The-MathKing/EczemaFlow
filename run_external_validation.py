"""
Run external validation on GSE197023.

Trains EczemaFlow on ALL GSE206391 (development cohort),
then evaluates zero-shot on GSE197023 (external cohort).
"""
import torch
import torch.optim as optim
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

import sys
sys.path.insert(0, '/Volumes/2TB/EczemaFlow-1')
from run_full_benchmarks import PrecomputedFoldDataset
from eczema_flow.dataset import PrecomputedVisiumDataset
from eczema_flow.model import EczemaFlowModel

device = torch.device('cpu')
os.makedirs("results", exist_ok=True)

print("="*60)
print("External Validation on GSE197023 (Mitamura et al. 2023)")
print("="*60)

# 1. Load full development cohort (all folds)
folds = [f"fold_{i}" for i in range(1, 6)] + ["test"]
train_ds = PrecomputedFoldDataset(folds)
print(f"Loaded full development cohort: {len(train_ds)} spots")

# Determine HVGs from development cohort
variances = train_ds.counts.var(dim=0)
top_idx = torch.argsort(variances, descending=True)[:500]

# 2. Train model on full development cohort
model = EczemaFlowModel(num_genes=500, num_experts=4, device=device).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

EPOCHS = 50
print(f"Training EczemaFlow for {EPOCHS} epochs on full development cohort...")
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
    if (epoch+1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS} completed")

# 3. Load external cohort
ext_path = "data/precomputed_gse197023/external_cohort.pt"
if not os.path.exists(ext_path):
    print(f"ERROR: {ext_path} not found. Did preprocess_gse197023.py complete successfully?")
    sys.exit(1)

ext_ds = PrecomputedVisiumDataset(ext_path)
print(f"Loaded external cohort: {len(ext_ds)} spots")

# 4. Evaluate zero-shot
ext_loader = DataLoader(ext_ds, batch_size=256, shuffle=False)
model.eval()

all_preds = []
all_targets = []
with torch.no_grad():
    for patches, b_counts, b_coords in ext_loader:
        # patches is (B, 4, 512) where CNN is :256 and TDA is 256:
        # We need to match the development set which had 64-dim TDA (total 320)
        cnn_part = patches[:, :, :256]
        tda_part = patches[:, :, 256:256+64]
        patches = torch.cat([cnn_part, tda_part], dim=-1)
        
        patches = patches.to(device)
        b_counts = b_counts[:, top_idx].to(device)  # We use the same top_idx! (aligned)
        b_coords = b_coords.to(device)
        
        target_log = torch.log1p(b_counts)
        preds = model.sample(patches, b_coords, num_steps=20, is_precomputed=True)
        
        all_preds.append(preds.cpu().numpy())
        all_targets.append(target_log.cpu().numpy())

preds_all = np.concatenate(all_preds, axis=0)
tgts_all = np.concatenate(all_targets, axis=0)

# Compute gene-wise PCC
pcc_list = []
for g in range(preds_all.shape[1]):
    p = preds_all[:, g]
    t = tgts_all[:, g]
    if np.std(p) > 1e-8 and np.std(t) > 1e-8:
        r, _ = pearsonr(p, t)
        if not np.isnan(r):
            pcc_list.append(r)

mean_pcc = float(np.mean(pcc_list)) if pcc_list else 0.0
mse = float(np.mean((preds_all - tgts_all)**2))

print(f"\n[RESULT] Zero-Shot External Validation (GSE197023):")
print(f"  PCC: {mean_pcc:.4f}")
print(f"  MSE: {mse:.4f}")

results = {
    'external_cohort': 'GSE197023',
    'num_spots': len(ext_ds),
    'num_hvgs_evaluated': 500,
    'mean_gene_pcc': mean_pcc,
    'mean_mse': mse
}

with open("results/external_validation.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to results/external_validation.json")
