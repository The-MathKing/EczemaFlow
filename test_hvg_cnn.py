import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from run_full_benchmarks import PrecomputedFoldDataset
from eczema_flow.model_baselines import CNNRegressor

device = torch.device('cpu')

train_dataset = PrecomputedFoldDataset(["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"])
val_dataset = PrecomputedFoldDataset(["test"])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# 1. Compute top 500 HVGs from training set
all_train_counts = train_dataset.counts
variances = all_train_counts.var(dim=0)
top_500_idx = torch.argsort(variances, descending=True)[:500]

model = CNNRegressor(num_genes=500).to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-3)

print("Training CNN Regressor on top 500 HVGs for 20 epochs...")
for epoch in range(20):
    model.train()
    for patches, b_counts, _ in train_loader:
        patches = patches.to(device)
        b_counts = b_counts[:, top_500_idx].to(device)
        
        optimizer.zero_grad()
        preds = model(patches, is_precomputed=True)
        target_log = torch.log1p(b_counts)
        loss = F.mse_loss(preds, target_log)
        loss.backward()
        optimizer.step()

# Eval
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for patches, b_counts, _ in val_loader:
        patches = patches.to(device)
        b_counts = b_counts[:, top_500_idx].to(device)
        target_log = torch.log1p(b_counts)
        preds = model(patches, is_precomputed=True)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(target_log.cpu().numpy())

f_preds = np.concatenate(all_preds, axis=0)
f_targets = np.concatenate(all_targets, axis=0)

f_pcc_list = []
for g in range(f_preds.shape[1]):
    pred_g = f_preds[:, g]
    targ_g = f_targets[:, g]
    if np.std(pred_g) > 1e-8 and np.std(targ_g) > 1e-8:
        r, _ = pearsonr(pred_g, targ_g)
        if not np.isnan(r): f_pcc_list.append(r)

pcc = np.mean(f_pcc_list) if f_pcc_list else 0.0
mse = np.mean((f_preds - f_targets)**2)
print(f"Results: MSE={mse:.3f}, PCC={pcc:.3f}")

