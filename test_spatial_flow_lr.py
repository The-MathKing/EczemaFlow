import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from run_full_benchmarks import PrecomputedFoldDataset
from eczema_flow.model import EczemaFlowModel
from eczema_flow.attention import ViTContextualEncoder

device = torch.device('cpu')

train_dataset = PrecomputedFoldDataset(["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"])
val_dataset = PrecomputedFoldDataset(["test"])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

all_train_counts = train_dataset.counts
variances = all_train_counts.var(dim=0)
top_500_idx = torch.argsort(variances, descending=True)[:500]

model = EczemaFlowModel(num_genes=500, num_experts=4, device=device).to(device)

# HACK: replace transformer with 1 layer to avoid collapse
embed_dim = model.conditioner.attention.embed_dim
model.conditioner.attention = ViTContextualEncoder(embed_dim=embed_dim, num_heads=4, num_layers=1).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4) # LOWER LR!

print("Training Spatially-Anchored Flow Matching with low LR for 20 epochs...")
for epoch in range(20):
    model.train()
    for patches, b_counts, b_coords in train_loader:
        patches = patches.to(device)
        b_counts = b_counts[:, top_500_idx].to(device)
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
        b_counts = b_counts[:, top_500_idx].to(device)
        b_coords = b_coords.to(device)
        target_log = torch.log1p(b_counts)
        preds = model.sample(patches, b_coords, num_steps=20, is_precomputed=True)
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

