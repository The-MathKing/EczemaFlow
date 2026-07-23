"""
Computational performance profiling:
- Parameter count
- Training time per epoch
- Inference time per slide (all spots)
- Peak memory usage
"""
import torch
import torch.optim as optim
import time
import numpy as np
import json
import os
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '/Volumes/2TB/EczemaFlow-1')
from run_full_benchmarks import PrecomputedFoldDataset
from eczema_flow.model import EczemaFlowModel

device = torch.device('cpu')
os.makedirs("results", exist_ok=True)

# Load one fold for profiling
train_ds = PrecomputedFoldDataset(["fold_1","fold_2","fold_3","fold_4","fold_5"])
val_ds = PrecomputedFoldDataset(["test"])
variances = train_ds.counts.var(dim=0)
top_idx = torch.argsort(variances, descending=True)[:500]

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

model = EczemaFlowModel(num_genes=500, num_experts=4, device=device).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# -------------------------------------------------------
# 1. Parameter count
# -------------------------------------------------------
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# -------------------------------------------------------
# 2. Training time per epoch
# -------------------------------------------------------
model.train()
epoch_times = []
for epoch in range(3):  # Profile 3 epochs
    start = time.time()
    for patches, b_counts, b_coords in train_loader:
        patches = patches.to(device)
        b_counts = b_counts[:, top_idx].to(device)
        b_coords = b_coords.to(device)
        optimizer.zero_grad()
        loss = model.compute_loss(patches, b_counts, b_coords, is_precomputed=True)
        loss.backward()
        optimizer.step()
    elapsed = time.time() - start
    epoch_times.append(elapsed)
    print(f"Epoch {epoch+1}: {elapsed:.1f}s")

avg_epoch_time = float(np.mean(epoch_times))
print(f"\nAverage training time per epoch: {avg_epoch_time:.1f}s")

# -------------------------------------------------------
# 3. Inference time (all val spots)
# -------------------------------------------------------
model.eval()
n_spots = len(val_ds)
start = time.time()
with torch.no_grad():
    for patches, b_counts, b_coords in val_loader:
        patches = patches.to(device)
        b_coords = b_coords.to(device)
        _ = model.sample(patches, b_coords, num_steps=20, is_precomputed=True)
inference_time = time.time() - start
per_spot_ms = (inference_time / n_spots) * 1000

print(f"\nInference time for {n_spots} spots: {inference_time:.1f}s")
print(f"Per-spot inference time: {per_spot_ms:.2f} ms")
print(f"ODE evaluations per spot: 20 steps")

# -------------------------------------------------------
# 4. Save results
# -------------------------------------------------------
results = {
    "total_parameters": total_params,
    "trainable_parameters": trainable_params,
    "avg_epoch_time_seconds": avg_epoch_time,
    "inference_time_seconds_all_spots": inference_time,
    "num_val_spots": n_spots,
    "per_spot_inference_ms": per_spot_ms,
    "ode_steps": 20,
}

with open("results/compute_profile.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nProfile saved to results/compute_profile.json")
print(json.dumps(results, indent=2))
