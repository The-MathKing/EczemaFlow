import torch
import numpy as np

data = torch.load("data/precomputed/fold_1_features.pt", map_location='cpu', weights_only=True)
counts = data['counts']

first_500 = counts[:, :500]
print("First 500 sparsity:", (first_500 == 0).float().mean().item())
print("First 500 mean:", first_500.mean().item())
print("First 500 variance:", first_500.var(dim=0).mean().item())

# Find the top 500 highly variable genes (highest variance)
variances = counts.var(dim=0)
top_500_idx = torch.argsort(variances, descending=True)[:500]
top_500 = counts[:, top_500_idx]

print("Top 500 sparsity:", (top_500 == 0).float().mean().item())
print("Top 500 mean:", top_500.mean().item())
print("Top 500 variance:", top_500.var(dim=0).mean().item())
