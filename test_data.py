import torch
import numpy as np
import matplotlib.pyplot as plt

data = torch.load("data/precomputed/fold_1_features.pt", map_location='cpu', weights_only=True)
counts = data['counts']
features = data['features']
coords = data['coords']

print("Counts shape:", counts.shape)
print("Features shape:", features.shape)
print("Counts sparsity (% zeros):", (counts == 0).float().mean().item())
print("Counts mean:", counts.mean().item())
print("Counts max:", counts.max().item())

# Check variance of features
print("Features variance:", features.var(dim=0).mean().item())

