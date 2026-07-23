import torch
import numpy as np

data = torch.load("data/precomputed/fold_1_features.pt", map_location='cpu', weights_only=True)
coords = data['coords']

print("Coords shape:", coords.shape)
print("Coords min:", coords.min(dim=0)[0])
print("Coords max:", coords.max(dim=0)[0])
print("Coords mean:", coords.mean(dim=0))

