import torch
import time
from torch.utils.data import DataLoader
from eczema_flow.dataset import VisiumDataset
d = VisiumDataset("data", "data/splits.csv", "train")
loader = DataLoader(d, batch_size=16)
start = time.time()
for i, (patches, counts, coords) in enumerate(loader):
    if i == 10:
        break
print(f"10 batches took {time.time()-start:.2f} seconds")
