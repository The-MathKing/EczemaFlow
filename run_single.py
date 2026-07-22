import os, json, torch, numpy as np
from run_full_benchmarks import train_and_eval_loocv
from eczema_flow.model import EczemaFlowModel

device = torch.device('cpu')
torch.set_num_threads(8)
num_genes = 500
epochs = 50

full_kwargs = {'num_genes': num_genes, 'num_experts': 4, 'device': device}
result = train_and_eval_loocv("EczemaFlow", EczemaFlowModel, full_kwargs, device, epochs)

# Load existing metrics, update EczemaFlow (Full), and save
with open("results/metrics.json", "r") as f:
    existing = json.load(f)

existing["EczemaFlow (Full)"] = result

with open("results/metrics.json", "w") as f:
    json.dump(existing, f, indent=4)
