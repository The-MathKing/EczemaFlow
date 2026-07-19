import torch
import traceback
import sys
try:
    from run_full_benchmarks import *
    device = torch.device('cpu')
    print("Initializing GaussianFlowModel...")
    gaussian_flow = GaussianFlowModel(num_genes=500, num_experts=4, device=device).to(device)
    print("GaussianFlowModel Initialized.")
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
