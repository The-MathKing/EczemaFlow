import torch
import numpy as np
from eczema_flow.model import EczemaFlowModel
from eczema_flow.model_baselines import GaussianPrior

class GaussianFlowModel(EczemaFlowModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior = GaussianPrior(num_genes=self.num_genes, device=self.device)

device = torch.device('cpu')
model = GaussianFlowModel(num_genes=500, cond_dim=256, tda_dim=64, num_experts=4, device=device)

# Monkey-patch the forward of MoEVectorField to print
original_vf_forward = model.vector_field.forward
def noisy_vf_forward(t, x, c):
    print(f"  Evaluating vf at t={t.item() if isinstance(t, torch.Tensor) else t}")
    return original_vf_forward(t, x, c)
model.vector_field.forward = noisy_vf_forward

print("Model created.")
patches = torch.randn(2, 9, 3, 224, 224)
print("Sampling...")
preds = model.sample(patches, num_steps=2)
print("Done sampling! Preds shape:", preds.shape)
