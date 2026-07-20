import torch
import numpy as np
from eczema_flow.model import EczemaFlowModel
from eczema_flow.model_baselines import GaussianPrior
import torch.nn as nn

class GaussianFlowModel(EczemaFlowModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior = GaussianPrior(num_genes=self.num_genes, device=self.device)

    def sample(self, patches, num_steps=50, is_precomputed=False):
        try:
            from torchdiffeq import odeint
        except ImportError:
            raise ImportError("Please install torchdiffeq to sample: pip install torchdiffeq")
            
        if is_precomputed:
            c = self.conditioner.forward_precomputed(patches)
        else:
            c = self.conditioner(patches)
        
        x_0 = self.prior.sample(c).to(self.device)
        print("x_0 sampled.")
        
        class ODEWrapper(nn.Module):
            def __init__(self, vf, c):
                super().__init__()
                self.vf = vf
                self.c = c
            def forward(self, t, x):
                print(f"ODEWrapper.forward called at t={t}")
                res = self.vf(t, x, self.c)
                print(f"ODEWrapper.forward finished at t={t}")
                return res
                
        ode_func = ODEWrapper(self.vector_field, c)
        
        t_span = torch.linspace(0, 1, num_steps, device=self.device)
        print("t_span:", t_span)
        print("x_0 device:", x_0.device)
        print("t_span device:", t_span.device)
        
        with torch.no_grad():
            print("Calling odeint...")
            trajectory = odeint(ode_func, x_0, t_span, method='euler')
            print("Finished odeint!")
            
        x_1 = trajectory[-1]
        return x_1

device = torch.device('cpu')
model = GaussianFlowModel(num_genes=500, cond_dim=256, tda_dim=64, num_experts=4, device=device)

print("Model created.")
patches = torch.randn(2, 9, 3, 224, 224)
print("Sampling...")
preds = model.sample(patches, num_steps=2)
print("Done sampling! Preds shape:", preds.shape)
