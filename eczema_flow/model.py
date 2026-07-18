import torch
import torch.nn as nn
from .attention import ConditioningNetwork
from .moe import MoEVectorField
from .prior import ZINBPrior, GaussianPrior

class EczemaFlowModel(nn.Module):
    """
    Complete EczemaFlow framework combining:
    - ViT Contextual Encoder Conditioning
    - MoE Vector Field
    - ZINB Prior integration
    """
    def __init__(self, num_genes=500, cond_dim=256, num_experts=4, device='cpu'):
        super().__init__()
        self.device = device
        self.conditioner = ConditioningNetwork(embed_dim=cond_dim)
        self.vector_field = MoEVectorField(
            num_experts=num_experts, 
            x_dim=num_genes, 
            cond_dim=cond_dim
        )
        # Use Gaussian Prior to avoid MPS CPU Fallbacks associated with ZINB
        self.prior = GaussianPrior(num_genes=num_genes, device=device)
        self.num_genes = num_genes

    def forward_vector_field(self, t, x, c):
        """Wrapper for ODE solvers."""
        return self.vector_field(t, x, c)

    def compute_loss(self, patches, target_counts):
        """
        Compute the Flow Matching objective.
        Optimal Transport Flow Matching (OT-FM) loss:
        L = || v_\theta(x_t, t, c) - u_t(x_0, x_1) ||^2
        
        where:
        x_0 ~ P_0 (ZINB prior)
        x_1 ~ P_1 (target transcriptomics)
        t ~ U[0, 1]
        x_t = t * x_1 + (1 - t) * x_0
        u_t(x_0, x_1) = x_1 - x_0
        """
        batch_size = patches.size(0)
        
        # 1. Obtain morphology condition
        c = self.conditioner(patches)
        
        # 2. Sample target x_1 (empirical ST data)
        # We apply log1p as standard preprocessing for counts
        x_1 = torch.log1p(target_counts).to(self.device)
        
        # 3. Sample base noise x_0 from ZINB prior
        x_0 = self.prior.sample(batch_size).to(self.device)
        
        # 4. Sample time t uniformly
        t = torch.rand(batch_size, 1, device=self.device)
        
        # 5. Construct interpolant x_t (Optimal transport path)
        x_t = t * x_1 + (1 - t) * x_0
        
        # 6. Target vector field u_t
        u_t = x_1 - x_0
        
        # 7. Predicted vector field v_theta
        v_theta = self.vector_field(t, x_t, c)
        
        # 8. MSE Loss
        loss = torch.mean((v_theta - u_t)**2)
        
        return loss

    def sample(self, patches, num_steps=50):
        """
        Sample from the model given morphology patches using an ODE solver.
        Requires `torchdiffeq`.
        """
        try:
            from torchdiffeq import odeint
        except ImportError:
            raise ImportError("Please install torchdiffeq to sample: pip install torchdiffeq")
            
        batch_size = patches.size(0)
        c = self.conditioner(patches)
        
        # Sample x_0 from ZINB prior
        x_0 = self.prior.sample(batch_size).to(self.device)
        
        # ODE wrapper
        class ODEWrapper(nn.Module):
            def __init__(self, vf, c):
                super().__init__()
                self.vf = vf
                self.c = c
            def forward(self, t, x):
                return self.vf(t, x, self.c)
                
        ode_func = ODEWrapper(self.vector_field, c)
        
        # Time integration from t=0 to t=1
        t_span = torch.linspace(0, 1, num_steps, device=self.device)
        
        # Solve ODE
        with torch.no_grad():
            trajectory = odeint(ode_func, x_0, t_span, method='euler')
            
        # trajectory is (num_steps, batch_size, x_dim)
        # Return the final state x_1
        x_1 = trajectory[-1]
        
        # Map back from log(1+x) space if needed, though often we just evaluate in log space
        return x_1
