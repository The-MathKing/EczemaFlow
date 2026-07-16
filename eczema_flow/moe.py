import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """
    A single expert sub-network for the MoE Vector Field.
    It takes the current state `x_t`, time `t`, and condition `c` 
    and predicts the vector field `v`.
    """
    def __init__(self, x_dim, cond_dim, hidden_dim=512):
        super().__init__()
        # We concatenate x_t (gene expression space), t (scalar), and c (condition)
        in_dim = x_dim + 1 + cond_dim
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, x_dim)
        )
        
    def forward(self, x, t, c):
        """
        x: (batch_size, x_dim)
        t: (batch_size, 1)
        c: (batch_size, cond_dim)
        """
        inp = torch.cat([x, t, c], dim=-1)
        return self.net(inp)

class MoEVectorField(nn.Module):
    """
    Mixture-of-Experts architecture for the Flow Matching vector field.
    The input WSI patches (condition `c`) dynamically route the ODE computation
    to specific expert sub-networks specialized in distinct tissue morphologies.
    """
    def __init__(self, num_experts=4, x_dim=500, cond_dim=256, hidden_dim=512, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        self.experts = nn.ModuleList([
            Expert(x_dim, cond_dim, hidden_dim) for _ in range(num_experts)
        ])
        
        # Router network: uses the morphology condition `c` to decide the expert
        self.router = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.SiLU(),
            nn.Linear(128, num_experts)
        )
        
    def forward(self, t, x, c):
        """
        This signature matches torchdiffeq requirements if wrapped properly,
        but for Flow Matching training, we evaluate it directly.
        t: scalar or (batch_size, 1)
        x: (batch_size, x_dim)
        c: (batch_size, cond_dim)
        """
        batch_size = x.size(0)
        
        # Ensure t is (batch_size, 1)
        if isinstance(t, float) or t.dim() == 0:
            t = torch.full((batch_size, 1), t, device=x.device)
        elif t.dim() == 1:
            t = t.view(-1, 1)
            
        # Get routing logits based purely on morphology `c`
        router_logits = self.router(c) # (batch_size, num_experts)
        
        # Softmax to get probabilities
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k routing (standard in MoE to save computation)
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities so they sum to 1
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Evaluate experts and combine
        # In a highly optimized implementation, we would use scattered evaluation.
        # Here we do a loop for simplicity.
        v_out = torch.zeros_like(x)
        
        for k in range(self.top_k):
            # The indices of the k-th selected expert for each sample in the batch
            expert_indices = top_k_indices[:, k]
            expert_probs = top_k_probs[:, k].unsqueeze(-1) # (batch_size, 1)
            
            # Since different samples go to different experts, we evaluate all experts 
            # and mask, or we can group samples. Grouping is faster.
            for i, expert in enumerate(self.experts):
                # Mask of which samples use expert `i` as their k-th choice
                mask = (expert_indices == i)
                if mask.any():
                    # Evaluate only for these samples
                    x_masked = x[mask]
                    t_masked = t[mask]
                    c_masked = c[mask]
                    
                    v_expert = expert(x_masked, t_masked, c_masked)
                    
                    # Weight by router probability
                    v_out[mask] += v_expert * expert_probs[mask]
                    
        return v_out
