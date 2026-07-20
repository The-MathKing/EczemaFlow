import torch
import torch.nn as nn

class ZINBPrior(nn.Module):
    """ Static ZINB Prior """
    def __init__(self, num_genes, device='cpu'):
        super().__init__()
        self.num_genes = num_genes
        self.device = device
        self.log_r = nn.Parameter(torch.zeros(num_genes))
        self.logit_p = nn.Parameter(torch.zeros(num_genes))
        self.logit_pi = nn.Parameter(torch.zeros(num_genes) - 1.0)
        
    def fit_to_data(self, adata):
        """
        Fits the static global ZINB prior to empirical spatial transcriptomics data.
        Calculates expected zero-inflation (pi) and negative binomial parameters
        (r, p) using method of moments on the raw count matrix.
        """
        print("Fitting ZINB prior to empirical distribution...")
        import numpy as np
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
            
        means = np.mean(X, axis=0)
        variances = np.var(X, axis=0)
        zero_props = np.mean(X == 0, axis=0)
        
        # Method of moments estimation
        # r = mean^2 / (var - mean), clamped for stability
        r_vals = np.square(means) / np.maximum(variances - means, 1e-4)
        r_vals = np.clip(r_vals, 1e-4, 100.0)
        
        # p = var / (mean + var), clamped
        p_vals = variances / np.maximum(means + variances, 1e-4)
        p_vals = np.clip(p_vals, 1e-4, 0.99)
        
        # zero inflation (pi) = empirical zero proportion
        pi_vals = np.clip(zero_props, 1e-4, 0.99)
        
        with torch.no_grad():
            self.log_r.copy_(torch.tensor(np.log(r_vals), dtype=torch.float32, device=self.device))
            # logit(p) = log(p / (1-p))
            self.logit_p.copy_(torch.tensor(np.log(p_vals / (1 - p_vals)), dtype=torch.float32, device=self.device))
            self.logit_pi.copy_(torch.tensor(np.log(pi_vals / (1 - pi_vals)), dtype=torch.float32, device=self.device))
        print("ZINB prior fitted successfully.")

    def sample(self, batch_size):
        with torch.no_grad():
            r = torch.exp(self.log_r).expand(batch_size, -1)
            p = torch.sigmoid(self.logit_p).expand(batch_size, -1)
            pi = torch.sigmoid(self.logit_pi).expand(batch_size, -1)
            
            from torch.distributions import Gamma, Poisson, Binomial
            rate = p / (1 - p + 1e-8)
            gamma_dist = Gamma(r, rate)
            lambdas = gamma_dist.sample()
            poisson_dist = Poisson(lambdas)
            nb_samples = poisson_dist.sample()
            binomial_dist = Binomial(1, probs=1 - pi)
            non_zero_mask = binomial_dist.sample()
            zinb_samples = nb_samples * non_zero_mask
            dequantized_samples = zinb_samples + torch.rand_like(zinb_samples)
            log_samples = torch.log1p(dequantized_samples)
            return log_samples

class DynamicZINBPrior(nn.Module):
    """ Morphology-conditioned ZINB Prior for Flow Matching. """
    def __init__(self, cond_dim, num_genes, device='cpu'):
        super().__init__()
        self.num_genes = num_genes
        self.device = device
        self.proj = nn.Linear(cond_dim, num_genes * 3)

    def fit_to_data(self, adata):
        """
        Dummy fit_to_data for interface compatibility. 
        DynamicZINBPrior parameters are learned end-to-end conditioned on morphology,
        so pre-fitting global parameters is not applicable.
        """
        pass

    def sample(self, c):
        batch_size = c.size(0)
        params = self.proj(c)
        log_r = params[:, :self.num_genes]
        logit_p = params[:, self.num_genes:2*self.num_genes]
        logit_pi = params[:, 2*self.num_genes:]
        
        with torch.no_grad():
            r = torch.exp(log_r)
            p = torch.sigmoid(logit_p)
            pi = torch.sigmoid(logit_pi)
            
            from torch.distributions import Gamma, Poisson, Binomial
            rate = p / (1 - p + 1e-8)
            gamma_dist = Gamma(r, rate)
            lambdas = gamma_dist.sample()
            poisson_dist = Poisson(lambdas)
            nb_samples = poisson_dist.sample()
            binomial_dist = Binomial(1, probs=1 - pi)
            non_zero_mask = binomial_dist.sample()
            zinb_samples = nb_samples * non_zero_mask
            dequantized_samples = zinb_samples + torch.rand_like(zinb_samples)
            log_samples = torch.log1p(dequantized_samples)
            return log_samples

class GaussianPrior(nn.Module):
    def __init__(self, num_genes, device='cpu'):
        super().__init__()
        self.num_genes = num_genes
        self.device = device
        
    def sample(self, batch_size):
        samples = torch.randn(batch_size, self.num_genes, device=self.device)
        return samples
