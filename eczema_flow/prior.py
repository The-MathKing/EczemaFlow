import torch
import torch.nn as nn

class ZINBPrior(nn.Module):
    r"""
    Zero-Inflated Negative Binomial (ZINB) Prior for Flow Matching.
    
    Standard Flow Matching assumes $x_0 \sim \mathcal{N}(0, I)$.
    However, for spatial transcriptomics, the base distribution should better reflect
    the zero-inflated and overdispersed data.
    
    Here we implement a parameterized ZINB prior that allows us to sample
    base noise $x_0$ that matches the marginal statistics of the dataset.
    """
    def __init__(self, num_genes, device='cpu'):
        super().__init__()
        self.num_genes = num_genes
        self.device = device
        
        # Learnable parameters for the prior (can be fitted to dataset marginals before training)
        # Log-dispersion (r)
        self.log_r = nn.Parameter(torch.zeros(num_genes))
        # Logit-probability for the binomial part of NB
        self.logit_p = nn.Parameter(torch.zeros(num_genes))
        # Logit-dropout (zero-inflation probability)
        self.logit_pi = nn.Parameter(torch.zeros(num_genes) - 1.0) # start with roughly 0.27 pi

    def sample(self, batch_size):
        """
        Sample from the ZINB prior.
        Note: The standard Poisson-Gamma mixture for NB is non-differentiable 
        with respect to standard reparameterization without custom gradients.
        For Flow Matching, we only need to sample the base distribution $x_0$, 
        which doesn't strictly need to be differentiable w.r.t its own parameters 
        if we fit the prior beforehand.
        """
        with torch.no_grad():
            r = torch.exp(self.log_r).expand(batch_size, -1)
            p = torch.sigmoid(self.logit_p).expand(batch_size, -1)
            pi = torch.sigmoid(self.logit_pi).expand(batch_size, -1)
            
            # Sample Gamma
            # PyTorch doesn't have a direct Gamma sample that is easy to parameterize with rate/shape directly 
            # in a unified tensor way without loop, but we can use torch.distributions
            from torch.distributions import Gamma, Poisson, Binomial
            
            # NB as Gamma-Poisson mixture:
            # lambda ~ Gamma(r, p / (1 - p))
            # x ~ Poisson(lambda)
            rate = p / (1 - p + 1e-8)
            gamma_dist = Gamma(r, rate)
            lambdas = gamma_dist.sample()
            
            poisson_dist = Poisson(lambdas)
            nb_samples = poisson_dist.sample()
            
            # Zero inflation
            binomial_dist = Binomial(1, probs=1 - pi)
            non_zero_mask = binomial_dist.sample()
            
            zinb_samples = nb_samples * non_zero_mask
            
            # Since flow matching operates on continuous spaces, we add a small 
            # continuous dequantization noise (e.g. uniform noise U(0, 1))
            dequantized_samples = zinb_samples + torch.rand_like(zinb_samples)
            
            # Standardize a bit to help the neural network
            # Log(1+x) transform is common in scRNA
            log_samples = torch.log1p(dequantized_samples)
            
            return log_samples

    def fit_to_data(self, dataloader):
        """
        A helper function to fit the prior parameters to the marginal 
        statistics of the dataset.
        For a mock implementation, we leave this as a placeholder.
        """
        # In practice, compute mean, variance, and dropout rate per gene
        # and set self.log_r, self.logit_p, self.logit_pi accordingly using moment matching.
        pass

class GaussianPrior(nn.Module):
    r"""
    Standard Gaussian Prior for Flow Matching.
    This bypasses all MPS CPU fallbacks associated with Poisson/Gamma sampling,
    enabling full hardware acceleration on Apple Silicon GPUs.
    """
    def __init__(self, num_genes, device='cpu'):
        super().__init__()
        self.num_genes = num_genes
        self.device = device
        
    def sample(self, batch_size):
        """
        Sample from N(0, I). Native MPS support.
        """
        # We sample N(0, I) and optionally scale it slightly to match expected input space
        samples = torch.randn(batch_size, self.num_genes, device=self.device)
        return samples
