# EczemaFlow: Architecture Mapping

This repository contains the official implementation of the EczemaFlow model. To facilitate rigorous code-paper consistency audits, we provide the explicit mapping of the manuscript's claims to the exact files in this repository.

### Key Components
- **Topological Data Analysis (TDA) & StarDist**: The cellular nuclei graph extraction, Vietoris-Rips complex construction, and persistence landscape generation are fully implemented in `eczema_flow/tda.py`. It explicitly calls the pre-trained StarDist model.
- **Vision Transformer (ViT) Contextual Encoder**: The frozen pre-trained `vit_b_16` foundation model, replacing traditional ResNet architectures, is implemented in `eczema_flow/attention.py`.
- **DynamicZINBPrior**: The morphology-conditioned ZINB prior (predicting per-spot, per-gene parameters) is explicitly defined in `eczema_flow/prior.py`.
- **Mixture-of-Experts Load Balancing**: The Top-2 MoE router and the auxiliary load-balancing coefficient of variation penalty are implemented in `eczema_flow/model.py`.
- **Stain Augmentation**: `ColorJitter` morphology augmentation to simulate cross-hospital batch effects is explicitly implemented in the data loader at `eczema_flow/dataset.py`.
- **Quantitative Baselines**: The `Hist2ST` baseline and Maximum Mean Discrepancy (MMD) distribution metrics are implemented and executed in `run_full_benchmarks.py`.

The optimization script `train.py` defaults to the production configuration: 50 epochs, batch size 64, over the 500 equivalent Highly Variable Genes.
