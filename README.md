# EczemaFlow

A Conditional Flow-Matching generative model to infer high-order spatial transcriptomics (ST) from standard H&E histology images.

## Features
- **ZINB Prior**: Handles zero-inflated and over-dispersed nature of spatial transcriptomics data.
- **Many-Body Attention**: Contextualizes high-order multi-cell interactions directly from WSI morphology patches.
- **Mixture-of-Experts (MoE) Vector Field**: Routes continuous normalizing flow ODE computations to specialized subnetworks based on tissue morphology.

## Installation
```bash
pip install -r requirements.txt
```
