import os
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from eczema_flow.dataset import VisiumDataset

def calculate_mse(adata, preds):
    # Simulated calculation
    return np.mean((adata.X - preds) ** 2)

def run_inference(h5ad_path, model_path):
    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(f"OOD Dataset {h5ad_path} not found. Please provide the external Psoriasis dataset.")
    
    adata = ad.read_h5ad(h5ad_path)
    # Placeholder for actual inference logic
    preds = np.random.rand(*adata.X.shape)
    
    return calculate_mse(adata, preds)

def main():
    parser = argparse.ArgumentParser(description="Evaluate OOD generalization on external datasets")
    parser.add_argument("--psoriasis_h5ad", type=str, default="data/external/Psoriasis_spatial.h5ad")
    parser.add_argument("--model_path", type=str, default="models/eczemaflow_fold_1.pth")
    args = parser.parse_args()
    
    os.makedirs('paper/figures', exist_ok=True)
    
    print("Running inference on Atopic Dermatitis (In-Domain)...")
    ad_mse = run_inference("data/GSE206391_spatial.h5ad", args.model_path)
    
    print("Running inference on Psoriasis (Near OOD)...")
    ps_mse = run_inference(args.psoriasis_h5ad, args.model_path)

if __name__ == "__main__":
    main()
