import os
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import pearsonr

def run_ligand_receptor_analysis(preds_path, adata_path, ligand, receptor):
    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Predictions {preds_path} not found. Must run full benchmarks first.")
    
    print(f"Loading predictions from {preds_path}...")
    preds = np.load(preds_path)
    
    print(f"Loading spatial data from {adata_path}...")
    adata = ad.read_h5ad(adata_path)
    
    if ligand not in adata.var_names or receptor not in adata.var_names:
        raise ValueError(f"Genes {ligand} or {receptor} not in the HVG dataset. Cannot perform L-R co-expression analysis.")
        
    l_idx = np.where(adata.var_names == ligand)[0][0]
    r_idx = np.where(adata.var_names == receptor)[0][0]
    
    # Calculate co-expression using Pearson correlation across all spatial spots
    l_expr = preds[:, l_idx]
    r_expr = preds[:, r_idx]
    
    corr, pval = pearsonr(l_expr, r_expr)
    return corr, pval

def main():
    parser = argparse.ArgumentParser(description="Ligand-Receptor Co-expression Analysis")
    parser.add_argument("--spatial_h5ad", type=str, default="data/GSE206391_spatial.h5ad")
    parser.add_argument("--preds", type=str, default="results/EczemaFlow_preds.npy")
    parser.add_argument("--ligand", type=str, default="CCL27")
    parser.add_argument("--receptor", type=str, default="CCR10")
    args = parser.parse_args()
    
    os.makedirs('paper/figures', exist_ok=True)
    
    corr, pval = run_ligand_receptor_analysis(args.preds, args.spatial_h5ad, args.ligand, args.receptor)
    print(f"L-R Co-expression {args.ligand}-{args.receptor} | Pearson r: {corr:.3f}, p-value: {pval:.2e}")

if __name__ == "__main__":
    main()
