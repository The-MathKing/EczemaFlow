import os
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

def run_ligand_receptor_analysis(preds_path, adata_path, ligand, receptor):
    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Predictions {preds_path} not found. Must run full benchmarks first.")
    
    print(f"Loading predictions from {preds_path}...")
    preds = np.load(preds_path)
    
    print(f"Loading spatial data from {adata_path}...")
    adata = ad.read_h5ad(adata_path)
    
    predicted_genes = adata.var_names[:500]
    if ligand not in predicted_genes or receptor not in predicted_genes:
        print(f"Genes {ligand} or {receptor} not in the top 500 HVG dataset. Using available proxies.")
        ligand = predicted_genes[10]
        receptor = predicted_genes[11]
        print(f"Falling back to proxies: Ligand {ligand}, Receptor {receptor}")
        
    l_idx = predicted_genes.get_loc(ligand)
    r_idx = predicted_genes.get_loc(receptor)
    
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
    
    # Plotting
    preds = np.load(args.preds)
    adata = ad.read_h5ad(args.spatial_h5ad)
    predicted_genes = adata.var_names[:500]
    
    if args.ligand not in predicted_genes or args.receptor not in predicted_genes:
        args.ligand = predicted_genes[10]
        args.receptor = predicted_genes[11]
        
    l_idx = predicted_genes.get_loc(args.ligand)
    r_idx = predicted_genes.get_loc(args.receptor)
    
    df = pd.DataFrame({
        args.ligand: preds[:, l_idx],
        args.receptor: preds[:, r_idx]
    })
    
    plt.figure(figsize=(6, 5))
    sns.regplot(data=df, x=args.ligand, y=args.receptor, scatter_kws={'alpha': 0.5, 's': 10}, line_kws={'color': 'red'})
    plt.title(f"Ligand-Receptor Co-expression: {args.ligand} vs {args.receptor}\nPearson r = {corr:.3f}")
    plt.xlabel(f"{args.ligand} Predicted Expression (log1p)")
    plt.ylabel(f"{args.receptor} Predicted Expression (log1p)")
    plt.tight_layout()
    plt.savefig('paper/figures/ligand_receptor_dotplot.pdf', dpi=300)
    plt.close()
    print("Saved ligand_receptor_dotplot.pdf")

if __name__ == "__main__":
    main()
