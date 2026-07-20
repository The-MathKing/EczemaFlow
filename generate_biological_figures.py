import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

def generate_benchmark_chart():
    if not os.path.exists("results/metrics.json"):
        print("Metrics file not found. Run benchmark script first.")
        return
        
    with open("results/metrics.json", "r") as f:
        results = json.load(f)
        
    os.makedirs('paper/figures', exist_ok=True)
    labels = list(results.keys())
    
    mses = [res[0] for res in results.values()]
    pccs = [res[1] for res in results.values()]
    mmds = [res[2] for res in results.values()]
    
    x = np.arange(len(labels))
    width = 0.5
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1.bar(x, mses, width, color='tab:red')
    ax1.set_ylabel('Mean Squared Error (log1p scale)')
    ax1.set_title('MSE Performance')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=25, ha='right')
    ax1.set_ylim(bottom=0)
    
    ax2.bar(x, pccs, width, color='tab:blue')
    ax2.set_ylabel('Pearson Correlation')
    ax2.set_title('PCC Performance')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=25, ha='right')
    y_max = max(abs(min(pccs)), abs(max(pccs))) * 1.2 if pccs else 1.0
    ax2.set_ylim(-y_max, y_max)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    ax3.bar(x, mmds, width, color='tab:purple')
    ax3.set_ylabel('Maximum Mean Discrepancy')
    ax3.set_title('Generative Calibration (MMD)')
    ax3.set_xticks(x); ax3.set_xticklabels(labels, rotation=25, ha='right')
    ax3.set_ylim(bottom=0)
    
    fig.suptitle('Performance Benchmarks of WSI-to-ST Inference Models (LOOCV)', fontsize=16)
    fig.tight_layout()
    plt.savefig('paper/figures/benchmark_chart.pdf', bbox_inches='tight', dpi=300)
    print("Saved benchmark chart.")

import scanpy as sc

def generate_spatial_scatter(adata, obs_key, title, filename, cmap="magma"):
    slide_id = adata.obs['sample'].iloc[0]
    sc.pl.spatial(adata, color=obs_key, library_id=slide_id, show=False, title=title, cmap=cmap)
    plt.savefig(f'paper/figures/{filename}', bbox_inches='tight', dpi=300)
    plt.close()

def generate_marker_maps():
    if not os.path.exists("results/EczemaFlow_test_preds.npy") or not os.path.exists("data/GSE206391_spatial.h5ad"):
        print("Predictions or adata not found. Skipping marker maps.")
        return
        
    preds = np.load("results/EczemaFlow_test_preds.npy")
    adata = sc.read_h5ad("data/GSE206391_spatial.h5ad")
    adata.var_names_make_unique()
    
    # Use the test slide Patient 8 (GSM6252942 -> slide 8-V19T12-006)
    test_slide = 'P16357_1028' # Assumed Patient 8 test slide
    adata_test = adata[adata.obs['sample'] == test_slide].copy()
    
    n_pts = min(len(adata_test), preds.shape[0])
    preds_subset = preds[:n_pts]
    adata_test = adata_test[:n_pts].copy()
    
    markers = ["CD3D", "COL18A1", "ERBB2"]
    for m in markers:
        if m in adata_test.var_names:
            idx = adata_test.var_names.get_loc(m)
            adata_test.obs[f"{m}_predicted"] = preds_subset[:, idx]
            generate_spatial_scatter(
                adata_test, f"{m}_predicted", 
                f"Predicted {m} Spatial Expression", 
                f"spatial_marker_pred_{m}.pdf"
            )
            generate_spatial_scatter(
                adata_test, m, 
                f"Ground Truth {m} Spatial Expression", 
                f"spatial_marker_gt_{m}.pdf"
            )
        
    print("Saved spatial marker maps.")

def generate_total_counts_map():
    if not os.path.exists("results/EczemaFlow_test_preds.npy") or not os.path.exists("data/GSE206391_spatial.h5ad"):
        return
        
    preds = np.load("results/EczemaFlow_test_preds.npy")
    adata = sc.read_h5ad("data/GSE206391_spatial.h5ad")
    adata.var_names_make_unique()
    test_slide = 'P16357_1028'
    adata_test = adata[adata.obs['sample'] == test_slide].copy()
    
    n_pts = min(len(adata_test), preds.shape[0])
    preds_subset = preds[:n_pts]
    adata_test = adata_test[:n_pts].copy()
    
    preds_counts = np.expm1(preds_subset)
    gt_counts = np.expm1(adata_test.X.toarray() if hasattr(adata_test.X, "toarray") else adata_test.X)
    
    adata_test.obs["Total_Predicted"] = preds_counts.sum(axis=1)
    adata_test.obs["Total_GroundTruth"] = gt_counts.sum(axis=1)
    
    generate_spatial_scatter(adata_test, "Total_Predicted", "Predicted Total UMI Counts", "spatial_total_counts_pred.pdf", cmap="viridis")
    generate_spatial_scatter(adata_test, "Total_GroundTruth", "Ground Truth Total UMI Counts", "spatial_total_counts_gt.pdf", cmap="viridis")
    
    print("Saved total counts maps.")

if __name__ == "__main__":
    generate_benchmark_chart()
    generate_marker_maps()
    generate_total_counts_map()
