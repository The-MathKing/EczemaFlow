import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import scanpy as sc


def generate_benchmark_chart():
    if not os.path.exists("results/metrics.json"):
        print("Metrics file not found. Run benchmark script first.")
        return
        
    with open("results/metrics.json", "r") as f:
        results = json.load(f)
        
    os.makedirs('paper/figures', exist_ok=True)
    labels = list(results.keys())
    
    mses = [res["MSE"] for res in results.values()]
    mse_cis = [res.get("MSE_CI_95", 0.0) for res in results.values()]
    pccs = [res["PCC"] for res in results.values()]
    
    x = np.arange(len(labels))
    width = 0.5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    colors = ['#d62728' if 'Full' in l else '#aec7e8' for l in labels]
    bars = ax1.bar(x, mses, width, color=colors, yerr=mse_cis, capsize=4)
    ax1.set_ylabel('Mean Squared Error (log1p scale)')
    ax1.set_title('MSE Performance (6-fold LOOCV, ±95% CI)')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=25, ha='right')
    ax1.set_ylim(bottom=0)
    
    ax2.bar(x, pccs, width, color=colors)
    ax2.set_ylabel('Mean Gene-wise Pearson Correlation')
    ax2.set_title('PCC Performance')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=25, ha='right')
    y_max = max(abs(min(pccs)), abs(max(pccs))) * 1.2 if pccs else 1.0
    ax2.set_ylim(-y_max, y_max)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    fig.suptitle('Performance Benchmarks of WSI-to-ST Inference Models (LOOCV)', fontsize=16)
    fig.tight_layout()
    plt.savefig('paper/figures/benchmark_chart.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved benchmark chart.")


def generate_tda_comparison():
    """
    Generate a figure comparing the full TDA ablation ladder:
    Density Only → H0 Only → H1 Only → No Topology → Full Model
    """
    if not os.path.exists("results/tda_ablation.json") or not os.path.exists("results/metrics.json"):
        print("Ablation or metrics file not found.")
        return
    
    with open("results/tda_ablation.json", "r") as f:
        ablation = json.load(f)
    with open("results/metrics.json", "r") as f:
        full_results = json.load(f)
    
    os.makedirs('paper/figures', exist_ok=True)
    
    # Build ordered ablation ladder
    labels = ["Density Only", "H0 Only", "H1 Only", "No Topology Flow", "EczemaFlow (Full)"]
    mses = []
    cis = []
    for l in labels:
        if l in ablation:
            mses.append(ablation[l]["MSE"])
            cis.append(ablation[l].get("MSE_CI_95", 0.0))
        elif l in full_results:
            mses.append(full_results[l]["MSE"])
            cis.append(full_results[l].get("MSE_CI_95", 0.0))
        else:
            mses.append(0.0)
            cis.append(0.0)
    
    colors = ['#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#d62728']
    
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, mses, 0.6, color=colors, yerr=cis, capsize=5, edgecolor='black', linewidth=0.7)
    ax.set_ylabel('Mean Squared Error (log1p scale)')
    ax.set_title('Topology Ablation Ladder: Increasing Topological Complexity', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylim(bottom=0.29)
    ax.axhline(mses[-1], color='#d62728', linewidth=1.2, linestyle='--', alpha=0.5, label='EczemaFlow (Full)')
    # Annotate bars with MSE values
    for bar, mse in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003, f'{mse:.3f}',
                ha='center', va='bottom', fontsize=8)
    ax.legend()
    fig.tight_layout()
    plt.savefig('paper/figures/tda_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved tda_comparison.pdf.")


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
    
    test_slide = 'P16357_1028'
    adata_test = adata[adata.obs['sample'] == test_slide].copy()
    
    n_pts = min(len(adata_test), preds.shape[0])
    preds_subset = preds[:n_pts]
    adata_test = adata_test[:n_pts].copy()
    
    markers = ["ACTA2", "TPM3", "PSMB4"]
    for m in markers:
        if m in adata_test.var_names:
            idx = adata_test.var_names.get_loc(m)
            adata_test.obs[f"{m}_predicted"] = preds_subset[:, idx]
            adata_test.obs[f"{m}_residual"] = adata_test[:, m].X.toarray().squeeze() - preds_subset[:, idx]
            
            generate_spatial_scatter(
                adata_test, f"{m}_predicted",
                f"Predicted {m} Expression",
                f"spatial_marker_pred_{m}.pdf"
            )
            generate_spatial_scatter(
                adata_test, m,
                f"Ground Truth {m} Expression",
                f"spatial_marker_gt_{m}.pdf"
            )
            generate_spatial_scatter(
                adata_test, f"{m}_residual",
                f"Residual Error: {m} (GT − Predicted)",
                f"spatial_marker_res_{m}.pdf",
                cmap="coolwarm"
            )
        
    print("Saved spatial marker maps.")

    # Generate gene_prediction_maps figure with gene symbols
    n_show = min(3, len(markers))
    matched = [m for m in markers if m in adata_test.var_names][:n_show]
    if matched:
        fig, axes = plt.subplots(n_show, 3, figsize=(14, 4 * n_show))
        if n_show == 1:
            axes = [axes]
        for i, m in enumerate(matched):
            idx = adata_test.var_names.get_loc(m)
            gt = adata_test[:, m].X.toarray().squeeze()
            pr = preds_subset[:, idx]
            res = gt - pr

            axes[i][0].scatter(adata_test.obsm['spatial'][:, 0], adata_test.obsm['spatial'][:, 1],
                               c=gt, cmap='magma', s=3)
            axes[i][0].set_title(f'Ground Truth: {m}', fontsize=9)
            axes[i][0].axis('off')

            axes[i][1].scatter(adata_test.obsm['spatial'][:, 0], adata_test.obsm['spatial'][:, 1],
                               c=pr, cmap='magma', s=3)
            axes[i][1].set_title(f'Predicted: {m}', fontsize=9)
            axes[i][1].axis('off')

            im = axes[i][2].scatter(adata_test.obsm['spatial'][:, 0], adata_test.obsm['spatial'][:, 1],
                               c=res, cmap='coolwarm', s=3)
            axes[i][2].set_title(f'Residual: {m} (GT−Pred)', fontsize=9)
            axes[i][2].axis('off')
            plt.colorbar(im, ax=axes[i][2], fraction=0.03)

        fig.suptitle('Spatial Gene Expression: Ground Truth | Predicted | Residual', fontsize=13)
        fig.tight_layout()
        plt.savefig('paper/figures/gene_prediction_maps.pdf', bbox_inches='tight', dpi=300)
        plt.close()
        print("Saved gene_prediction_maps.pdf with gene symbols and residuals.")


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
    generate_tda_comparison()
    generate_marker_maps()
    generate_total_counts_map()
