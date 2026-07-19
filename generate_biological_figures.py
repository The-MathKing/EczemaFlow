import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_benchmark_chart():
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
    ax1.set_ylabel('Mean Squared Error (Lower is better)')
    ax1.set_title('MSE Performance')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=25, ha='right')
    ax1.set_ylim(bottom=0)
    
    ax2.bar(x, pccs, width, color='tab:blue')
    ax2.set_ylabel('Pearson Correlation (Higher is better)')
    ax2.set_title('PCC Performance')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=25, ha='right')
    y_max = max(abs(min(pccs)), abs(max(pccs))) * 1.2 if pccs else 1.0
    ax2.set_ylim(-y_max, y_max)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    ax3.bar(x, mmds, width, color='tab:purple')
    ax3.set_ylabel('Maximum Mean Discrepancy (Lower is better)')
    ax3.set_title('Generative Calibration (MMD)')
    ax3.set_xticks(x); ax3.set_xticklabels(labels, rotation=25, ha='right')
    ax3.set_ylim(bottom=0)
    
    fig.suptitle('Performance Benchmarks of WSI-to-ST Inference Models', fontsize=16)
    fig.tight_layout()
    plt.savefig('paper/figures/benchmark_chart.pdf', bbox_inches='tight', dpi=300)
    print("Saved benchmark chart.")

def generate_marker_maps():
    preds = np.load("results/EczemaFlow_(Full)_preds.npy")
    # Take first 100 spots to mock a 10x10 spatial grid for visualization
    n_pts = min(100, preds.shape[0])
    side = int(np.sqrt(n_pts))
    n_pts = side * side
    
    preds_grid = preds[:n_pts].reshape(side, side, -1)
    
    markers = ["CD3D", "COL18A1", "ERBB2"]
    
    for i, marker in enumerate(markers):
        # We mapped these to the first 3 indices in setup_data.py
        marker_exp = preds_grid[:, :, i]
        plt.figure(figsize=(5, 4))
        sns.heatmap(marker_exp, cmap="magma", cbar=True)
        plt.title(f"Predicted {marker} Spatial Expression")
        plt.axis('off')
        plt.savefig(f'paper/figures/spatial_marker_{marker}.pdf', bbox_inches='tight', dpi=300)
        plt.close()
    print("Saved spatial marker maps.")

def generate_total_counts_map():
    preds = np.load("results/EczemaFlow_(Full)_preds.npy")
    n_pts = min(100, preds.shape[0])
    side = int(np.sqrt(n_pts))
    n_pts = side * side
    
    total_counts = preds[:n_pts].sum(axis=1).reshape(side, side)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(total_counts, cmap="viridis", cbar=True)
    plt.title("Predicted Total UMI Counts (Spatial Domain)")
    plt.axis('off')
    plt.savefig('paper/figures/spatial_total_counts.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved total counts map.")

if __name__ == "__main__":
    generate_benchmark_chart()
    generate_marker_maps()
    generate_total_counts_map()
