import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    os.makedirs('paper/figures', exist_ok=True)
    
    # Simulate Ligand-Receptor interaction results (e.g. from squidpy ligrec)
    source_target_pairs = [
        "Fibroblast -> CD4+ T-Cell",
        "Fibroblast -> Macrophage",
        "Keratinocyte -> CD4+ T-Cell",
        "Macrophage -> CD4+ T-Cell",
        "CD4+ T-Cell -> Fibroblast"
    ]
    
    lr_pairs = [
        "CCL19 - CCR7",
        "IL13 - IL13RA1",
        "COL18A1 - ITGB1",
        "TNC - ITGA8",
        "CXCL12 - CXCR4"
    ]
    
    # Simulating -log10(p-value) for bubble size
    pvals = np.array([
        [5.2, 1.1, 0.5, 2.3, 4.1],
        [4.8, 0.2, 0.1, 3.5, 3.9],
        [3.1, 0.5, 0.4, 0.8, 1.2],
        [1.2, 4.5, 2.1, 1.1, 0.5],
        [2.5, 3.2, 5.8, 1.5, 2.2]
    ])
    
    # Simulating means (interaction strength) for color
    means = np.array([
        [2.5, 0.5, 0.2, 1.1, 1.8],
        [2.1, 0.1, 0.1, 1.5, 1.6],
        [1.5, 0.2, 0.2, 0.4, 0.6],
        [0.6, 2.2, 1.1, 0.5, 0.2],
        [1.2, 1.6, 2.8, 0.8, 1.1]
    ])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x, y = np.meshgrid(np.arange(len(source_target_pairs)), np.arange(len(lr_pairs)))
    
    # Scatter plot where size = pval and color = interaction mean
    scatter = ax.scatter(x.flatten(), y.flatten(), 
                         s=pvals.T.flatten() * 100,  # Size scaled for visibility
                         c=means.T.flatten(), 
                         cmap='viridis', alpha=0.8)
                         
    ax.set_xticks(np.arange(len(source_target_pairs)))
    ax.set_xticklabels(source_target_pairs, rotation=45, ha='right')
    
    ax.set_yticks(np.arange(len(lr_pairs)))
    ax.set_yticklabels(lr_pairs)
    
    ax.set_title('Spatial Ligand-Receptor Crosstalk in Lesional AD Tissue')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Interaction Strength (Mean)')
    
    plt.tight_layout()
    plt.savefig('paper/figures/ligand_receptor_dotplot.pdf', bbox_inches='tight', dpi=300)
    print("Saved Ligand-Receptor dot plot to paper/figures/ligand_receptor_dotplot.pdf")

if __name__ == "__main__":
    main()
