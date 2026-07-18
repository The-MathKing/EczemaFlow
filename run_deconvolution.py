import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls

def generate_mock_reference(num_genes):
    """Simulate a single-cell signature matrix (Genes x Cell Types)."""
    cell_types = ['Keratinocyte', 'Fibroblast (COL18A1+)', 'CD4+ T-Cell', 'Macrophage']
    
    # Random baseline expression
    signatures = np.random.exponential(scale=1.0, size=(num_genes, len(cell_types)))
    
    # Add specific markers
    # Keratinocyte marker (e.g. KRT14)
    signatures[0, 0] += 10.0 
    # Fibroblast marker (e.g. COL18A1)
    signatures[1, 1] += 10.0
    # T-Cell marker (e.g. CD3D)
    signatures[2, 2] += 10.0
    # Macrophage marker (e.g. CD68)
    signatures[3, 3] += 10.0
    
    return signatures, cell_types

def generate_spatial_mixtures(signatures, num_spots=400):
    """Simulate inferred ST data by mixing cell types spatially."""
    grid_size = int(np.sqrt(num_spots))
    x, y = np.meshgrid(np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size))
    
    # Spatially varying true proportions
    prop_keratinocyte = np.exp(- (x - 0.5)**2 / 0.2) * 0.8
    prop_fibroblast = np.exp(- (x + 0.5)**2 / 0.2) * 0.7
    prop_tcell = np.exp(- ((x + 0.5)**2 + (y - 0.5)**2) / 0.1) * 0.9 # Clustered with fibroblasts
    prop_macrophage = np.random.uniform(0, 0.2, size=x.shape)
    
    true_props = np.stack([prop_keratinocyte, prop_fibroblast, prop_tcell, prop_macrophage], axis=-1)
    true_props = true_props / true_props.sum(axis=-1, keepdims=True)
    
    true_props_flat = true_props.reshape(-1, len(signatures[0]))
    
    # mixture = proportions * signatures
    mixtures = true_props_flat @ signatures.T
    
    # Add noise typical of ST inference
    mixtures += np.random.normal(0, 0.5, size=mixtures.shape)
    mixtures = np.clip(mixtures, 0, None)
    
    coords = np.stack([x.flatten(), y.flatten()], axis=1)
    return mixtures, coords, true_props

def main():
    os.makedirs('paper/figures', exist_ok=True)
    num_genes = 200
    num_spots = 1600 # 40x40 grid
    
    print("Generating single-cell reference and spatial mixtures...")
    signatures, cell_types = generate_mock_reference(num_genes)
    st_mixtures, coords, true_props = generate_spatial_mixtures(signatures, num_spots)
    
    print("Performing spatial deconvolution using Non-Negative Least Squares (NNLS)...")
    predicted_props = np.zeros((num_spots, len(cell_types)))
    for i in range(num_spots):
        # Solves argmin || Sig * p - spot ||_2 s.t. p >= 0
        p, _ = nnls(signatures, st_mixtures[i])
        predicted_props[i] = p
        
    # Normalize proportions
    predicted_props = predicted_props / (predicted_props.sum(axis=1, keepdims=True) + 1e-8)
    
    # Plotting
    grid_size = int(np.sqrt(num_spots))
    fig, axes = plt.subplots(1, len(cell_types), figsize=(16, 4))
    
    for i, ct in enumerate(cell_types):
        prop_grid = predicted_props[:, i].reshape(grid_size, grid_size)
        im = axes[i].imshow(prop_grid, cmap='magma', origin='lower')
        axes[i].set_title(ct)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
    fig.suptitle('Spatial Deconvolution: Predicting Cell Type Proportions from Inferred Transcriptomics', fontsize=16)
    fig.tight_layout()
    plt.savefig('paper/figures/deconvolution_map.pdf', bbox_inches='tight', dpi=300)
    print("Saved deconvolution maps to paper/figures/deconvolution_map.pdf")

if __name__ == "__main__":
    main()
