import scanpy as sc
import numpy as np
from scipy.interpolate import griddata
from skimage.metrics import structural_similarity as ssim

def calc_ssim(adata_path, pred_path):
    adata = sc.read_h5ad(adata_path)
    adata.var_names_make_unique()
    adata_test_full = adata[adata.obs['patient'].astype(str) == '8'].copy()
    test_slide = 'P16357_1028'
    
    predicted_genes = adata_test_full.var_names[:500]
    preds = np.load(pred_path)
    
    markers = ['COL18A1', 'CCL19', 'KRT14', 'ERBB2', 'CD3D']
    markers = [m for m in markers if m in predicted_genes]
    
    adata_plot = adata_test_full[adata_test_full.obs['sample'] == test_slide].copy()
    
    # Get physical coordinates
    coords = adata_plot.obsm['spatial']
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Create grid
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    
    results = {}
    for m in markers:
        # Observed
        obs_vals = adata_plot[:, m].X.toarray().flatten()
        
        # Predicted
        idx = predicted_genes.get_loc(m)
        pred_vals = preds[adata_test_full.obs['sample'] == test_slide, idx]
        
        # Interpolate to grid
        grid_obs = griddata((x, y), obs_vals, (grid_x, grid_y), method='cubic', fill_value=0)
        grid_pred = griddata((x, y), pred_vals, (grid_x, grid_y), method='cubic', fill_value=0)
        
        # Normalize to 0-1 range for SSIM
        grid_obs_norm = (grid_obs - grid_obs.min()) / (grid_obs.max() - grid_obs.min() + 1e-8)
        grid_pred_norm = (grid_pred - grid_pred.min()) / (grid_pred.max() - grid_pred.min() + 1e-8)
        
        # Calculate SSIM
        score, _ = ssim(grid_obs_norm, grid_pred_norm, full=True, data_range=1.0)
        results[m] = score
        print(f"{m} SSIM: {score:.3f}")

if __name__ == "__main__":
    calc_ssim('data/GSE206391_spatial.h5ad', 'results/EczemaFlow_test_preds.npy')
