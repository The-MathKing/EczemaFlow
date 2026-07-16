import os
import scanpy as sc
import matplotlib.pyplot as plt

def generate_visuals(data_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading data from {data_path}...")
    adata = sc.read_h5ad(data_path)
    
    # Basic QC metrics
    adata.var_names_make_unique()
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    
    # 1. Spatial overlay with total counts
    print("Generating Spatial Plot (Total Counts)...")
    sc.pl.spatial(adata, color="total_counts", show=False)
    plt.savefig(os.path.join(output_dir, 'spatial_total_counts.pdf'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Spatial overlay with number of genes
    print("Generating Spatial Plot (Number of Genes)...")
    sc.pl.spatial(adata, color="n_genes_by_counts", show=False)
    plt.savefig(os.path.join(output_dir, 'spatial_n_genes.pdf'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. Simple H&E image without spots (just the tissue)
    print("Generating H&E reference image...")
    sc.pl.spatial(adata, color=None, alpha=0.0, show=False)
    plt.savefig(os.path.join(output_dir, 'he_reference.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Visuals successfully saved to {output_dir}")

if __name__ == "__main__":
    dataset_name = 'V1_Breast_Cancer_Block_A_Section_1'
    data_path = f'./data/{dataset_name}.h5ad'
    output_dir = './paper/figures'
    
    if os.path.exists(data_path):
        generate_visuals(data_path, output_dir)
    else:
        print(f"Data file {data_path} not found. Please wait for the download to complete.")
