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
    
    # 3. Biological Validation: Mapping Specific Marker Genes
    print("Generating Biological Marker Gene Maps...")
    # Using Visium Breast Cancer dataset as a surrogate for spatial pipeline mapping
    # We will map standard structural/immune markers as proof of concept.
    markers = []
    for gene in ['COL18A1', 'CCL19', 'KRT14', 'ERBB2', 'CD3D']:
        if gene in adata.var_names:
            markers.append(gene)
            
    if markers:
        sc.pl.spatial(adata, color=markers, show=False)
        plt.savefig(os.path.join(output_dir, 'spatial_markers.pdf'), bbox_inches='tight', dpi=300)
        plt.close()
    
    # 4. Simple H&E image without spots (just the tissue)
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
