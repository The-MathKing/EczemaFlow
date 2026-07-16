import os
import scanpy as sc

def download_visium_dataset(dataset_name='V1_Breast_Cancer_Block_A_Section_1', save_dir='./data'):
    """
    Downloads a 10x Genomics Visium dataset using scanpy.
    Available examples:
    - 'V1_Breast_Cancer_Block_A_Section_1'
    - 'V1_Human_Lymph_Node'
    - 'V1_Mouse_Brain_Sagittal_Anterior'
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Downloading Visium dataset: {dataset_name}...")
    # This downloads the spatial dataset (AnnData with .obsm['spatial'] and .uns['spatial'])
    adata = sc.datasets.visium_sge(sample_id=dataset_name, include_hires_tiff=True)
    
    # Save the h5ad file
    save_path = os.path.join(save_dir, f"{dataset_name}.h5ad")
    print(f"Saving AnnData object to {save_path}...")
    adata.write_h5ad(save_path)
    print("Download complete!")
    return adata

if __name__ == "__main__":
    # Let's download a breast cancer spatial transcriptomics slide
    # since public AD Visium datasets are massive and usually hosted on custom portals (GEO/Zenodo).
    # This provides a perfect surrogate for training the image-to-ST flow model.
    download_visium_dataset('V1_Breast_Cancer_Block_A_Section_1')
