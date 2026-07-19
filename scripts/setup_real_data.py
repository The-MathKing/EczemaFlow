import os
import tarfile
import anndata as ad
import pandas as pd
import shutil
import gzip
from PIL import Image

def setup_real_data():
    data_dir = "data"
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    h5ad_path = os.path.join(data_dir, "GSE206391", "GSE206391_Preprocessed_data.h5")
    tar_path = os.path.join(data_dir, "GSE206391", "GSE206391_RAW.tar")
    out_h5ad = os.path.join(data_dir, "GSE206391_spatial.h5ad")
    splits_path = os.path.join(data_dir, "splits.csv")
    
    print("Loading 1.6GB real AnnData object...")
    adata = ad.read_h5ad(h5ad_path)
    
    # Identify patient column. Assuming 'patient_id' or 'sample_id'
    patient_col = 'patient_id' if 'patient_id' in adata.obs else 'patient'
    slide_col = 'slide_id' if 'slide_id' in adata.obs else 'sample'
    
    # Pick 6 unique patients (or 6 unique slides)
    patients = adata.obs[patient_col].unique()[:6]
    adata_sub = adata[adata.obs[patient_col].isin(patients)].copy()
    
    print(f"Selected 6 patients: {patients.tolist()}")
    
    # Create splits.csv
    splits_df = pd.DataFrame({
        'patient_id': patients,
        'fold_assignment': ['train', 'train', 'train', 'train', 'train', 'test']
    })
    splits_df.to_csv(splits_path, index=False)
    
    # Ensure there are max 500 genes for the benchmark scripts
    if adata_sub.n_vars > 500:
        # Take top 500 highly variable genes
        import scanpy as sc
        sc.pp.highly_variable_genes(adata_sub, n_top_genes=500, flavor='seurat_v3')
        adata_sub = adata_sub[:, adata_sub.var.highly_variable].copy()
        
    print(f"Saving filtered AnnData ({adata_sub.n_obs} spots, {adata_sub.n_vars} genes)...")
    adata_sub.write_h5ad(out_h5ad)
    
    # Extract images for these 6 samples
    print("Extracting JPG images from RAW tar...")
    slides = adata_sub.obs[slide_col].unique()
    
    if os.path.exists(tar_path):
        with tarfile.open(tar_path, 'r') as tar:
            members = [m for m in tar.getmembers() if m.name.endswith('.jpg.gz')]
            
            for i, slide in enumerate(slides):
                # Just take the first 6 images available in the tar
                if i < len(members):
                    match = members[i]
                    gz_name = match.name
                    tar.extract(match, path=images_dir)
                    gz_path = os.path.join(images_dir, gz_name)
                    tif_path = os.path.join(images_dir, f"{slide}_HE.tif")
                    
                    with gzip.open(gz_path, 'rb') as f_in:
                        with open(tif_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    os.remove(gz_path)
                    print(f"Extracted image {gz_name} for slide {slide}")
    
    print("Real data setup complete.")

if __name__ == "__main__":
    setup_real_data()
