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
    
    # Correct provenance mapping for the 6 AD patients based on GSE206391 metadata
    gsm_mapping = {
        1: {'samples': ['P15509_1001', 'P15509_1002', 'P15509_1003', 'P15509_1004'], 'cond': 'Lesional AD', 'prefix': 'V19523-003'},
        2: {'samples': ['P16357_1001', 'P16357_1002', 'P16357_1003', 'P16357_1004'], 'cond': 'Lesional AD', 'prefix': '2-V19S23-004'},
        3: {'samples': ['P16357_1005', 'P16357_1006', 'P16357_1007', 'P16357_1008'], 'cond': 'Lesional AD', 'prefix': '3-V19S23-005'},
        5: {'samples': ['P16357_1013', 'P16357_1014', 'P16357_1015', 'P16357_1016'], 'cond': 'Non-lesional AD', 'prefix': '5-V19S18-093'},
        6: {'samples': ['P16357_1017', 'P16357_1018', 'P16357_1019', 'P16357_1020'], 'cond': 'Non-lesional AD', 'prefix': '6-V19T12-047'},
        8: {'samples': ['P16357_1025', 'P16357_1026', 'P16357_1027', 'P16357_1028'], 'cond': 'Non-lesional AD', 'prefix': '8-V19T12-006'}
    }
    
    target_slides = []
    for meta in gsm_mapping.values():
        target_slides.extend(meta['samples'])
        
    slide_col = 'sample'
    adata_sub = adata[adata.obs[slide_col].isin(target_slides)].copy()
    
    print(f"Selected {len(target_slides)} slides across 6 patients.")
    
    # Create splits.csv with patient cross-validation folds (1 to 6)
    splits_data = []
    for i, (patient, meta) in enumerate(gsm_mapping.items()):
        for sample in meta['samples']:
            splits_data.append({
                'patient_id': patient,
                'slide_id': sample,
                'condition': meta['cond'],
                'fold_assignment': f'fold_{i+1}'
            })
    
    splits_df = pd.DataFrame(splits_data)
    splits_df.to_csv(splits_path, index=False)
    
    # Ensure there are exactly 500 genes for the benchmark scripts
    if adata_sub.n_vars > 500:
        import numpy as np
        X = adata_sub.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-500:]
        adata_sub = adata_sub[:, top_indices].copy()
        
    print(f"Saving filtered AnnData ({adata_sub.n_obs} spots, {adata_sub.n_vars} genes)...")
    adata_sub.write_h5ad(out_h5ad)
    
    # Extract images for these 24 samples
    print("Extracting JPG images from RAW tar...")
    
    if os.path.exists(tar_path):
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()
            
            for patient, meta in gsm_mapping.items():
                prefix = meta['prefix']
                for i, sample in enumerate(meta['samples']):
                    v_suffix = f"V{i+1}"
                    # Find matching gzip member by prefix and suffix
                    match = next((m for m in members if prefix in m.name and v_suffix in m.name), None)
                        
                    if match:
                        gz_name = match.name
                        tar.extract(match, path=images_dir)
                        gz_path = os.path.join(images_dir, gz_name)
                        tif_path = os.path.join(images_dir, f"{sample}_HE.tif")
                        
                        with gzip.open(gz_path, 'rb') as f_in:
                            with open(tif_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        
                        os.remove(gz_path)
                        print(f"Extracted image {gz_name} for patient {patient} (Slide {sample})")
                    else:
                        print(f"Warning: Could not find image for {sample} (prefix={prefix}, suffix={v_suffix}) in tar")
    
    print("Real data setup complete.")

if __name__ == "__main__":
    setup_real_data()
