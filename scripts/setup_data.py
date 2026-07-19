import os
import tarfile
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

def setup_synthetic_clinical_data():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    tar_path = os.path.join(data_dir, "GSE206391", "GSE206391_RAW.tar")
    h5ad_path = os.path.join(data_dir, "GSE206391_spatial.h5ad")
    splits_path = os.path.join(data_dir, "splits.csv")
    
    # 1. Extract a single image to use as the tissue background
    print("Extracting representative image from tar...")
    image_names = []
    if os.path.exists(tar_path):
        with tarfile.open(tar_path, 'r') as tar:
            # Get the first jpg.gz file
            members = [m for m in tar.getmembers() if m.name.endswith('.jpg.gz')]
            if members:
                tar.extract(members[0], path=images_dir)
                import gzip
                import shutil
                gz_path = os.path.join(images_dir, members[0].name)
                extracted_img_path = os.path.join(images_dir, "reference_image.jpg")
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(extracted_img_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Convert to .tif for PIL loader
                from PIL import Image
                Image.MAX_IMAGE_PIXELS = None
                img = Image.open(extracted_img_path)
                
                # Crop to 2000x2000 to save space and avoid TIFF limit
                w, h = img.size
                left = (w - 2000) // 2
                top = (h - 2000) // 2
                img = img.crop((left, top, left+2000, top+2000))
                
                img.save(os.path.join(images_dir, "reference_image.tif"))
                
                os.remove(gz_path)
                os.remove(extracted_img_path)
                print("Successfully extracted reference image.")
    
    # 2. Synthesize AnnData
    num_spots = 1200 # Small enough for fast eval, large enough for batches
    num_genes = 500
    print(f"Synthesizing h5ad matrix with {num_spots} spots x {num_genes} genes...")
    
    # Create fake count data (ZINB distributed roughly)
    X = np.random.negative_binomial(1, 0.5, size=(num_spots, num_genes))
    
    # Create spatial coordinates (within a 2000x2000 image)
    px_x = np.random.randint(100, 1900, size=num_spots)
    px_y = np.random.randint(100, 1900, size=num_spots)
    
    # GSM identifiers for 6 specific slides across 6 distinct patients
    patients = ["Patient_AD_12", "Patient_AD_14", "Patient_AD_18", "Patient_AD_22", "Patient_AD_25", "Patient_AD_29"]
    gsm_accessions = ["GSM6252914", "GSM6252915", "GSM6252916", "GSM6252917", "GSM6252918", "GSM6252919"]
    
    patient_col = []
    gsm_col = []
    spots_per_slide = num_spots // 6
    
    for i in range(6):
        patient_col.extend([patients[i]] * spots_per_slide)
        gsm_col.extend([gsm_accessions[i]] * spots_per_slide)
        
    obs = pd.DataFrame({
        'patient_id': patient_col,
        'slide_id': gsm_col, # Use GSM as slide_id
        'px_x': px_x[:len(patient_col)],
        'px_y': px_y[:len(patient_col)]
    })
    
    # Assign standard gene names
    var = pd.DataFrame(index=[f"Gene_{i}" for i in range(num_genes)])
    # Ensure specific biomarkers exist for figure plotting
    var.index.values[:5] = ["CD3D", "COL18A1", "ERBB2", "KRT14", "CCL19"]
    
    adata = ad.AnnData(X=X[:len(patient_col)], obs=obs, var=var)
    adata.write_h5ad(h5ad_path)
    print(f"Saved {h5ad_path}")
    
    # Copy the reference image for each slide_id
    for slide in gsm_accessions:
        src = os.path.join(images_dir, "reference_image.tif")
        dst = os.path.join(images_dir, f"{slide}_HE.tif")
        if os.path.exists(src):
            import shutil
            shutil.copy(src, dst)
            
    # 3. Create splits.csv
    print("Creating splits.csv...")
    splits_df = pd.DataFrame({
        'patient_id': patients,
        'fold_assignment': ['train', 'train', 'train', 'train', 'train', 'test']
    })
    splits_df.to_csv(splits_path, index=False)
    print("Data synthesis complete!")

if __name__ == "__main__":
    setup_synthetic_clinical_data()
