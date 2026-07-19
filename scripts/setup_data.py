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
    num_spots = 22788
    num_genes = 500
    print(f"Synthesizing h5ad matrix with {num_spots} spots x {num_genes} genes...")
    
    # Create fake count data (ZINB distributed roughly)
    X = np.random.negative_binomial(1, 0.5, size=(num_spots, num_genes))
    
    # Create spatial coordinates (within a 2000x2000 image)
    px_x = np.random.randint(100, 1900, size=num_spots)
    px_y = np.random.randint(100, 1900, size=num_spots)
    
    # Distribute across 5 patients, each with 2 slides
    patients = [f"Patient_{i}" for i in range(5)]
    slide_ids = [f"Slide_{i}_A" for i in range(5)] + [f"Slide_{i}_B" for i in range(5)]
    
    obs = pd.DataFrame({
        'patient_id': np.random.choice(patients, size=num_spots),
        'slide_id': np.random.choice(slide_ids, size=num_spots),
        'px_x': px_x,
        'px_y': px_y
    })
    
    adata = ad.AnnData(X=X, obs=obs)
    adata.write_h5ad(h5ad_path)
    print(f"Saved {h5ad_path}")
    
    # Copy the reference image for each slide_id
    for slide in slide_ids:
        src = os.path.join(images_dir, "reference_image.tif")
        dst = os.path.join(images_dir, f"{slide}_HE.tif")
        if os.path.exists(src):
            import shutil
            shutil.copy(src, dst)
            
    # 3. Create splits.csv
    print("Creating splits.csv...")
    splits_df = pd.DataFrame({
        'patient_id': patients,
        'fold_assignment': ['train', 'train', 'train', 'train', 'test']
    })
    splits_df.to_csv(splits_path, index=False)
    print("Data synthesis complete!")

if __name__ == "__main__":
    setup_synthetic_clinical_data()
