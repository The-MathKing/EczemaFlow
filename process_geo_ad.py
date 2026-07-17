import os
import scanpy as sc
import pandas as pd

def process_geo_dataset(geo_accessions, output_filename="GSE206391_AD_MultiSlide.h5ad"):
    output_path = os.path.join("data", output_filename)
    adata_list = []
    
    for geo_acc in geo_accessions:
        data_dir = os.path.join("data", geo_acc)
        h5_path = os.path.join(data_dir, "filtered_feature_bc_matrix.h5")
        spatial_dir = os.path.join(data_dir, "spatial")
        
        if os.path.exists(h5_path) and os.path.exists(spatial_dir):
            print(f"Found raw SpaceRanger files for {geo_acc}. Compiling AnnData...")
            try:
                # Load raw Visium format
                adata = sc.read_visium(data_dir)
                adata.var_names_make_unique()
                
                # Tag with slide ID
                adata.obs['slide_id'] = geo_acc
                
                # Optional: Basic QC filtering
                sc.pp.filter_cells(adata, min_counts=500)
                sc.pp.filter_genes(adata, min_cells=10)
                
                adata_list.append(adata)
                print(f"Successfully processed {geo_acc}.")
            except Exception as e:
                print(f"Failed to compile AnnData for {geo_acc}: {e}")
        else:
            print(f"WARNING: Raw spatial files for {geo_acc} not found in {data_dir}. Skipping.")
            
    if not adata_list:
        raise RuntimeError("CRITICAL ERROR: No slides were successfully processed. Check data directory.")
        
    print("Concatenating multi-slide cohort...")
    combined_adata = adata_list[0].concatenate(adata_list[1:], batch_key="batch_slide")
    
    print(f"Saving compiled massive AnnData to {output_path}")
    combined_adata.write(output_path)
    return True

if __name__ == "__main__":
    # Define a robust 6-slide cohort (3 lesional, 3 non-lesional)
    cohort = ["GSM6251268", "GSM6251269", "GSM6251270", "GSM6251271", "GSM6251272", "GSM6251273"]
    process_geo_dataset(cohort)
