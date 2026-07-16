import os
import urllib.request
import tarfile
import gzip
import shutil

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Failed to download {url}. Error: {e}")
        return False

def extract_tar_gz(file_path, extract_dir):
    print(f"Extracting {file_path} to {extract_dir}...")
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print("Extraction complete.")
        return True
    except Exception as e:
        print(f"Failed to extract {file_path}. Error: {e}")
        return False

def main():
    # GSE206391 is the Atopic Dermatitis dataset.
    # We target a specific lesional sample, e.g., GSM6251268 (AD Lesional Rep 1).
    geo_acc = "GSM6251268"
    base_url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={geo_acc}&format=file&file="
    
    # Typically, Visium GEO uploads contain:
    # 1. filtered_feature_bc_matrix.h5
    # 2. spatial.tar.gz
    
    output_dir = os.path.join("data", geo_acc)
    os.makedirs(output_dir, exist_ok=True)
    
    h5_filename = f"{geo_acc}_filtered_feature_bc_matrix.h5"
    h5_url = base_url + h5_filename
    h5_path = os.path.join(output_dir, "filtered_feature_bc_matrix.h5")
    
    spatial_filename = f"{geo_acc}_spatial.tar.gz"
    spatial_url = base_url + spatial_filename
    spatial_tar_path = os.path.join(output_dir, spatial_filename)
    
    # Attempt download
    h5_success = download_file(h5_url, h5_path)
    spatial_success = download_file(spatial_url, spatial_tar_path)
    
    if spatial_success:
        spatial_dir = os.path.join(output_dir, "spatial")
        os.makedirs(spatial_dir, exist_ok=True)
        extract_tar_gz(spatial_tar_path, output_dir)
        
    if h5_success and spatial_success:
        print(f"\\nSuccessfully acquired {geo_acc} from GEO! Ready for processing.")
    else:
        print("\\nNOTE: Automated GEO downloads can occasionally be blocked or require specific credentials.")
        print("If this failed, you can manually download the supplementary files from:")
        print(f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={geo_acc}")

if __name__ == "__main__":
    main()
