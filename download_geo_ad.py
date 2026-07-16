import os
import urllib.request
import tarfile

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    try:
        # User-Agent is sometimes required for NCBI
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Failed to download {url}. Error: {e}")
        return False

def extract_tar(file_path, extract_dir):
    print(f"Extracting {file_path} to {extract_dir}...")
    try:
        with tarfile.open(file_path, "r") as tar:
            tar.extractall(path=extract_dir)
        print("Extraction complete.")
        return True
    except Exception as e:
        print(f"Failed to extract {file_path}. Error: {e}")
        return False

def main():
    # GSE206391 Atopic Dermatitis dataset
    base_url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE206391&format=file&file="
    
    output_dir = os.path.join("data", "GSE206391")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Preprocessed H5 (1.5 GB)
    h5_filename = "GSE206391_Preprocessed_data.h5"
    h5_url = base_url + h5_filename
    h5_path = os.path.join(output_dir, h5_filename)
    
    # 2. RAW Spatial JPGs (7.7 GB)
    tar_filename = "GSE206391_RAW.tar"
    tar_url = base_url + tar_filename
    tar_path = os.path.join(output_dir, tar_filename)
    
    # Attempt download
    print("Initiating massive GEO download (this may take over an hour depending on bandwidth)...")
    download_file(h5_url, h5_path)
    
    spatial_success = download_file(tar_url, tar_path)
    if spatial_success:
        extract_tar(tar_path, output_dir)
        
    print(f"\\nDownload pipeline complete. Files saved to {output_dir}")

if __name__ == "__main__":
    main()
