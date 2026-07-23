"""
Download GSE197023 (Mitamura et al., 2023) - External AD Visium Cohort
Swiss Institute of Allergy, Davos -- completely different institution from GSE206391.

14 AD samples (7 patients, lesional + non-lesional) + 6 HC samples.
We use only the 14 AD samples for external validation (matching our AD-only scope).

GEO samples: GSM5907077 - GSM5907090 (14 AD samples)
             GSM5907091 - GSM5907096 (6 HC samples, skipped)
"""
import os
import urllib.request
import tarfile
import time

# AD-only GSM sample IDs (7 patients x 2 = 14 slides)
AD_SAMPLES = [
    "GSM5907077",  # Patient AD-1, lesional
    "GSM5907078",  # Patient AD-1, non-lesional
    "GSM5907079",  # Patient AD-2, lesional
    "GSM5907080",  # Patient AD-2, non-lesional
    "GSM5907081",  # Patient AD-3, lesional
    "GSM5907082",  # Patient AD-3, non-lesional
    "GSM5907083",  # Patient AD-4, lesional
    "GSM5907084",  # Patient AD-4, non-lesional
    "GSM5907085",  # Patient AD-5, lesional
    "GSM5907086",  # Patient AD-5, non-lesional
    "GSM5907087",  # Patient AD-6, lesional
    "GSM5907088",  # Patient AD-6, non-lesional
    "GSM5907089",  # Patient AD-7, lesional
    "GSM5907090",  # Patient AD-7, non-lesional
]

OUTPUT_BASE = os.path.join("data", "GSE197023")
os.makedirs(OUTPUT_BASE, exist_ok=True)

def download_file(url, dest_path, retries=3):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000:
        print(f"  [SKIP] Already downloaded: {os.path.basename(dest_path)}")
        return True
    for attempt in range(retries):
        try:
            print(f"  Downloading (attempt {attempt+1}): {url}")
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; Python urllib)'
            })
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, 'wb') as f:
                total = 0
                while True:
                    chunk = resp.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
                    print(f"    {total / 1e6:.1f} MB downloaded...", end='\r')
            print(f"\n  Done: {os.path.basename(dest_path)} ({total / 1e6:.1f} MB)")
            return True
        except Exception as e:
            print(f"\n  Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return False

def extract_tar(tar_path, extract_to):
    print(f"  Extracting {os.path.basename(tar_path)}...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
        print(f"  Extracted to {extract_to}")
        return True
    except Exception:
        try:
            with tarfile.open(tar_path, "r:*") as tar:
                tar.extractall(path=extract_to)
            print(f"  Extracted to {extract_to}")
            return True
        except Exception as e2:
            print(f"  Extraction failed: {e2}")
            return False

def main():
    # First, try downloading the full supplementary tar for GSE197023
    print("="*60)
    print("Downloading GSE197023 (Mitamura et al. 2023, Swiss cohort)")
    print("External validation cohort for EczemaFlow")
    print("="*60)

    # Try the bulk download first
    bulk_url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE197023&format=file"
    bulk_tar = os.path.join(OUTPUT_BASE, "GSE197023_RAW.tar")
    
    print("\n[1/2] Attempting bulk GEO download (all supplementary files)...")
    success = download_file(bulk_url, bulk_tar)
    
    if success and os.path.getsize(bulk_tar) > 100000:
        print("\n[2/2] Extracting bulk download...")
        extract_tar(bulk_tar, OUTPUT_BASE)
    else:
        # Fall back to per-sample downloads
        print("\n[2/2] Bulk download failed, trying per-sample downloads...")
        for gsm in AD_SAMPLES:
            sample_dir = os.path.join(OUTPUT_BASE, gsm)
            os.makedirs(sample_dir, exist_ok=True)
            
            # Try to get the filtered feature-barcode matrix
            for filename, suffix in [
                ("filtered_feature_bc_matrix.h5", "_filtered_feature_bc_matrix.h5.gz"),
                ("spatial.tar.gz", "_spatial.tar.gz"),
            ]:
                url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={gsm}&format=file&file={gsm}{suffix}"
                dest = os.path.join(sample_dir, filename)
                download_file(url, dest)
    
    print("\n" + "="*60)
    print("Download complete. Files are in:", OUTPUT_BASE)
    print("Next: Run preprocess_gse197023.py to build AnnData objects")
    print("="*60)

if __name__ == "__main__":
    main()
