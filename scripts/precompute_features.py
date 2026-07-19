import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

# Add parent directory to path so we can import eczema_flow
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eczema_flow.dataset import VisiumDataset
from eczema_flow.attention import MorphologyEncoder
from eczema_flow.tda import TDAFeatureExtractor

def precompute_features(fold='train', num_genes=500, batch_size=32):
    print(f"Initializing feature pre-computation for fold: {fold}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # 1. Load the heavy raw image dataset
    try:
        dataset = VisiumDataset(data_path="data", split_manifest="data/splits.csv", fold=fold, num_genes=num_genes)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Cannot precompute features because the raw clinical dataset is missing.")
        return
        
    # 2. Initialize the frozen encoders
    print("Loading ResNet50 Morphology Encoder and TDA Extractor...")
    cnn_encoder = MorphologyEncoder(embed_dim=256, use_pretrained=True).to(device)
    cnn_encoder.eval() # Must be in eval mode
    
    tda_extractor = TDAFeatureExtractor(output_dim=64).to(device)
    tda_extractor.eval()
    
    # 3. Store the extracted features
    all_combined_features = []
    all_counts = []
    all_coords = []
    
    print("Extracting features (this will run once and save hours of training time)...")
    with torch.no_grad():
        for patches, counts, coords in tqdm(loader):
            patches = patches.to(device)
            b, n, c, h, w = patches.shape
            
            # Pass through CNN
            patches_flat = patches.view(b * n, c, h, w)
            patch_features = cnn_encoder(patches_flat) # (b*n, 256)
            
            # Pass through TDA
            tda_features = tda_extractor(patches) # (b, n, 64)
            tda_features_flat = tda_features.view(b * n, -1)
            
            # Concatenate to get the final (b*n, 320) vectors that the ViT expects
            combined_features = torch.cat([patch_features, tda_features_flat], dim=-1)
            
            # Reshape back to (b, n, 320)
            combined_features = combined_features.view(b, n, -1)
            
            all_combined_features.append(combined_features.cpu())
            all_counts.append(counts.cpu())
            all_coords.append(coords.cpu())
            
    # Concatenate all batches
    final_features = torch.cat(all_combined_features, dim=0)
    final_counts = torch.cat(all_counts, dim=0)
    final_coords = torch.cat(all_coords, dim=0)
    
    # 4. Save to disk
    os.makedirs('data/precomputed', exist_ok=True)
    out_path = f"data/precomputed/{fold}_features.pt"
    print(f"Saving {final_features.shape[0]} spots to {out_path}...")
    torch.save({
        'features': final_features,
        'counts': final_counts,
        'coords': final_coords
    }, out_path)
    
    print("Pre-computation complete!")

if __name__ == "__main__":
    precompute_features(fold='train')
    precompute_features(fold='test')
