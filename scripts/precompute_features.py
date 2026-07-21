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

def precompute_features(fold, cnn_encoder, tda_extractor, device, num_genes=500, batch_size=32):
    print(f"\nInitializing feature pre-computation for fold: {fold}")
    
    try:
        dataset = VisiumDataset(data_path="data", split_manifest="data/splits.csv", fold=[fold], num_genes=num_genes)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Cannot precompute features because the raw clinical dataset is missing.")
        return
        
    all_combined_features = []
    all_counts = []
    all_coords = []
    
    with torch.no_grad():
        for patches, counts, coords in tqdm(loader, desc=f"Extracting {fold}"):
            patches = patches.to(device)
            b, n, c, h, w = patches.shape
            
            patches_flat = patches.view(b * n, c, h, w)
            patch_features = cnn_encoder(patches_flat)
            
            tda_features = tda_extractor(patches)
            tda_features_flat = tda_features.view(b * n, -1)
            
            combined_features = torch.cat([patch_features, tda_features_flat], dim=-1)
            combined_features = combined_features.view(b, n, -1)
            
            all_combined_features.append(combined_features.cpu())
            all_counts.append(counts.cpu())
            all_coords.append(coords.cpu())
            
    final_features = torch.cat(all_combined_features, dim=0)
    final_counts = torch.cat(all_counts, dim=0)
    final_coords = torch.cat(all_coords, dim=0)
    
    os.makedirs('data/precomputed', exist_ok=True)
    out_path = f"data/precomputed/{fold}_features.pt"
    print(f"Saving {final_features.shape[0]} spots to {out_path}...")
    torch.save({
        'features': final_features,
        'counts': final_counts,
        'coords': final_coords
    }, out_path)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    cnn_encoder = MorphologyEncoder(embed_dim=256, use_pretrained=True).to(device)
    cnn_encoder.eval()
    tda_extractor = TDAFeatureExtractor(output_dim=64).to(device)
    tda_extractor.eval()
    
    fold = sys.argv[1] if len(sys.argv) > 1 else 'train'
    precompute_features(fold, cnn_encoder, tda_extractor, device, batch_size=64)
