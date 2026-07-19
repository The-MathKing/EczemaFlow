import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from eczema_flow.dataset import PrecomputedVisiumDataset
from eczema_flow.model import EczemaFlowModel
from tqdm import tqdm
import time
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
torch.set_num_threads(7)

def main():
    print("Initializing EczemaFlow Framework (Optimized M4 Pipeline)...")
    
    # Production Hyperparameters for Cluster Training
    batch_size = 64
    num_genes = 500
    cond_dim = 256
    num_experts = 4
    epochs = 500
    
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Data - load lightweight precomputed embeddings instead of raw images
    print("Loading lightweight precomputed dataset...")
    try:
        train_dataset = PrecomputedVisiumDataset("data/precomputed/train_features.pt")
        val_dataset = PrecomputedVisiumDataset("data/precomputed/test_features.pt")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run scripts/precompute_features.py first.")
        return
        
    print(f"Loaded {len(train_dataset)} training spots and {len(val_dataset)} validation spots.")
    
    # Model
    print("Building EczemaFlow model...")
    model = EczemaFlowModel(
        num_genes=num_genes, 
        cond_dim=cond_dim, 
        num_experts=num_experts, 
        device=device
    ).to(device)
    
    # Using AdamW as specified in the paper
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    
    # Create checkpoints dir
    os.makedirs("checkpoints", exist_ok=True)
    
    start_epoch = 0
    checkpoint_path = "checkpoints/latest_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming at epoch {start_epoch + 1}")
    
    print("Starting full training loop...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, epochs):
        model.train()
        total_train_loss = 0
        start_time = time.time()
        
        # Training
        for batch_idx, (patches, counts, coords) in enumerate(train_loader):
            patches = patches.to(device)
            counts = counts.to(device)
            
            optimizer.zero_grad()
            loss = model.compute_loss(patches, counts, is_precomputed=True)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Sparse Validation (Evaluate every 25 epochs to save massive ODE compute time)
        if (epoch + 1) % 25 == 0 or epoch == epochs - 1:
            model.eval()
            total_val_loss = 0
            batches_evaluated = 0
            
            print(f"Epoch [{epoch+1}/{epochs}] - Running sparse ODE evaluation...")
            with torch.no_grad():
                for batch_idx, (patches, counts, coords) in enumerate(val_loader):
                    # Subsample validation set (10%) to estimate loss rapidly
                    if batch_idx % 10 != 0:
                        continue
                        
                    patches = patches.to(device)
                    counts = counts.to(device)
                    val_loss = model.compute_loss(patches, counts, is_precomputed=True)
                    total_val_loss += val_loss.item()
                    batches_evaluated += 1
                    
            avg_val_loss = total_val_loss / max(batches_evaluated, 1)
            
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss (Sparse): {avg_val_loss:.4f} - Time: {time.time() - start_time:.2f}s")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "checkpoints/best_model.pth")
                print("  -> Best validation checkpoint saved!")
        else:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Time: {time.time() - start_time:.2f}s")
            
        # Save latest checkpoint for resuming
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, checkpoint_path)

    print("Training complete.")

if __name__ == "__main__":
    main()
