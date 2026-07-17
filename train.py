import torch
import torch.optim as optim
from eczema_flow.data import get_dataloaders
from eczema_flow.model import EczemaFlowModel
from tqdm import tqdm
import time
import os

def main():
    print("Initializing EczemaFlow Framework...")
    # Set to optimal thread count for HPC
    torch.set_num_threads(8)
    
    # Robust Hyperparameters for Full Scale Training
    batch_size = 64
    num_genes = 500
    cond_dim = 128
    num_experts = 4
    epochs = 500
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data - load full dataset without dry-run cap
    print("Loading multi-slide dataset...")
    try:
        train_loader, val_loader = get_dataloaders(
            batch_size=batch_size, 
            num_samples=None, # Load all samples
            num_genes=num_genes
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python process_geo_ad.py' first to compile the dataset.")
        return
        
    print(f"Loaded {len(train_loader.dataset)} training spots and {len(val_loader.dataset)} validation spots.")
    
    # Model
    print("Building EczemaFlow model...")
    model = EczemaFlowModel(
        num_genes=num_genes, 
        cond_dim=cond_dim, 
        num_experts=num_experts, 
        device=device
    ).to(device)
    
    # Using AdamW as specified in the paper
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Create checkpoints dir
    os.makedirs("checkpoints", exist_ok=True)
    
    print("Starting full training loop...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        start_time = time.time()
        
        # Training
        for batch_idx, (patches, counts, coords) in enumerate(train_loader):
            patches = patches.to(device)
            counts = counts.to(device)
            
            optimizer.zero_grad()
            loss = model.compute_loss(patches, counts)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for patches, counts, coords in val_loader:
                patches = patches.to(device)
                counts = counts.to(device)
                val_loss = model.compute_loss(patches, counts)
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Time: {time.time() - start_time:.2f}s")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("  -> Checkpoint saved!")

    print("Training complete.")

if __name__ == "__main__":
    main()
