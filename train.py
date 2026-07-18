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
    
    # Robust Hyperparameters for Local M-Series Training (30-hour limit)
    batch_size = 16
    num_genes = 150
    cond_dim = 128
    num_experts = 4
    epochs = 50
    
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # 70% Compute Cap Configuration
    compute_utilization_target = 0.70
    print(f"Enforcing strict compute utilization cap: {compute_utilization_target * 100}% duty cycle.")
    
    # Data - load full dataset without dry-run cap
    print("Loading multi-slide dataset...")
    try:
        train_loader, val_loader = get_dataloaders(
            batch_size=batch_size, 
            num_samples=15000, # Subsample highly informative spots
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
            batch_start = time.time()
            
            patches = patches.to(device)
            counts = counts.to(device)
            
            optimizer.zero_grad()
            loss = model.compute_loss(patches, counts)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Enforce 70% Compute Cap via Duty Cycle Sleep
            batch_time = time.time() - batch_start
            sleep_time = batch_time * ((1.0 - compute_utilization_target) / compute_utilization_target)
            time.sleep(sleep_time)
            
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
        
        # Save latest checkpoint for resuming
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
        }, checkpoint_path)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("  -> Best validation checkpoint saved!")

    print("Training complete.")

if __name__ == "__main__":
    main()
