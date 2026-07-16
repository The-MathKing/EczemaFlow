import torch
import torch.optim as optim
from eczema_flow.data import get_dataloaders
from eczema_flow.model import EczemaFlowModel
from tqdm import tqdm
import time

def main():
    print("Initializing EczemaFlow Framework...")
    torch.set_num_threads(7)
    
    # Hyperparameters
    batch_size = 4
    num_samples = 16 # Tiny dataset for dry run
    num_genes = 100
    cond_dim = 128
    num_experts = 3
    epochs = 2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print("Loading mock dataset...")
    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size, 
        num_samples=num_samples, 
        num_genes=num_genes
    )
    
    # Model
    print("Building model...")
    model = EczemaFlowModel(
        num_genes=num_genes, 
        cond_dim=cond_dim, 
        num_experts=num_experts, 
        device=device
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print("Starting training loop (dry run)...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()
        for batch_idx, (patches, counts, coords) in enumerate(train_loader):
            patches = patches.to(device)
            counts = counts.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass & Flow Matching Loss
            loss = model.compute_loss(patches, counts)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Time: {time.time() - start_time:.2f}s")
        
    print("Testing ODE sampling...")
    model.eval()
    with torch.no_grad():
        sample_patches, _, _ = next(iter(val_loader))
        sample_patches = sample_patches.to(device)
        
        try:
            # Sample using 10 Euler steps
            generated_st = model.sample(sample_patches, num_steps=10)
            print(f"Successfully generated ST expression of shape: {generated_st.shape}")
        except ImportError:
            print("Skipping ODE sampling test because torchdiffeq is not installed.")

if __name__ == "__main__":
    main()
