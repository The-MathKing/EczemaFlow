import torch
import sys
from eczema_flow.data import get_dataloaders
from eczema_flow.model import EczemaFlowModel

def main():
    print("Verifying Clinical Dataset Integration...")
    try:
        train_loader, _ = get_dataloaders(batch_size=4, num_samples=32)
        batch = next(iter(train_loader))
        patches, counts, coords = batch
        
        print(f"Loaded Clinical Batch! Patches shape: {patches.shape}, Counts shape: {counts.shape}")
        
        device = 'cpu'
        model = EczemaFlowModel(num_genes=counts.shape[1], num_experts=2, device=device)
        
        print("Running Forward Pass Integration Test...")
        loss = model.compute_loss(patches, counts)
        print(f"Integration Successful! Loss: {loss.item():.4f}")
        sys.exit(0)
    except Exception as e:
        print(f"Integration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
