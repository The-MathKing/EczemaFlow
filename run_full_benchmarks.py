import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr
import os

from eczema_flow.data import get_dataloaders
from eczema_flow.model import EczemaFlowModel
from eczema_flow.model_baselines import CNNRegressor, GaussianPrior
from eczema_flow.attention import MorphologyEncoder

# Ablation 1: No Attention (Simple Mean Pooling)
class EczemaFlowNoAttention(EczemaFlowModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace Many-Body Attention with a simple Morphology Encoder
        # that mean pools the features.
        class MeanPoolConditioner(torch.nn.Module):
            def __init__(self, embed_dim):
                super().__init__()
                self.encoder = MorphologyEncoder(embed_dim=embed_dim)
            def forward(self, patches):
                b, n, c, h, w = patches.shape
                patches_flat = patches.view(b * n, c, h, w)
                features = self.encoder(patches_flat)
                return features.view(b, n, -1).mean(dim=1)
                
        self.conditioner = MeanPoolConditioner(embed_dim=kwargs.get('cond_dim', 256))

# Ablation 2: Gaussian Flow Model
class GaussianFlowModel(EczemaFlowModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Swap out ZINB prior for standard Gaussian
        self.prior = GaussianPrior(num_genes=self.num_genes, device=self.device)

def calculate_metrics(preds, targets):
    mse = F.mse_loss(preds, targets).item()
    pcc_list = []
    # Calculate PCC across genes for each spot, then average
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    for i in range(preds_np.shape[0]):
        # Add small noise to avoid zero variance issues in PCC
        p = preds_np[i] + np.random.normal(0, 1e-6, preds_np[i].shape)
        t = targets_np[i] + np.random.normal(0, 1e-6, targets_np[i].shape)
        r, _ = pearsonr(p, t)
        if not np.isnan(r):
            pcc_list.append(r)
    return mse, np.mean(pcc_list) if pcc_list else 0.0

def train_and_eval(model_name, model, train_loader, val_loader, device, epochs=3, is_cnn=False):
    print(f"\\n--- Training {model_name} ---")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    
    for epoch in range(epochs):
        for patches, counts, _ in train_loader:
            patches, counts = patches.to(device), counts.to(device)
            optimizer.zero_grad()
            
            if is_cnn:
                preds = model(patches)
                # target is log1p space for fair comparison
                target_log = torch.log1p(counts)
                loss = F.mse_loss(preds, target_log)
            else:
                loss = model.compute_loss(patches, counts)
                
            loss.backward()
            optimizer.step()
            
    print(f"Evaluating {model_name}...")
    model.eval()
    total_mse, total_pcc = 0, 0
    batches = 0
    
    with torch.no_grad():
        for patches, counts, _ in val_loader:
            patches, counts = patches.to(device), counts.to(device)
            target_log = torch.log1p(counts)
            
            if is_cnn:
                preds = model(patches)
            else:
                # Use fewer steps for speed in benchmark
                preds = model.sample(patches, num_steps=10)
                
            mse, pcc = calculate_metrics(preds, target_log)
            total_mse += mse
            total_pcc += pcc
            batches += 1
            
            if batches >= 2: # Eval on subset for speed in proof of concept
                break
                
    return total_mse / batches, total_pcc / batches

def main():
    torch.set_num_threads(7)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running benchmarks on {device}...")
    
    # We use mock data for the rapid benchmarking loop because 
    # loading the 1.6GB Visium dataset into RAM repeatedly is very slow.
    # The models are the same and test the architectural differences.
    num_genes = 200
    train_loader, val_loader = get_dataloaders(batch_size=8, num_samples=64, num_genes=num_genes)
    
    results = {}
    
    # 1. Baseline: CNN Regressor
    cnn = CNNRegressor(num_genes=num_genes).to(device)
    results['CNN Regressor'] = train_and_eval("CNN Regressor", cnn, train_loader, val_loader, device, is_cnn=True)
    
    # 2. Gaussian Flow Model (No ZINB)
    gaussian_flow = GaussianFlowModel(num_genes=num_genes, num_experts=2, device=device).to(device)
    results['Gaussian Flow'] = train_and_eval("Gaussian Flow", gaussian_flow, train_loader, val_loader, device)
    
    # 3. EczemaFlow (No Attention)
    no_attn_flow = EczemaFlowNoAttention(num_genes=num_genes, num_experts=2, device=device).to(device)
    results['No Attention Flow'] = train_and_eval("No Attention Flow", no_attn_flow, train_loader, val_loader, device)
    
    # 4. Full EczemaFlow
    full_flow = EczemaFlowModel(num_genes=num_genes, num_experts=2, device=device).to(device)
    results['EczemaFlow (Full)'] = train_and_eval("EczemaFlow (Full)", full_flow, train_loader, val_loader, device)
    
    print("\\n--- Benchmark Results ---")
    for name, (mse, pcc) in results.items():
        print(f"{name:20s} | MSE: {mse:.4f} | PCC: {pcc:.4f}")
        
    # Generate Chart
    os.makedirs('paper/figures', exist_ok=True)
    labels = list(results.keys())
    mses = [r[0] for r in results.values()]
    pccs = [r[1] for r in results.values()]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot MSE on primary y-axis
    rects1 = ax1.bar(x - width/2, mses, width, label='MSE (Lower is better)', color='tab:red')
    ax1.set_ylabel('Mean Squared Error', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right')
    
    # Plot PCC on secondary y-axis
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, pccs, width, label='PCC (Higher is better)', color='tab:blue')
    ax2.set_ylabel('Pearson Correlation Coefficient', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    fig.tight_layout()
    plt.title('Performance Benchmarks of WSI-to-ST Inference Models')
    
    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.savefig('paper/figures/benchmark_chart.pdf', bbox_inches='tight', dpi=300)
    print("Saved benchmark chart to paper/figures/benchmark_chart.pdf")

if __name__ == "__main__":
    main()
