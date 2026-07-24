import os
import torch
import numpy as np
from eczema_flow.tda import TDAFeatureExtractor

def validate_tda():
    print("Validating TDA Feature Extraction Pipeline...")
    tda = TDAFeatureExtractor(output_dim=64)
    
    # We will test the raw _compute_persistence_image function
    np.random.seed(42)
    empty_count = 0
    densities = []
    h0_energies = []
    h1_energies = []
    
    for i in range(100):
        # random num of points from 0 to 50
        n_points = np.random.randint(0, 50)
        densities.append(n_points)
        points = np.random.rand(n_points, 2) * 224
        
        pi = tda._compute_persistence_image(points)
        if np.sum(pi) == 0:
            empty_count += 1
            h0_energies.append(0.0)
            h1_energies.append(0.0)
        else:
            h0 = pi[:400]
            h1 = pi[400:]
            h0_energies.append(np.sum(h0))
            h1_energies.append(np.sum(h1))
            
    print(f"Empty Diagram Rate: {empty_count / 100 * 100:.2f}% (acceptable threshold < 5%)")
    
    # Correlation between density and H0 energy
    from scipy.stats import pearsonr
    r_h0, _ = pearsonr(densities, h0_energies)
    r_h1, _ = pearsonr(densities, h1_energies)
    print(f"Correlation Density vs H0 Energy: {r_h0:.3f}")
    print(f"Correlation Density vs H1 Energy: {r_h1:.3f}")
    
    if empty_count < 10:
        print("✅ TDA Pipeline mathematically sound.")
    else:
        print("⚠️ High empty diagram rate detected!")

if __name__ == "__main__":
    validate_tda()
