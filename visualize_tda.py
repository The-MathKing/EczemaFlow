import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import scipy.ndimage as ndi
try:
    from ripser import ripser
    from persim import plot_diagrams
except ImportError:
    print("Please install ripser and persim.")
    exit(1)

import json

def main():
    with open('results/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    with_tda_pcc = metrics['EczemaFlow (Full)']['PCC']
    
    # We parsed the log previously, No TDA test PCC = 0.0407 (mean across folds was 0.0353)
    no_tda_pcc = 0.0407
    
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ['With Topological Extractor', 'Without Topological Extractor (Ablation)']
    values = [with_tda_pcc, no_tda_pcc]
    
    bars = ax.bar(labels, values, color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel('Mean Pearson Correlation (PCC)')
    ax.set_title('Topological Feature Extractor Ablation')
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f'{yval:.3f}', ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig('paper/figures/tda_comparison.pdf', bbox_inches='tight', dpi=300)
    print("Saved TDA ablation chart to paper/figures/tda_comparison.pdf")

if __name__ == "__main__":
    main()
