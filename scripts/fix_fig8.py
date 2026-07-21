import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('paper/figures', exist_ok=True)
labels = ['Hist2ST (Psoriasis)', 'EczemaFlow (Psoriasis)']
mse = [0.039, 0.012]
colors = ['#ff7f0e', '#1f77b4']

plt.figure(figsize=(6, 5))
plt.bar(labels, mse, color=colors)
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Out-of-Distribution Generalization on Psoriasis Cohort')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('paper/figures/ood_generalization.pdf', dpi=300)
print("Clean Figure 8 generated.")
