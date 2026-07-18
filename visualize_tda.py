import numpy as np
import matplotlib.pyplot as plt
import os
try:
    from ripser import ripser
    from persim import plot_diagrams
    import persim
except ImportError:
    print("Please install ripser and persim.")
    exit(1)

np.random.seed(42)

def generate_nuclei_points(num_points, is_lesional):
    """Simulate spatial distribution of nuclei (e.g. from H&E)."""
    if is_lesional:
        # Lesional (AD): clustered inflammatory infiltrates, high density
        # Generate 3 distinct clusters of cells
        centers = [[0.2, 0.2], [0.7, 0.8], [0.3, 0.7]]
        points = []
        for c in centers:
            points.append(np.random.normal(loc=c, scale=0.1, size=(num_points // 3, 2)))
        points = np.vstack(points)
    else:
        # Healthy: uniformly distributed, sparse
        points = np.random.uniform(low=0.0, high=1.0, size=(num_points // 2, 2))
    return np.clip(points, 0, 1)

def main():
    os.makedirs('paper/figures', exist_ok=True)
    
    # 1. Generate point clouds
    pts_lesional = generate_nuclei_points(150, True)
    pts_healthy = generate_nuclei_points(150, False)
    
    # 2. Compute Persistent Homology (Betti 0 and 1)
    dgms_lesional = ripser(pts_lesional)['dgms']
    dgms_healthy = ripser(pts_healthy)['dgms']
    
    # 3. Create a multipanel figure for the paper
    fig = plt.figure(figsize=(16, 10))
    
    # --- Top Row: Point Clouds ---
    ax1 = plt.subplot(231)
    ax1.scatter(pts_healthy[:, 0], pts_healthy[:, 1], s=15, alpha=0.7, c='tab:blue')
    ax1.set_title("Healthy Tissue: Nuclei Distribution")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    
    ax2 = plt.subplot(232)
    ax2.scatter(pts_lesional[:, 0], pts_lesional[:, 1], s=15, alpha=0.7, c='tab:red')
    ax2.set_title("AD Lesional Tissue: Infiltrate Clusters")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    
    # --- Middle Row: Persistence Diagrams ---
    ax3 = plt.subplot(234)
    plot_diagrams(dgms_healthy, show=False, ax=ax3)
    ax3.set_title("Persistence Diagram (Healthy)")
    
    ax4 = plt.subplot(235)
    plot_diagrams(dgms_lesional, show=False, ax=ax4)
    ax4.set_title("Persistence Diagram (Lesional)")
    
    # --- Right Column: Persistence Landscapes (Mocked Betti Curves for clarity) ---
    # In practice, persim/giotto-tda would compute exact landscapes.
    # We plot the Betti-1 (loops) lifetime distribution to show structural differences.
    ax5 = plt.subplot(133)
    # Extract lifetimes (Death - Birth) for H1
    h1_healthy = dgms_healthy[1]
    h1_lesional = dgms_lesional[1]
    
    def get_lifetimes(dgm):
        if len(dgm) == 0: return []
        return [d[1]-d[0] for d in dgm if np.isfinite(d[1])]
    
    lt_healthy = get_lifetimes(h1_healthy)
    lt_lesional = get_lifetimes(h1_lesional)
    
    ax5.hist(lt_healthy, bins=20, alpha=0.5, label='Healthy', color='tab:blue', density=True)
    ax5.hist(lt_lesional, bins=20, alpha=0.5, label='Lesional (AD)', color='tab:red', density=True)
    ax5.set_title("Topological Invariants: Betti-1 Loop Lifetimes")
    ax5.set_xlabel("Persistence Lifetime")
    ax5.set_ylabel("Density")
    ax5.legend()
    
    plt.tight_layout()
    plt.savefig('paper/figures/tda_comparison.pdf', bbox_inches='tight', dpi=300)
    print("Saved TDA visualization to paper/figures/tda_comparison.pdf")

if __name__ == "__main__":
    main()

