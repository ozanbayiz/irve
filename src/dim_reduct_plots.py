import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Path to the output file from your SAE pipeline
output_h5 = "path/to/output.hdf5"

# Load latent activations and labels
with h5py.File(output_h5, "r") as f:
    # Choose which split to visualize
    split = "training"  # or "validation"
    
    X = f[split]['latent_activations'][:]
    
    # Assuming race labels are stored as integer class labels (0–5 or similar)
    y = f[split]['labels']['race'][:]  # <-- update key if different

# Optional: downsample for t-SNE speed
max_samples = 5000
if X.shape[0] > max_samples:
    idx = np.random.choice(X.shape[0], max_samples, replace=False)
    X = X[idx]
    y = y[idx]

# PCA
X_pca = PCA(n_components=2).fit_transform(X)

# t-SNE
X_tsne = TSNE(n_components=2, init='pca', random_state=42).fit_transform(X)

# Plotting function
def plot_2d(data, labels, title):
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.colorbar(scatter, ticks=np.unique(labels))
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.show()

# Visualizations
plot_2d(X_pca, y, "PCA of Latent SAE Representations (Colored by Race)")
plot_2d(X_tsne, y, "t-SNE of Latent SAE Representations (Colored by Race)")
