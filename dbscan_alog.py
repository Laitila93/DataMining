from sklearn.metrics import davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'",
    category=FutureWarning
)


def load_embeddings_from_csv(filename="df_with_embeddings.csv"):
    """Load dataframe and embeddings matrix from CSV."""
    df = pd.read_csv(filename, dtype={"id": str}, low_memory=False)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = df[emb_cols].values
    print(f"Loaded {X.shape[0]} embeddings of size {X.shape[1]}")
    return df, X


def preprocess_embeddings_umap(df, X_raw, n_components=50):
    """Deduplicate, reduce embeddings with UMAP (cosine), and normalize for clustering."""
    _, idx = np.unique(X_raw, axis=0, return_index=True)
    df = df.iloc[idx].reset_index(drop=True)

    reducer = umap.UMAP(n_components=n_components, metric="cosine", random_state=42)
    X_reduced = reducer.fit_transform(X_raw[idx])

    X_norm = normalize(X_reduced)
    return df, X_norm


def cluster_embeddings_dbscan(X_reduced_normalized, eps=0.5, min_samples=50):
    """Cluster embeddings using DBSCAN (cosine via normalization + Euclidean)."""
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = clusterer.fit_predict(X_reduced_normalized)
    return labels


def visualize_clusters(X_reduced, labels, out_file="clusters.png"):
    """Project embeddings with UMAP (2D) for visualization and save cluster plot."""
    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    X_2d = reducer.fit_transform(X_reduced)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels, cmap="tab20", s=10, alpha=0.7
    )
    plt.colorbar(scatter, label="Cluster (-1 = noise)")
    plt.title("arXiv Abstracts (DBSCAN + UMAP)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_clusters(X_reduced, labels, categories):
    """Compute DBI and ARI (skip noise points for DBI/ARI)."""
    mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    dbi = None
    if mask.sum() > 1 and n_clusters > 1:
        dbi = davies_bouldin_score(X_reduced[mask], labels[mask])

    ari = None
    if n_clusters > 1:
        ari = adjusted_rand_score(categories[mask], labels[mask])

    return n_clusters, dbi, ari


def visualize_by_category(X_reduced, df, label_col="categories", out_file="categories.png"):
    """Project embeddings with UMAP and save category plot with direct colored labels."""
    reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
    X_2d = reducer.fit_transform(X_reduced)

    codes, uniques = pd.factorize(df[label_col])
    cmap = plt.get_cmap("tab20")

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=codes, cmap="tab20", s=10, alpha=0.7
    )

    # Annotate centroids
    for i, cat in enumerate(uniques):
        mask = codes == i
        if mask.sum() == 0:
            continue
        cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
        color = cmap(i % 20)
        plt.annotate(
            cat,
            (cx, cy),
            fontsize=12,
            ha="center",
            va="center",
            color=color,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor=color, boxstyle="round,pad=0.4"),
            arrowprops=dict(arrowstyle="-", lw=0.6, color=color)
        )

    plt.title(f"arXiv Abstracts Visualized by {label_col}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def run_clustering_pipeline(df, X_reduced, eps=0.5, min_samples=50):
    """Run DBSCAN clustering and evaluation."""
    labels = cluster_embeddings_dbscan(X_reduced, eps=eps, min_samples=min_samples)
    df["cluster"] = labels

    noise_pct = (labels == -1).sum() / len(labels) * 100
    print(f"Noise points: {noise_pct:.2f}% of all papers")

    os.makedirs("cluster_plots", exist_ok=True)
    os.makedirs("category_plots", exist_ok=True)

    visualize_clusters(X_reduced, labels,
                       out_file=f"cluster_plots/dbscan_umap50_eps{eps}_min{min_samples}.png")
    visualize_by_category(X_reduced, df, label_col="categories",
                          out_file=f"category_plots/dbscan_categories_umap50_eps{eps}_min{min_samples}.png")

    n_clusters, dbi, ari = evaluate_clusters(X_reduced, labels, df["categories"])
    print(f"\nUMAP=50, eps={eps}, min_samples={min_samples}")
    print(f"Clusters found = {n_clusters}")
    print(f"Davies-Bouldin Index = {dbi}")
    print(f"Adjusted Rand Index (vs categories) = {ari}")


def main():
    df, X = load_embeddings_from_csv("arxiv_with_embeddings_specter_2.csv")

    # Optional: subsample for testing
    df = df.sample(20000, random_state=42)
    X = X[df.index]

    df, X_reduced = preprocess_embeddings_umap(df, X, n_components=50)

    # Reuse same visualization UMAP
    umap_2d = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    X_2d = umap_2d.fit_transform(X_reduced)
        
    find_best_eps(X_reduced, min_samples=50,
              eps_values=np.linspace(0.003, 0.006, 8),
              out_file="dbscan_diagnostics.png")

    """
    eps_list = np.linspace(0.0033, 0.0058, 4)
    
    for eps in eps_list:
        for min_samples in [25, 50, 75]:
            print(f"\n=== Running DBSCAN with eps={eps}, min_samples={min_samples} ===")
            run_clustering_pipeline(df, X_reduced, eps=eps, min_samples=min_samples)
    """    

def find_best_eps(X, min_samples=50, n_neighbors=50, eps_values=None, out_file="dbscan_diagnostics.png"):
    """
    Plot sorted k-distances and overlay DBI/SSE for visual eps selection.
    No automatic best-eps detection — user picks by eye.
    """
    if eps_values is None:
        eps_values = np.linspace(0.001, 0.01, 10)

    # --- Step 1: compute sorted k-distances ---
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    # --- Step 2: evaluate DBI and SSE for each eps ---
    dbi_scores, sse_scores = [], []
    for eps in eps_values:
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        labels = clusterer.fit_predict(X)
        mask = labels != -1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters > 1 and mask.sum() > 1:
            dbi = davies_bouldin_score(X[mask], labels[mask])
            sse = sum(np.linalg.norm(X[mask][labels[mask] == k] -
                    X[mask][labels[mask] == k].mean(axis=0), axis=1).sum()
                    for k in set(labels[mask]))
        else:
            dbi, sse = np.nan, np.nan

        dbi_scores.append(dbi)
        sse_scores.append(sse)

    # --- Step 3: plot ---
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(k_distances, color="steelblue", lw=1.2)
    ax1.set_xlabel("Points (sorted by distance)")
    ax1.set_ylabel(f"{n_neighbors}th-nearest distance", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    # Mark eps values
    for eps in eps_values:
        ax1.axhline(y=eps, color="gray", linestyle="--", alpha=0.4)

    # Overlay DBI + SSE
    ax2 = ax1.twinx()
    ax2.plot(eps_values, dbi_scores, "o-", color="orange", label="DBI (↓ better)")
    ax2.plot(eps_values, sse_scores, "o--", color="red", label="SSE (↓ tighter)")
    ax2.set_ylabel("DBI / SSE", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.legend(loc="upper right")

    plt.title(f"DBSCAN Diagnostics (min_samples={min_samples})")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved DBSCAN diagnostic plot → {out_file}")


if __name__ == "__main__":
    main()