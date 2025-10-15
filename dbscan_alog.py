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

    visualize_clusters(X_reduced, labels, out_file=f"cluster_plots/dbscan_umap50_eps{eps}_min{min_samples}.png")

    n_clusters, dbi, ari = evaluate_clusters(X_reduced, labels, df["categories"])
    print(f"\nUMAP=50, eps={eps}, min_samples={min_samples}")
    print(f"Clusters found = {n_clusters}")
    print(f"Davies-Bouldin Index = {dbi}")
    print(f"Adjusted Rand Index (vs categories) = {ari}")

def main():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="umap")

    df, X = load_embeddings_from_csv("arxiv_with_embeddings_specter_2.csv")
    df = df.sample(20000, random_state=42)
    X = X[df.index]
    df, X_reduced = preprocess_embeddings_umap(df, X, n_components=50)

    eps_list = np.linspace(0.0082, 0.0120, 6)
    min_samples_list = [40, 60, 80, 100]
    
    parameter_elbow_plots = True
    if parameter_elbow_plots:
        print("\n=== Step 1: Finding good eps (elbow method) ===")
        find_best_eps(
            X_reduced,
            n_neighbors=50,
            eps_values=np.linspace(0.005, 0.012, 8),
            out_file="dbscan_kdistance.png"
        )

        print("\n=== Step 2: Analyzing neighbor density to pick min_samples ===")
        analyze_min_samples_domain(
            X_reduced,
            eps=0.009,  # use approximate eps from visual elbow
            max_neighbors=200,
            out_file="min_samples_diagnostics.png"
        )

    
    
    results = []
    print("\nRunning DBSCAN grid search...\n")

    for eps in eps_list:
        for min_samples in min_samples_list:
            labels = cluster_embeddings_dbscan(X_reduced, eps=eps, min_samples=min_samples)
            df["cluster"] = labels

            noise_pct = (labels == -1).sum() / len(labels) * 100
            n_clusters, dbi, ari, largest_cluster_ratio, cluster_entropy = evaluate_clusters(
                X_reduced, labels, df["categories"]
            )
            
            visualize_clusters(X_reduced, labels, out_file=f"cluster_plots/dbscan_umap50_eps{eps}_min{min_samples}.png")

            results.append({
                "eps": eps,
                "min_samples": min_samples,
                "clusters": n_clusters,
                "noise_percent": noise_pct,
                "DBI": dbi,
                "ARI": ari,
                "largest_ratio": largest_cluster_ratio,
                "entropy": cluster_entropy
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="ARI", ascending=False)

    print("\n" + "=" * 100)
    print(f"{'eps':<10}{'min_samples':<15}{'clusters':<10}{'noise_percent':<10}"
          f"{'DBI':<12}{'ARI':<10}{'largest_ratio':<18}{'entropy':<10}")
    print("=" * 100)
    results_df = results_df.rename(columns={'noise_percent': 'noise_percent'})
    for _, r in results_df.iterrows():
        print(f"{r['eps']:<10.5f}{int(r['min_samples']):<15}{int(r['clusters']):<10}"
          f"{r['noise_percent']:<10.2f}{r['DBI']:<12.4f}{r['ARI']:<10.4f}"
          f"{r['largest_ratio']:<18.3f}{r['entropy']:<10.3f}")


    print("=" * 100)
    best = results_df.iloc[0]
    print(f"\nBest config → eps={best.eps:.5f}, min_samples={int(best.min_samples)} "
          f"(DBI={best.DBI:.4f}, ARI={best.ARI:.4f}, largest_ratio={best.largest_ratio:.3f})")

def evaluate_clusters(X_reduced, labels, categories):
    """Compute DBI, ARI, and cluster distribution metrics."""
    mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    largest_cluster_ratio = np.nan
    cluster_entropy = np.nan

    if n_clusters > 0 and mask.sum() > 0:
        counts = np.bincount(labels[mask])
        largest_cluster_ratio = counts.max() / counts.sum()
        p = counts / counts.sum()
        cluster_entropy = -np.sum(p * np.log2(p))

    dbi = ari = np.nan
    if mask.sum() > 1 and n_clusters > 1:
        dbi = davies_bouldin_score(X_reduced[mask], labels[mask])
        ari = adjusted_rand_score(categories[mask], labels[mask])

    return n_clusters, dbi, ari, largest_cluster_ratio, cluster_entropy

def find_best_eps(X, n_neighbors=50, eps_values=None, out_file="dbscan_kdistance.png"):
    """
    Plot sorted k-distances for visual DBSCAN eps selection (classic elbow method).
    """
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt
    import numpy as np

    if eps_values is None:
        eps_values = np.linspace(0.001, 0.01, 10)

    # Step 1: compute sorted k-distances
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    # Step 2: plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances, color="steelblue", lw=1.8, label=f"{n_neighbors}th-nearest distance")
    
    for eps in eps_values:
        plt.axhline(y=eps, color="gray", linestyle="--", alpha=0.4, lw=1.0)

    plt.title(f"DBSCAN k-Distance Curve (n_neighbors={n_neighbors})", fontsize=13, pad=12)
    plt.xlabel("Points (sorted by distance)", fontsize=11)
    plt.ylabel(f"{n_neighbors}th-nearest distance", fontsize=11)
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="gray")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved k-distance plot → {out_file}")


def analyze_min_samples_domain(X, eps, max_neighbors=200, out_file="min_samples_diagnostics.png"):
    """
    Analyze neighbor density for a given eps to guide min_samples choice.
    Plots histogram of how many neighbors each point has within eps.
    """
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt
    import numpy as np

    nn = NearestNeighbors(n_neighbors=max_neighbors, metric="euclidean")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    neighbor_counts = (distances < eps).sum(axis=1)

    # Summary stats
    median_n = np.median(neighbor_counts)
    mean_n = np.mean(neighbor_counts)
    p90_n = np.percentile(neighbor_counts, 90)

    print(f"\nNeighbor count stats for eps={eps:.4f}:")
    print(f"  Mean     = {mean_n:.2f}")
    print(f"  Median   = {median_n:.2f}")
    print(f"  90th pct = {p90_n:.2f}")
    print(f"  Suggested min_samples domain: [{int(median_n/2)} … {int(p90_n)}]")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(neighbor_counts, bins=50, color="steelblue", alpha=0.75, edgecolor="black")
    plt.axvline(median_n, color="orange", linestyle="--", lw=1.8, label=f"Median = {median_n:.1f}")
    plt.axvline(mean_n, color="green", linestyle="--", lw=1.8, label=f"Mean = {mean_n:.1f}")
    plt.axvline(p90_n, color="red", linestyle="--", lw=1.8, label=f"90th percentile = {p90_n:.1f}")

    plt.title(f"Neighbor Count Distribution (eps={eps})", fontsize=13, pad=12)
    plt.xlabel("Number of neighbors within eps", fontsize=11)
    plt.ylabel("Points", fontsize=11)
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="gray")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved neighbor count histogram → {out_file}")


if __name__ == "__main__":
    main()