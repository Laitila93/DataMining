import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score as DBI
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def load_embeddings_from_csv(filename="df_with_embeddings.csv"):
    df = pd.read_csv(filename)
    # Extract embeddings (all columns starting with "emb_")
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = df[emb_cols].values
    print(f"Loaded {X.shape[0]} embeddings of size {X.shape[1]}")
    return df, X

def preview_clusters(df, cat_col="categories", n_top=3):
    for cluster_id, group in df.groupby("cluster"):
        cluster_label = cluster_id if cluster_id != -1 else "Noise"
        cluster_size = len(group)

        cat_counts = group[cat_col].value_counts()
        top_cats = cat_counts.head(n_top)

        print(f"\n--- Cluster {cluster_label} (size: {cluster_size}) ---")
        for cat, count in zip(top_cats.index, top_cats.values):
            print(f"{cat}: {count}")

def plot_cluster_category_histograms(df, X=None, cat_col="categories", max_clusters=20, top_n=10):
    """
    Show a grid of histograms of categories in each cluster, with cohesion metrics.
    Cohesion is defined as average squared distance to the cluster centroid.
    """
    clusters = sorted(df["cluster"].unique())
    n_clusters = min(len(clusters), max_clusters)

    n_cols = 4
    n_rows = int(np.ceil(n_clusters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), constrained_layout=True)
    axes = axes.flatten()

    for i, cluster_id in enumerate(clusters[:n_clusters]):
        ax = axes[i]
        cluster_df = df[df["cluster"] == cluster_id]

        counts = cluster_df[cat_col].value_counts().head(top_n)
        counts.plot(kind="bar", ax=ax)

        # Cohesion metric (only if embeddings X provided)
        cohesion = None
        if X is not None and cluster_id != -1:
            cluster_idx = cluster_df.index
            cluster_points = X[cluster_idx]
            centroid = cluster_points.mean(axis=0)
            cohesion = np.mean(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)

        cluster_label = cluster_id if cluster_id != -1 else "Noise"
        title = f"Cluster {cluster_label} (n={len(cluster_df)})"
        if cohesion is not None:
            title += f"\nCohesion (Avg. SSE)={cohesion:.3f}"

        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.set_xlabel("Category")

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.show()


def kMeans(X, df, k):
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    df["cluster"] = labels

    preview_clusters(df, cat_col="categories")
    print(f"K = {k}, DB-index = {DBI(X, kmeans.labels_)}")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="tab10", s=5)
    plt.title("Text Clusters (PCA projection)")
    plt.show()

    # Show category histograms
    plot_cluster_category_histograms(df, X, cat_col="categories", max_clusters=20, top_n=10)

if __name__ == "__main__":
    df, X = load_embeddings_from_csv("arxiv_with_embeddings_specter_2.csv")
    kMeans(X, df, k=20)
    # dbscan_with_pca(X, df=df, eps=3, min_samples=10, pca_dim=10)
