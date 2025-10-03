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

def plot_cluster_category_histograms(df, cat_col="categories", max_clusters=20, top_n=10):
    """
    Show a grid of histograms of categories in each cluster.
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

        cluster_label = cluster_id if cluster_id != -1 else "Noise"
        ax.set_title(f"Cluster {cluster_label} (n={len(cluster_df)})")
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
    plot_cluster_category_histograms(df, cat_col="categories", max_clusters=20, top_n=10)

def dbscan_with_pca(X, df=None, eps=0.5, min_samples=5, pca_dim=10):
    cat_col = "categories"

    pca = PCA(n_components=pca_dim)
    X_reduced = pca.fit_transform(X)
    print(f"Reduced embeddings shape: {X_reduced.shape}")

    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(X_reduced)
    distances, _ = neighbors.kneighbors(X_reduced)
    k_distances = np.sort(distances[:, -1])

    plt.figure(figsize=(8,5))
    plt.plot(k_distances)
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{min_samples}-NN distance")
    plt.title("k-distance plot for DBSCAN eps selection")
    plt.show()

    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(X_reduced)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"DBSCAN results: {n_clusters} clusters, {n_noise} noise points")

    if df is not None:
        df["cluster"] = labels
        preview_clusters(df, cat_col=cat_col, n_top=3)
        plot_cluster_category_histograms(df, cat_col="categories", max_clusters=20, top_n=10)

    mask = labels != -1
    if np.sum(mask) > 1 and n_clusters > 1:
        dbi_score = DBI(X_reduced[mask], labels[mask])
        print(f"Davies-Bouldin Index (excluding noise): {dbi_score:.4f}")
    else:
        print("Not enough clusters or points to compute DBI.")

if __name__ == "__main__":
    df, X = load_embeddings_from_csv("arxiv_with_embeddings_specter_2.csv")
    kMeans(X, df, k=20)
    # dbscan_with_pca(X, df=df, eps=3, min_samples=10, pca_dim=10)
