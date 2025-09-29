import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score as DBI
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def load_embeddings_from_csv(filename="df_with_embeddings.csv"):
    df = pd.read_csv(filename)
    # Extract embeddings (all columns starting with "emb_")
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = df[emb_cols].values
    print(f"Loaded {X.shape[0]} embeddings of size {X.shape[1]}")
    return df, X

def preview_clusters(df, cat_col="categories", n_top=3):
    """
    Print cluster size and the top N categories with their percentage for each cluster,
    including noise if present.
    """
    for cluster_id, group in df.groupby("cluster"):
        cluster_label = cluster_id if cluster_id != -1 else "Noise"
        cluster_size = len(group)

        # Compute category counts and percentages
        cat_counts = group[cat_col].value_counts()
        top_cats = cat_counts.head(n_top)
        top_cats_pct = top_cats / cluster_size * 100

        # Print cluster info
        print(f"\n--- Cluster {cluster_label} (size: {cluster_size}) ---")
        for cat, pct in zip(top_cats.index, top_cats_pct):
            print(f"{cat}: {pct:.1f}%")

def kMeans(X, df, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = kmeans.labels_
    df["cluster"] = labels

    preview_clusters(df, cat_col="categories")
    print(f"K = {k}, DB-index = {DBI(X, kmeans.labels_)}")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="tab10")
    plt.title("Text Clusters (PCA projection)")
    plt.show()
    return
def dbscan_with_pca(X, df=None, eps=0.5, min_samples=5, pca_dim=10):
    text_col = "title"
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

    mask = labels != -1
    if np.sum(mask) > 1 and n_clusters > 1:
        dbi_score = DBI(X_reduced[mask], labels[mask])
        print(f"Davies-Bouldin Index (excluding noise): {dbi_score:.4f}")
    else:
        print("Not enough clusters or points to compute DBI.")

    return

if __name__ == "__main__":
    df, X = load_embeddings_from_csv("arxiv_with_embeddings_specter_2.csv")
    kMeans(X, df, k=20)
    #dbscan_with_pca(X, df=df, eps=3, min_samples=10, pca_dim=10)

