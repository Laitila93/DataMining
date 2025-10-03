import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score as DBI
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

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


def hierarchical_clustering(X, df, n_clusters=10, linkage="ward", pca_dim=50):

    # Reduce embeddings first
    if X.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim)
        X = pca.fit_transform(X)
        print(f"Reduced dimensions to {X.shape[1]} with PCA")
    
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = hc.fit_predict(X)
    df["cluster"] = labels

    # Preview cluster info
    preview_clusters(df, cat_col="categories")

    # Evaluate clustering with Davies-Bouldin Index
    if n_clusters > 1:
        dbi_score = DBI(X, labels)
        print(f"Hierarchical Clustering (linkage={linkage})")
        print(f"Clusters: {n_clusters}, DBI = {dbi_score:.4f}")
    else:
        print("Not enough clusters to compute DBI.")

    # Reduce to 2D with PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="tab10")
    plt.title(f"Hierarchical Clustering (PCA projection, linkage={linkage})")
    plt.show()

def plot_dendrogram(X, sample_size=200):

    # Reduce dataset size for dendrogram
    if X.shape[0] > sample_size:
        idx = np.random.choice(X.shape[0], sample_size, replace=False)
        X = X[idx]
        print(f"Using {sample_size} samples for dendrogram")

    # Reduce dimensionality with PCA for speed
    if X.shape[1] > 50:
        pca = PCA(n_components=50)
        X = pca.fit_transform(X)
        print(f"Reduced to {X.shape[1]} dimensions with PCA")

    # Compute the linkage matrix
    Z = linkage(X, method="ward")  

    # Plot the dendrogram
    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode="level", p=5)
    plt.title("Hierarchical Clustering Dendrogram (truncated)")
    plt.xlabel("Sample index or (cluster size)")
    plt.ylabel("Distance")
    plt.show()

if __name__ == "__main__":
    df, X = load_embeddings_from_csv("arxiv_with_embeddings_specter_2.csv")
    
    # Sample before dendrogram 
    plot_dendrogram(X, sample_size=300)  

    # Hierarchical clustering on sample
    df_sample = df.sample(n=5000, random_state=42)
    X_sample = df_sample[[c for c in df.columns if c.startswith("emb_")]].values
    hierarchical_clustering(X_sample, df_sample, n_clusters=15, linkage="ward")

