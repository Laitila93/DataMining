import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (
    davies_bouldin_score, 
    silhouette_score, 
    normalized_mutual_info_score
    )
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

def load_embeddings_from_csv(filename="df_with_embeddings.csv"):
    df = pd.read_csv(filename)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = df[emb_cols].values
    print(f"Loaded {X.shape[0]} embeddings of size {X.shape[1]}")
    return df, X

def hierarchical_clustering(X, df=None, n_clusters=None, linkage_method="average", cat_col=None):
    X_reduced = PCA(n_components=50, random_state=42).fit_transform(X)

    # Select n number of clusters based on best silhouette score
    if n_clusters is None:
        print(" Searching for optimal number of clusters")
        scores = { k: silhouette_score(X_reduced, 
                                       AgglomerativeClustering(
                                           n_clusters=k, 
                                           metric="cosine", 
                                           linkage=linkage_method
                                        ).fit_predict(X_reduced)) for k in range(5, 31, 5)
        }
        n_clusters, best_sil = max(scores, key=scores.get), max(scores.values())
        print(f"Best k={n_clusters} (Silhouette={best_sil:.3f})")
    labels = AgglomerativeClustering(n_clusters=n_clusters, 
                                     metric="cosine", 
                                     linkage=linkage_method
                                     ).fit_predict(X_reduced)


    df["cluster"] = labels
    # Validation scores (silhouette and db)
    sil = silhouette_score(X_reduced, labels)
    dbi = davies_bouldin_score(X_reduced, labels)
    print(f" Silhouette Score: {sil:.3f}")
    print(f" Davies–Bouldin Index: {dbi:.3f}")
      

    if df is not None and cat_col:
        purity, nmi = evaluate_clusters(df, cat_col)
        preview_clusters(df, cat_col)
        print(f"Average purity={purity:.3f}, NMI={nmi:.3f}")

    return labels


def evaluate_clusters(df, cat_col="categories", cluster_col="cluster"):
    # Compute average cluster purity
    purities = [        
        group[cat_col].value_counts().max() / len(group)
        for _, group in df.groupby(cluster_col)
    ]
    avg_purity = np.mean(purities)
    nmi = normalized_mutual_info_score(df[cat_col], df[cluster_col])
    return avg_purity, nmi


#top categories for cluster
def preview_clusters(df, cat_col="categories", n_top=3):
    for cid, group in df.groupby("cluster"):
        top = group[cat_col].value_counts().head(n_top)
        print(f"\nCluster {cid} (n={len(group)}):")
        print(top.to_string())


# Plot the dendrogram with sample size = 100 for vizualisation
def plot_dendrogram(df, X, sample_size=100, method="average", label_col="categories"):

    n = min(sample_size, len(X))
    sample_idx = np.random.choice(len(X), size=n, replace=False)
    X_sample = X[sample_idx]

    labels = df.iloc[sample_idx][label_col].astype(str).values

    Z = linkage(X_sample, method=method)

    plt.figure(figsize=(16, 6))
    dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=10,
        truncate_mode=None
    )
    plt.title(f"Hierarchical Clustering Dendrogram ({method})")
    plt.xlabel(label_col)
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

# Heatmap to evaluate category distribution in clusters
def cluster_category_heatmap(df, cat_col="categories", cluster_col="cluster"):

    overlap = pd.crosstab(df[cluster_col], df[cat_col], normalize='index')

    plt.figure(figsize=(12, 6))
    sns.heatmap(overlap, cmap='viridis')
    plt.title("Cluster × Category Fraction Heatmap")
    plt.xlabel("Category")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    df, X = load_embeddings_from_csv("arxiv_with_embeddings_specter_2.csv")

    df_sample = df.sample(n=5000, random_state=42)
    X_sample = df_sample.filter(like="emb_").values

    hierarchical_clustering(
        X_sample,
        df_sample,
        cat_col="categories"
    )

    plot_dendrogram(df, X, sample_size=100, label_col="categories")

    cluster_category_heatmap(df_sample)
