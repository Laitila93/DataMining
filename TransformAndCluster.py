import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def loadData(data):
    df = pd.read_csv(data, sep=",", on_bad_lines="skip")
    print("Loaded data from csv: ", df.head())
    return df

def transformData(df, column):
    texts = df[column].astype(str).tolist()
    model = SentenceTransformer("allenai-specter")  # small but powerful #all-MiniLM-L6-v2
    X = model.encode(texts, show_progress_bar=True)
    print("Shape of vectorized data: ", X.shape)
    return X

def save_embeddings_with_df(df, X, filename="df_with_embeddings.csv"):
    # Convert X (numpy array) into a DataFrame with numbered columns
    emb_df = pd.DataFrame(X, columns=[f"emb_{i}" for i in range(X.shape[1])])

    # Concatenate with original df (row-wise alignment is preserved)
    merged = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    # Save to CSV
    merged.to_csv(filename, index=False)
    print(f"Saved embeddings + data to {filename}")

if __name__ == "__main__":
    df = loadData("arxiv-stratified-sample-3.csv")
    X = transformData(df, "abstract")
    # Save as csv
    save_embeddings_with_df(df, X, "arxiv_with_embeddings_specter_3.csv")
