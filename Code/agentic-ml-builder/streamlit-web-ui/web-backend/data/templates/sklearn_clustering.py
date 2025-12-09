
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

def cluster_data(data_path, n_clusters=3, method='kmeans'):
    # Load data
    try:
        df = pd.read_csv(data_path)
        # Drop non-numeric columns for clustering
        df_numeric = df.select_dtypes(include=['float64', 'int64']).dropna()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    # Clustering
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=5)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)

    labels = model.fit_predict(X_scaled)
    df['cluster'] = labels

    print(f"Clustering complete using {method}")
    print(df['cluster'].value_counts())

    # Visualization (PCA to 2D)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.title(f'Clustering Results ({method})')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.savefig('cluster_plot.png')
    print("Plot saved to cluster_plot.png")

    return df

if __name__ == "__main__":
    # cluster_data('data.csv')
    pass
