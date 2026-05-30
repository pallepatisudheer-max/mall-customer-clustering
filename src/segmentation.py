from sklearn.cluster import KMeans

def perform_clustering(df, X):
    kmeans = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X)

    df['Cluster'] = clusters

    return df, kmeans