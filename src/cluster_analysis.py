import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_optimal_clusters(X):
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(
            n_clusters=i,
            random_state=42,
            n_init=10
        )
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8,5))
    plt.plot(range(1,11), wcss, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.show()