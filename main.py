from src.data_preprocessing import load_and_preprocess
from src.cluster_analysis import find_optimal_clusters
from src.segmentation import perform_clustering
from src.pca_visualization import visualize_clusters

def main():

    file_path = "data/Mall_Customers.csv"

    df, X = load_and_preprocess(file_path)

    print("Dataset Loaded Successfully")
    print(df.head())

    # Elbow Method
    find_optimal_clusters(X)

    # Clustering
    df, model = perform_clustering(df, X)

    print("\nCluster Counts:")
    print(df['Cluster'].value_counts())

    # Visualization
    visualize_clusters(df)

if __name__ == "__main__":
    main()