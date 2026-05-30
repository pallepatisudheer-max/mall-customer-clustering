import matplotlib.pyplot as plt
import seaborn as sns

def visualize_clusters(df):
    plt.figure(figsize=(10,6))

    sns.scatterplot(
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        hue='Cluster',
        palette='Set1',
        data=df,
        s=100
    )

    plt.title("Customer Segmentation")
    plt.show()