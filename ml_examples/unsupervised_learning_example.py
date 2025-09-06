"""
Unsupervised Learning Example: Clustering and Dimensionality Reduction

This example demonstrates:
1. K-Means Clustering
2. Hierarchical Clustering
3. Principal Component Analysis (PCA)
4. t-SNE for visualization

We'll use the Iris dataset and create synthetic data for demonstration.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import seaborn as sns

def clustering_example():
    """Example of unsupervised learning for clustering"""
    print("=" * 50)
    print("UNSUPERVISED LEARNING: CLUSTERING")
    print("=" * 50)
    
    # Load Iris dataset
    iris = load_iris()
    X, y_true = iris.data, iris.target
    feature_names = iris.feature_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"True number of clusters: {len(np.unique(y_true))}")
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means Clustering
    print("\n--- K-Means Clustering ---")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # Evaluate K-Means
    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
    kmeans_ari = adjusted_rand_score(y_true, kmeans_labels)
    
    print(f"K-Means Silhouette Score: {kmeans_silhouette:.4f}")
    print(f"K-Means Adjusted Rand Index: {kmeans_ari:.4f}")
    
    # Hierarchical Clustering
    print("\n--- Hierarchical Clustering ---")
    hierarchical = AgglomerativeClustering(n_clusters=3)
    hierarchical_labels = hierarchical.fit_predict(X_scaled)
    
    # Evaluate Hierarchical
    hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels)
    hierarchical_ari = adjusted_rand_score(y_true, hierarchical_labels)
    
    print(f"Hierarchical Silhouette Score: {hierarchical_silhouette:.4f}")
    print(f"Hierarchical Adjusted Rand Index: {hierarchical_ari:.4f}")
    
    # Visualize clustering results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original data (first two features)
    scatter1 = axes[0, 0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    axes[0, 0].set_title('True Labels')
    axes[0, 0].set_xlabel(feature_names[0])
    axes[0, 0].set_ylabel(feature_names[1])
    plt.colorbar(scatter1, ax=axes[0, 0])
    
    # K-Means results
    scatter2 = axes[0, 1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
    axes[0, 1].set_title(f'K-Means Clustering (Silhouette: {kmeans_silhouette:.3f})')
    axes[0, 1].set_xlabel(feature_names[0])
    axes[0, 1].set_ylabel(feature_names[1])
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    # Hierarchical results
    scatter3 = axes[1, 0].scatter(X[:, 0], X[:, 1], c=hierarchical_labels, cmap='viridis', alpha=0.7)
    axes[1, 0].set_title(f'Hierarchical Clustering (Silhouette: {hierarchical_silhouette:.3f})')
    axes[1, 0].set_xlabel(feature_names[0])
    axes[1, 0].set_ylabel(feature_names[1])
    plt.colorbar(scatter3, ax=axes[1, 0])
    
    # Cluster centers for K-Means
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    axes[1, 1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
    axes[1, 1].scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
    axes[1, 1].set_title('K-Means with Cluster Centers')
    axes[1, 1].set_xlabel(feature_names[0])
    axes[1, 1].set_ylabel(feature_names[1])
    
    plt.tight_layout()
    plt.savefig('/Users/vignesh/Documents/GitHub/Generative AI/ml_examples/unsupervised_clustering_results.png')
    plt.show()
    
    return kmeans, hierarchical

def dimensionality_reduction_example():
    """Example of unsupervised learning for dimensionality reduction"""
    print("\n" + "=" * 50)
    print("UNSUPERVISED LEARNING: DIMENSIONALITY REDUCTION")
    print("=" * 50)
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    
    print(f"Original data shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Principal Component Analysis (PCA)
    print("\n--- Principal Component Analysis (PCA) ---")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"PCA components shape: {X_pca.shape}")
    
    # t-SNE
    print("\n--- t-SNE ---")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)
    print(f"t-SNE components shape: {X_tsne.shape}")
    
    # Visualize dimensionality reduction results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original data (first two features)
    scatter1 = axes[0, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    axes[0, 0].set_title('Original Data (First 2 Features)')
    axes[0, 0].set_xlabel(feature_names[0])
    axes[0, 0].set_ylabel(feature_names[1])
    plt.colorbar(scatter1, ax=axes[0, 0])
    
    # PCA results
    scatter2 = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    axes[0, 1].set_title(f'PCA (Explained Variance: {pca.explained_variance_ratio_.sum():.3f})')
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    # t-SNE results
    scatter3 = axes[1, 0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
    axes[1, 0].set_title('t-SNE')
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter3, ax=axes[1, 0])
    
    # Feature importance in PCA
    feature_importance = np.abs(pca.components_).sum(axis=0)
    axes[1, 1].bar(feature_names, feature_importance)
    axes[1, 1].set_title('Feature Importance in PCA')
    axes[1, 1].set_ylabel('Importance')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/Users/vignesh/Documents/GitHub/Generative AI/ml_examples/unsupervised_dimensionality_reduction_results.png')
    plt.show()
    
    return pca, tsne

def elbow_method_example():
    """Demonstrate the elbow method for choosing optimal number of clusters"""
    print("\n" + "=" * 50)
    print("ELBOW METHOD FOR OPTIMAL CLUSTERS")
    print("=" * 50)
    
    # Create synthetic data with known clusters
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=1.5, random_state=42)
    
    # Test different numbers of clusters
    k_range = range(1, 11)
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        
        if k > 1:  # Silhouette score requires at least 2 clusters
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        else:
            silhouette_scores.append(0)
    
    # Visualize elbow method
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow curve
    axes[0].plot(k_range, inertias, 'bo-')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')
    axes[0].grid(True)
    
    # Silhouette scores
    axes[1].plot(k_range, silhouette_scores, 'ro-')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score vs Number of Clusters')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/Users/vignesh/Documents/GitHub/Generative AI/ml_examples/elbow_method_results.png')
    plt.show()
    
    # Find optimal k
    optimal_k = k_range[np.argmax(silhouette_scores[1:]) + 1]  # Skip k=1
    print(f"Optimal number of clusters (based on silhouette score): {optimal_k}")
    
    return optimal_k

def main():
    """Main function to run all unsupervised learning examples"""
    print("UNSUPERVISED LEARNING EXAMPLES")
    print("This example demonstrates clustering and dimensionality reduction")
    print("using the Iris dataset and synthetic data.\n")
    
    # Run clustering example
    kmeans, hierarchical = clustering_example()
    
    # Run dimensionality reduction example
    pca, tsne = dimensionality_reduction_example()
    
    # Run elbow method example
    optimal_k = elbow_method_example()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("Unsupervised learning finds hidden patterns in data without")
    print("labeled examples. Key techniques demonstrated:")
    print("1. Clustering: Groups similar data points together")
    print("2. Dimensionality Reduction: Reduces feature space while")
    print("   preserving important information")
    print("3. Elbow Method: Helps determine optimal number of clusters")
    print(f"\nOptimal clusters for synthetic data: {optimal_k}")

if __name__ == "__main__":
    main()
