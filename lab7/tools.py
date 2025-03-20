import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def load_small_similarity_data():
    """Load a small similarity matrix for example animals"""
    # Define a simple similarity matrix for 5 animals
    sim_data = np.array([
        [1.0, 0.8, 0.3, 0.2, 0.3],  # Dog
        [0.8, 1.0, 0.3, 0.2, 0.2],  # Wolf
        [0.3, 0.3, 1.0, 0.1, 0.3],  # Eagle
        [0.2, 0.2, 0.1, 1.0, 0.7],  # Goldfish
        [0.3, 0.2, 0.3, 0.7, 1.0]   # Dolphin
    ])

    # sim_data = np.array([
    #     [np.nan, np.nan, np.nan, np.nan, np.nan],  # Dog
    #     [0.8, np.nan, np.nan, np.nan, np.nan],  # Wolf
    #     [0.3, 0.3, np.nan, np.nan, np.nan],  # Eagle
    #     [0.2, 0.2, 0.1, np.nan, np.nan],  # Goldfish
    #     [0.3, 0.2, 0.3, 0.7, np.nan]   # Dolphin
    # ])
    
    animal_names = ["Dog", "Wolf", "Eagle", "Goldfish", "Dolphin"]
    df_sim = pd.DataFrame(sim_data, index=animal_names, columns=animal_names)
    
    return df_sim

def display_simple_hierarchy():
    """Display a simple hierarchical structure for illustration"""
    # Create a simple hierarchy diagram
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Draw the nodes and connections
    ax.plot([1, 1, 2, 2], [1, 2, 1, 2], 'ko-')  # Dog and Wolf to Land Mammals
    ax.plot([3, 3], [1, 2], 'ko-')              # Eagle to Birds
    ax.plot([4, 4, 5, 5], [1, 2, 1, 2], 'ko-')  # Goldfish and Dolphin to Water Animals
    
    ax.plot([2, 2, 3, 3], [2, 3, 2, 3], 'ko-')  # Land Mammals and Birds to Land Animals
    ax.plot([2, 2, 5, 5], [3, 4, 2, 4], 'ko-')  # Land Animals and Water Animals to Animals
    
    # Label the nodes
    ax.text(1, 1, "Dog", ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    ax.text(2, 1, "Wolf", ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    ax.text(3, 1, "Eagle", ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    ax.text(4, 1, "Goldfish", ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    ax.text(5, 1, "Dolphin", ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    
    ax.text(1.5, 2, "Land Mammals", ha='center', va='center', bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round'))
    ax.text(3, 2, "Birds", ha='center', va='center', bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round'))
    ax.text(4.5, 2, "Water Animals", ha='center', va='center', bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round'))
    
    ax.text(2.5, 3, "Land Animals", ha='center', va='center', bbox=dict(facecolor='lightgreen', edgecolor='black', boxstyle='round'))
    
    ax.text(3.5, 4, "Animals", ha='center', va='center', bbox=dict(facecolor='lightyellow', edgecolor='black', boxstyle='round'))
    
    # Set limits and remove axes
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    plt.title('Simple Hierarchical Structure')
    plt.tight_layout()
    plt.show()

def compute_similarity(df_features, weights):
    """Compute similarity between objects based on weighted features
    
    Parameters:
    -----------
    df_features : DataFrame
        Feature matrix where rows are objects and columns are features
    weights : array-like
        Weights for each feature
        
    Returns:
    --------
    df_sim : DataFrame
        Similarity matrix
    """
    # Get object names
    objects = df_features.index
    
    # Initialize similarity matrix
    n_objects = len(objects)
    similarities = np.zeros((n_objects, n_objects))
    
    # Compute similarity for each pair of objects
    for i in range(n_objects):
        for j in range(n_objects):
            # Diagonal elements (self-similarity) are always 1
            if i == j:
                similarities[i, j] = 1.0
                continue
                
            # For each feature, check if both objects have it
            shared_features = df_features.iloc[i].values * df_features.iloc[j].values
            
            # Compute weighted sum of shared features
            similarity = np.sum(weights * shared_features)
            similarities[i, j] = similarity
    
    # Create DataFrame with proper indices
    df_sim = pd.DataFrame(similarities, index=objects, columns=objects)
    
    return df_sim

def compute_error(df_actual, df_predicted):
    """Compute mean squared error between actual and predicted similarities
    
    Parameters:
    -----------
    df_actual : DataFrame
        Actual similarity matrix
    df_predicted : DataFrame
        Predicted similarity matrix
        
    Returns:
    --------
    mse : float
        Mean squared error
    """
    # Get the lower triangle (excluding diagonal) for both matrices
    actual_values = []
    predicted_values = []
    
    for i in range(len(df_actual)):
        for j in range(i):  # Only lower triangle
            actual_values.append(df_actual.iloc[i, j])
            predicted_values.append(df_predicted.iloc[i, j])
    
    # Convert to numpy arrays
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)
    
    # Compute MSE
    mse = np.mean((actual_values - predicted_values) ** 2)
    
    return mse

def find_best_weights(df_features, df_sim):
    """Find weights that minimize error between predicted and actual similarities
    
    Parameters:
    -----------
    df_features : DataFrame
        Feature matrix
    df_sim : DataFrame
        Actual similarity matrix
        
    Returns:
    --------
    weights : array
        Optimal weights
    """
    def error_function(weights):
        # Compute predicted similarities
        df_pred = compute_similarity(df_features, weights)
        
        # Compute error
        return compute_error(df_sim, df_pred)
    
    # Initial weights (equal weights)
    n_features = df_features.shape[1]
    initial_weights = np.ones(n_features) / n_features
    
    # Constraints: weights must be non-negative
    bounds = [(0, None) for _ in range(n_features)]
    
    # Find weights that minimize error
    result = minimize(error_function, initial_weights, bounds=bounds)
    
    return result.x

def plot_simple_dendrogram(merge_history, all_objects):
    """Plot a simple dendrogram from merge history
    
    Parameters:
    -----------
    merge_history : list of tuples
        List of (cluster1, cluster2) pairs that were merged
    all_objects : list
        List of all object names
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Place objects on x-axis
    x_positions = {obj: i for i, obj in enumerate(all_objects)}
    
    # Plot objects as leaf nodes
    for i, obj in enumerate(all_objects):
        ax.plot([i, i], [0, 0.1], 'k-')
        ax.text(i, -0.05, obj, ha='center', va='top', rotation=90)
    
    # Track merged clusters
    merged_clusters = {tuple([obj]): (i, 0) for i, obj in enumerate(all_objects)}
    
    # Plot each merge
    y_offset = 0.2
    for step, (cluster1, cluster2) in enumerate(merge_history):
        # Convert clusters to tuples for dictionary keys
        key1 = tuple(sorted(cluster1))
        key2 = tuple(sorted(cluster2))
        
        # Get positions of existing clusters
        x1, y1 = merged_clusters[key1]
        x2, y2 = merged_clusters[key2]
        
        # Compute position for new cluster
        new_x = (x1 + x2) / 2
        new_y = max(y1, y2) + y_offset
        
        # Draw vertical lines up to new_y
        ax.plot([x1, x1], [y1, new_y], 'k-')
        ax.plot([x2, x2], [y2, new_y], 'k-')
        
        # Draw horizontal line connecting them
        ax.plot([x1, x2], [new_y, new_y], 'k-')
        
        # Track this new merged cluster
        merged_clusters[tuple(sorted(cluster1 + cluster2))] = (new_x, new_y)
    
    # Set limits and labels
    ax.set_xlim(-0.5, len(all_objects) - 0.5)
    ax.set_title('Hierarchical Clustering Dendrogram')
    ax.set_xlabel('Animals')
    ax.set_ylabel('Merge Height')
    ax.get_xaxis().set_ticks([])
    
    plt.tight_layout()
    plt.show()

def create_feature_matrix(clusters, all_objects):
    """Create a feature matrix from clusters
    
    Parameters:
    -----------
    clusters : list of lists
        List of clusters, where each cluster is a list of object names
    all_objects : list
        List of all object names
        
    Returns:
    --------
    df_features : pandas DataFrame
        Feature matrix where rows are objects and columns are features
    """
    # Find non-singleton clusters
    non_singleton_clusters = [cluster for cluster in clusters if len(cluster) > 1]
    
    # Create an empty feature matrix
    n_objects = len(all_objects)
    n_features = len(non_singleton_clusters)
    features = np.zeros((n_objects, n_features))
    
    # Fill in the feature matrix
    for j, cluster in enumerate(non_singleton_clusters):
        for i, obj in enumerate(all_objects):
            if obj in cluster:
                features[i, j] = 1
    
    # Create column names for the features
    feature_names = [f"Cluster_{j+1}" for j in range(n_features)]
    
    # Create the DataFrame
    df_features = pd.DataFrame(features, index=all_objects, columns=feature_names)
    
    return df_features

def merge_clusters(clusters, i, j):
    """Merge two clusters
    
    Parameters:
    -----------
    clusters : list of lists
        List of clusters
    i, j : int, int
        Indices of clusters to merge
        
    Returns:
    --------
    new_clusters : list of lists
        Updated list of clusters with i and j merged
    """
    # Create a copy of the clusters
    new_clusters = clusters.copy()
    
    # Create the merged cluster
    merged_cluster = new_clusters[i] + new_clusters[j]
    
    # Remove the original clusters (be careful with indices!)
    # Remove the higher index first to avoid shifting problems
    if i > j:
        new_clusters.pop(i)
        new_clusters.pop(j)
    else:
        new_clusters.pop(j)
        new_clusters.pop(i)
    
    # Add the merged cluster
    new_clusters.append(merged_cluster)
    
    return new_clusters
