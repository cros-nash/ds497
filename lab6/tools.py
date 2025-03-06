import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import minimize
import itertools
import scipy.io

def load_first_guess():
    features = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    df = pd.DataFrame(
        features,
        index=[3, 4, 6, 8], 
        columns=["is_3", "is_4", "is_6", "is_8"]
    )
    return df

def compute_similarity(df_feats, weights):
    n_objects = df_feats.shape[0]
    similarities = np.full((n_objects, n_objects), np.nan)
    
    for i in range(n_objects):
        for j in range(n_objects):
            if i <= j:
                continue
            shared_features = df_feats.iloc[i].values * df_feats.iloc[j].values
            similarity = np.sum(weights * shared_features)
            similarities[i, j] = similarity

    df_sim = pd.DataFrame(similarities, index=df_feats.index, columns=df_feats.index)
    return df_sim

def load_sim_data(dataset='numbers_subset'):
    if dataset == 'numbers_subset':
        df, weights = load_number_features(with_weights=True, subset=True)
    elif dataset == 'numbers_full':
        df, weights = load_number_features(with_weights=True, subset=False)

    df_sim = compute_similarity(df, weights)
    return df_sim

def load_number_features(with_weights=False, subset=True):

    # features from Shepard's paper
    feats = {
        "Powers of Two": [1 if i in [2, 4, 8] else 0 for i in range(10)],
        "Large Numbers": [1 if i in [6, 7, 8, 9] else 0 for i in range(10)],
        "Middle Numbers": [1 if i in [3, 4, 5, 6] else 0 for i in range(10)],
        "Small Nonzero Numbers": [1 if i in [1, 2, 3] else 0 for i in range(10)],
        "Multiples of Three": [1 if i in [3, 6, 9] else 0 for i in range(10)],
        "Additive & Multiplicative Identities": [1 if i in [0, 1] else 0 for i in range(10)],
        "Odd Numbers": [1 if i in [1, 3, 5, 7, 9] else 0 for i in range(10)],
        "Moderately Large Numbers": [1 if i in [5, 6, 7] else 0 for i in range(10)],
        "Small Numbers": [1 if i in [0, 1, 2] else 0 for i in range(10)],
        "Smallish Numbers": [1 if i in [0, 1, 2, 3, 4] else 0 for i in range(10)]
    }
    df = pd.DataFrame(feats, index=range(10))

    # weights from Shepard's paper
    weights = np.array([0.577, 0.326, 0.305, 0.299, 0.277, 0.165, 0.150, 0.138, 0.112, 0.101])

    # a simpler subset of the data for learning purposes
    if subset:
        
        # only look at three of the features
        feature_subset = [
            "Powers of Two",
            "Large Numbers",
            "Middle Numbers"
        ]
        feature_subset_indices = [0, 1, 2]
        
        # only look at numbers 3, 4, 6, 8
        number_subset = [3, 4, 6, 8]

        # take feature subset
        df = df.loc[number_subset, feature_subset]

        # take weight subset
        weights = weights[feature_subset_indices]

    if with_weights:
        return df, weights
    else:
        return df
    
def compute_error(sim, sim_hat):
    n = sim.shape[0]
    error_sum = 0
    for i in range(n):
        for j in range(n):
            if i <= j:
                continue
            error = (sim.iloc[i, j] - sim_hat.iloc[i, j]) ** 2
            error_sum += error
    return error_sum

def error_given_weights(weights, df_feats, human_sim_df):
    estimated_sim = compute_similarity(df_feats, weights)
    se = compute_error(human_sim_df, estimated_sim)
    return se

def get_best_weights_given_features(df_feats, human_sim_df):
    # bound weights to be non-negative
    # don't have a right bound
    output = minimize(
        error_given_weights, 
        np.ones(df_feats.shape[1])*0.1, args=(df_feats, human_sim_df),
        bounds=((0, None),) * df_feats.shape[1],
    )
    best_weights = np.array(output.x)
    return best_weights

def generate_all_binary_matrices(n_objects, n_features):

    if n_objects > 4 or n_features > 3:
        raise ValueError("Too large to generate.")

    column_names = [f"feature{i+1}" for i in range(n_features)]
    if n_objects == 4:
        row_names = [3, 4, 6, 8]
    else:
        row_names = list(range(1, n_objects+1))

    reps = []
    for rep in itertools.product([0, 1], repeat=n_objects*n_features):
        df = pd.DataFrame(
            np.array(rep).reshape(n_objects, n_features), 
            index=row_names, 
            columns=column_names
        )
        reps.append(df)

    return reps

def load_numbers_full():
    mat_file = scipy.io.loadmat('abstractnumbers.mat')
    sim_matrix = mat_file['s']  # get the similarity matrix
    
    # reorder the matrix: need to reorder both rows and columns
    reordered = sim_matrix[[9, 0, 1, 2, 3, 4, 5, 6, 7, 8], :]  # reorder rows
    reordered = reordered[:, [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]]  # reorder columns
    
    df = pd.DataFrame(
        reordered,
        index=range(10),
        columns=range(10)
    )
    return df

def check_ex2():
    return 0

def check_ex3():
    return 4

def check_ex12():
    return 1267650600228229401496703205376

def check_ex13():
    return [0, 4, 5, 6]

def check_ex15():
    return 2, 4

def plot_spatial_representation(points, features=None):
    # Create figure (single plot)
    plt.figure(figsize=(7, 7))
    
    # Plot 2D NMDS solution
    plt.scatter(points[:, 0], points[:, 1], s=50, c='black', alpha=1.0)

    # Add number labels
    for i, num in enumerate(range(10)):
        plt.annotate(str(num), (points[i, 0], points[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=14, fontweight='bold')
        
    if features is not None:
        # Draw ellipses for features
        feature_colors = plt.cm.tab10(np.linspace(0, 1, features.shape[1]))
        for feat_idx, feature_name in enumerate(features.columns):
            # Get indices of numbers that have this feature
            feature_indices = np.where(features[feature_name].values == 1)[0]
            
            if len(feature_indices) > 1:  # Need at least two points for an ellipse
                # Get points for this feature
                feature_points = points[feature_indices]
                
                # Calculate centroid
                centroid = np.mean(feature_points, axis=0)
                
                # Calculate covariance to determine ellipse shape
                # Add a small value to ensure non-zero covariance
                cov = np.cov(feature_points, rowvar=False) + np.eye(2) * 0.01
                
                # Calculate eigenvalues and eigenvectors of covariance matrix
                eigvals, eigvecs = np.linalg.eig(cov)
                
                # Calculate width and height of ellipse (3 std devs covers ~95% of points)
                width, height = 3 * np.sqrt(eigvals)
                
                # Calculate angle of ellipse
                angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                
                # Create ellipse
                # outline but not filled
                ellipse = Ellipse(
                    xy=centroid, width=width, height=height, 
                    angle=angle, alpha=0.7,
                    edgecolor=feature_colors[feat_idx], linewidth=1,
                    fill=False
                )
                
                # Add ellipse to plot
                plt.gca().add_patch(ellipse)

    plt.title("2D Spatial Representation with Features Circled", fontsize=14)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)

    plt.tight_layout()
    plt.show()

import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

def find_and_plot_hierarchical_features(df_sim):

    similarities = df_sim.values
    dissimilarities = 1 - similarities

    # Perform hierarchical clustering
    Z = sch.linkage(squareform(dissimilarities), method='average')

    # Visualization of the hierarchy
    plt.figure(figsize=(10, 6))
    dendrogram = sch.dendrogram(
        Z,
        labels=df_sim.index, # Use the indices as labels
        leaf_font_size=12,
    )
    plt.title('Hierarchical Clustering', fontsize=14)
    plt.ylabel('Dissimilarity', fontsize=12)
    plt.tight_layout()
    plt.show()

    return Z

def hierarchical_to_feature_matrix(Z, labels):
    """Convert hierarchical clustering results to a feature matrix"""
    n = len(labels)
    # Each merge in Z creates a potential feature
    features = []
    
    # Initialize clusters - each object starts in its own cluster
    clusters = [[i] for i in range(n)]
    
    # For each merge in Z
    for i, merge in enumerate(Z):
        # Get the two clusters being merged
        cluster1 = clusters[int(merge[0])]
        cluster2 = clusters[int(merge[1])]
        
        # Create a new cluster by merging these two
        new_cluster = cluster1 + cluster2
        clusters.append(new_cluster)
        
        # Create a feature for this cluster (if it contains more than 1 and fewer than n objects)
        if 1 < len(new_cluster) < n:
            feature = np.zeros(n)
            for idx in new_cluster:
                feature[idx] = 1
            features.append(feature)
    
    # Convert to DataFrame
    feature_names = [f"hier_feature{i+1}" for i in range(len(features))]
    df_features = pd.DataFrame(np.column_stack(features), 
                              index=labels, 
                              columns=feature_names)
    
    return df_features.astype(int)