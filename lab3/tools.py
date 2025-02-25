import numpy as np
import pandas as pd

# function to create new random df_points matrices
def create_random_points():
    return pd.DataFrame(
        np.random.rand(3, 2),
        index=['dog', 'cat', 'wolf'],
        columns=['mind_variable_1', 'mind_variable_2']
    )


def load_mental_representation():

    points=np.array([
            [0.8, 0.7],  # Dog: [loyalty, pack_behavior]
            [0.3, 0.4],  # Cat: [loyalty, pack_behavior]
            [0.7, 0.9]   # Wolf: [loyalty, pack_behavior]
    ])
    df_points = pd.DataFrame(
        points, 
        columns=['mind_variable_1', 'mind_variable_2'],
        index=['dog', 'cat', 'wolf']
    )

    return df_points

def load_sim_data(
        points=np.array([
            [0.8, 0.7],  # Dog: [loyalty, pack_behavior]
            [0.3, 0.4],  # Cat: [loyalty, pack_behavior]
            [0.7, 0.9]   # Wolf: [loyalty, pack_behavior]
        ]), 
        decay_rate=3.0
    ):
    """
    Generate similarity matrix based on Shepard's law: s = e^(-d)
    
    Parameters:
    -----------
    points : array-like
        Coordinates of points in n-dimensional space
    decay_rate : float
        Rate of similarity decay with distance (default=1.0)
    decimals : int
        Number of decimal places to round to (default=1)
        
    Returns:
    --------
    similarities : ndarray
        Square symmetric matrix of similarities
    """
    # Convert points to numpy array if not already
    points = np.array(points)
    n_points = len(points)
    
    # Initialize similarity matrix
    similarities = np.zeros((n_points, n_points))
    
    # Calculate pairwise distances and convert to similarities
    for i in range(n_points):
        for j in range(n_points):
            # Euclidean distance between points
            distance = np.linalg.norm(points[i] - points[j])
            # add a tiny bit of noise
            np.random.seed(0)
            # Apply Shepard's law and round
            similarities[i, j] = np.exp(-decay_rate * distance)

    labels = ['dog', 'cat', 'wolf']
    df_sim = pd.DataFrame(similarities, columns=labels, index=labels)
    
    return df_sim