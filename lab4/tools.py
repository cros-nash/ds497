import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_mental_representation():
    """
    Load mental representation of animals in 2D space
    
    Returns:
    --------
    points : ndarray
        Coordinates of points in 2D space
    """
    points = np.array([
        [0.8, 0.6],  # Dog: [size, fur_length]
        [0.3, 0.7],  # Cat: [size, fur_length]
        [1.0, 0.8],  # Wolf: [size, fur_length]
        [0.2, 0.4]   # Rabbit: [size, fur_length]
    ])
    return pd.DataFrame(
        points, 
        columns=['size', 'fur_length'], 
        index=['dog', 'cat', 'wolf', 'rabbit']
    )

def load_sim_data():
    """
    Generate similarity matrix based on Shepard's law: s = e^(-d)
    
    Parameters:
    -----------
    points : array-like
        Coordinates of points in n-dimensional space
    decay_rate : float
        Rate of similarity decay with distance (default=1.0)
        
    Returns:
    --------
    similarities : ndarray
        Square symmetric matrix of similarities
    """
    points = load_mental_representation().values
    
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
            # Apply Shepard's law with decay_rate=1.0
            similarities[i, j] = np.exp(-distance)
    
    labels = ['dog', 'cat', 'wolf', 'rabbit']
    df_sim = pd.DataFrame(similarities, columns=labels, index=labels)
    
    return df_sim

def flatten_lower_triangle(matrix):
    if type(matrix) == pd.DataFrame:
        matrix = matrix.values
    flat = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i > j:
                flat.append(matrix[i, j])
    return np.array(flat)

def get_point(df, animal):
    return df.loc[animal].values

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def get_pairwise_distances(df):
    distances = pd.DataFrame(index=df.index, columns=df.index)
    for animal1 in df.index:
        point1 = get_point(df, animal1)
        for animal2 in df.index:
            point2 = get_point(df, animal2)
            distances.loc[animal1, animal2] = euclidean_distance(point1, point2)
    return distances

def get_gradient_for_one_point(df_inferred_points, df_sim, df_inferred_distances, animal):
    point = get_point(df_inferred_points, animal)

    # gradients toward individual points
    directions = pd.DataFrame(index=df_inferred_points.index, columns=['dim1', 'dim2'])
    directions.loc[animal] = [0.0, 0.0]

    # sum of the above, overall gradient
    gradient = np.zeros(2)

    for other_animal in df_inferred_points.index:
        if animal == other_animal:
            continue
        other_point = get_point(df_inferred_points, other_animal)
        sim = df_sim.loc[animal, other_animal]
        neg_log_sim = -np.log(sim)
        dist = df_inferred_distances.loc[animal, other_animal]
        diff = point - other_point
        abs_error = dist - neg_log_sim
        grad = 2.0 * abs_error * (diff / dist)
        directions.loc[other_animal] = grad
        # print(type(gradient[0]), type(grad[0]))
        gradient += grad.astype(float)
    return gradient, directions

def get_grads_for_all_points(df_inferred_points, df_sim, df_inferred_distances):
    gradients = pd.DataFrame(index=df_inferred_points.index, columns=['dim1', 'dim2'])
    for animal in df_inferred_points.index:
        grad, _ = get_gradient_for_one_point(df_inferred_points, df_sim, df_inferred_distances, animal)
        gradients.loc[animal] = grad
    return gradients

def update_points(df_guesses_new, df_sim, lr=0.2):
    df_inferred_dist_new = get_pairwise_distances(df_guesses_new)
    gradients = get_grads_for_all_points(df_guesses_new, df_sim, df_inferred_dist_new)
    df_guesses_new = df_guesses_new - lr * gradients.values
    return df_guesses_new