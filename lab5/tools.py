import numpy as np
import pandas as pd

def load_animal_features(with_weights=False):

    data = {
        "Mammal": [1, 1, 1, 0, 0, 0, 0, 1],  # Dog, Cat, Horse, Salamander, Snake, Eagle, Goldfish, Dolphin
        "Pet": [1, 1, 0, 0, 0, 0, 1, 0],  # Dog, Cat, Goldfish
        # "Can Fly": [0, 0, 0, 0, 0, 1, 0, 0],  # Eagle
        "Lives in Water": [0, 0, 0, 1, 0, 0, 1, 1],  # Salamander, Goldfish, Dolphin
        "Has Fur": [1, 1, 1, 0, 0, 0, 0, 0],  # Dog, Cat, Horse
        "Carnivore": [1, 1, 0, 1, 1, 1, 0, 1],  # Dog, Cat, Salamander, Snake, Eagle, Dolphin
        "Domesticated": [1, 1, 1, 0, 0, 0, 1, 0]  # Dog, Cat, Horse, Goldfish
    }

    # dataFrame with animal names as index
    animal_names = ["Dog", "Cat", "Horse", "Salamander", "Snake", "Eagle", "Goldfish", "Dolphin"]
    df = pd.DataFrame(data, index=animal_names)

    # feature_weights = np.array([0.30, 0.20, 0.15, 0.15, 0.10, 0.05, 0.05])
    feature_weights = np.array([0.30, 0.20, 0.15, 0.10, 0.05, 0.05])

    if with_weights:
        return df, feature_weights
    else:
        return df

def load_sim_data(dataset='animals'):
    if dataset == 'animals':
        df, weights = load_animal_features(with_weights=True)
    else:
        # df, weights = load_number_features()
        pass

    n_objects = df.shape[0]
    sims = np.full((n_objects, n_objects), np.nan)
    
    for i in range(n_objects):
        for j in range(n_objects):
            if i <= j:
                continue
            shared_features = df.iloc[i].values * df.iloc[j].values
            sim = np.sum(weights * shared_features)
            sims[i, j] = sim

    df_sim = pd.DataFrame(sims, index=df.index, columns=df.index)
    return df_sim