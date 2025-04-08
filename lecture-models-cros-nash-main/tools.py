import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_learning_phase_stimuli():
    # Learning phase stimuli
    learning_data = pd.DataFrame({
        'stimulus': ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5'],
        'category': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
        'size': np.array([0.35, 0.28, 0.40, 0.30, 0.36, 0.45, 0.38, 0.50, 0.43, 0.48]),
        'color': [0.40, 0.32, 0.45, 0.25, 0.37, 0.50, 0.58, 0.54, 0.62, 0.45],
    })
    return learning_data

def load_test_phase_data(with_proportions=False):
    if with_proportions:
        return pd.DataFrame({
            'stimulus': ['Test1', 'Test2', 'Test3', 'Test4', 'Test5', 'Test6', 'Test7', 'Test8', 'Test9'],
            'size': [0.30, 0.60, 0.45, 0.35, 0.55, 0.40, 0.25, 0.50, 0.35],
            'color': [0.45, 0.55, 0.50, 0.35, 0.35, 0.60, 0.55, 0.40, 0.60],
            # 'categorization': ['A', 'B', 'B', 'A', 'B', 'B', 'A', 'B', 'A']
            # 'categorization': ['A', 'B', 'B', 'A', 'B', 'B', 'A', 'B', 'B']
            'prop_chose_A': [0.85, 0.10, 0.40, 0.70, 0.15, 0.30, 0.60, 0.25, 0.50]
        })
    else:
        return pd.DataFrame({
            'stimulus': ['Test1', 'Test2', 'Test3', 'Test4', 'Test5', 'Test6', 'Test7', 'Test8', 'Test9'],
            'size': [0.30, 0.60, 0.45, 0.35, 0.55, 0.40, 0.25, 0.50, 0.35],
            'color': [0.45, 0.55, 0.50, 0.35, 0.35, 0.60, 0.55, 0.40, 0.60],
            # 'categorization': ['A', 'B', 'B', 'A', 'B', 'B', 'A', 'B', 'A']
            'categorization': ['A', 'B', 'B', 'A', 'B', 'B', 'A', 'B', 'B']
            # 'prop_chose_A': [0.85, 0.10, 0.40, 0.70, 0.15, 0.30, 0.60, 0.25, 0.50]
        })

def plot_stimuli(df_stims, stimuli=True, edges=False, prototypes=None, legend=True):

    plt.figure(figsize=(6, 6))

    marker_size_bias = 1
    marker_size_mult = 1000

    # Plot the category members - using actual size and color in the markers
    for i, row in df_stims.iterrows():

        if not stimuli:
            break

        # Size is represented by marker size
        marker_size = marker_size_bias + (row['size'] * marker_size_mult)
        
        # Color is represented by grayscale (from light to dark)
        marker_color = str(1 - row['color'])  # Convert to grayscale string
        
        # Add edge color based on category if it exists
        edge_color = 'none'
        if edges and 'category' in df_stims.columns:
            if row['category'] == 'A':
                edge_color = 'red'
            elif row['category'] == 'B':
                edge_color = 'blue'
        
        plt.scatter(row['size'], row['color'], 
                    s=marker_size, color=marker_color, edgecolor=edge_color, linewidth=1,
                    alpha=1.0)
        
        # Add object labels
        plt.annotate(row['stimulus'], 
                    (row['size'], row['color']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10)


    # If prototypes should be plotted
    if prototypes is not None and 'category' in df_stims.columns:
        # Compute prototypes for each category
        categories = df_stims['category'].unique()
        
        for category in categories:
            # Filter by category
            cat_data = df_stims[df_stims['category'] == category]
            
            # Compute prototype as the mean of size and color
            prototype = [cat_data['size'].mean(), cat_data['color'].mean()]
            
            # Set color based on category
            edge_color = 'red' if category == 'A' else 'blue'
            if not edges:
                edge_color = 'none'
            
            # Plot the prototype
            # if prototype is "render", show prototype as a stimulus (size/color set using means)
            # otherwise, show the start
            if prototypes == 'render':
                marker_size = marker_size_bias + (prototype[0] * marker_size_mult)
                marker_color = str(1 - prototype[1]) 
                plt.scatter(prototype[0], prototype[1], s=marker_size, 
                            color=marker_color, edgecolor=edge_color, linewidth=1)
                plt.annotate(f'{category} Prototype', 
                    (prototype[0], prototype[1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10)
            else:
                edge_color = 'red' if category == 'A' else 'blue'
                plt.scatter(prototype[0], prototype[1], marker='*', s=300, 
                            color='white', edgecolor=edge_color, linewidth=1,
                            label=f'Prototype {category}')
        if legend:
            # Add legend if prototypes are shown
            plt.legend(fontsize=10)

    # Format the plot
    plt.xlabel('Size Feature', fontsize=12)
    plt.ylabel('Color Feature (Darkness)', fontsize=12)
    plt.xlim(0.2, 0.6)
    plt.ylim(0.2, 0.7)
    plt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()