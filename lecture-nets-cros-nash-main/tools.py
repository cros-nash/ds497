import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_step_function(step_function):
    z = np.linspace(0, 4, 100)
    phi_z = np.array([step_function(z_i, 2) for z_i in z])
    plt.plot(z, phi_z)
    plt.title(r'Step function $\phi$')
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\phi(z)$')
    plt.show()

def plot_stimuli(df):
    for i, row in df.iterrows():
        color = 'red' if row['feature1'] == 0 else 'blue'
        marker = 'o' if row['feature2'] == 0 else 's'
        plt.scatter(row['feature1'], row['feature2'], color=color, marker=marker, s=100)
        plt.text(row['feature1'] + 0.05, row['feature2'] + 0.05, str(row['category']), fontsize=12)
    plt.xlim(-0.5, 1.5), plt.ylim(-0.5, 1.5)
    plt.xticks([0, 1]), plt.yticks([0, 1])
    plt.xlabel('Feature 1 (Red or Blue)'), plt.ylabel('Feature 2 (Circle or Square)')