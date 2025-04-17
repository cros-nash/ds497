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

def update_mlp_params(params, datapoint, eta):
    """
    Update MLP parameters using gradient descent for a single datapoint.
    
    Args:
        params: List of 9 parameters [n1_w1, n1_w2, n1_b, n2_w1, n2_w2, n2_b, n3_w1, n3_w2, n3_b]
        datapoint: Dictionary with 'feature1', 'feature2', and 'category' keys
        eta: Learning rate
        
    Returns:
        Updated parameters after one step of gradient descent
    """
    # Unpack parameters
    n1_w1, n1_w2, n1_b, n2_w1, n2_w2, n2_b, n3_w1, n3_w2, n3_b = params
    
    # Get inputs and target output
    x1 = datapoint['feature1']
    x2 = datapoint['feature2']
    y = datapoint['category']
    
    # Forward pass
    # Calculate neuron 1 activation (hidden layer)
    z1 = x1 * n1_w1 + x2 * n1_w2 + n1_b
    a1 = sigmoid(z1)
    
    # Calculate neuron 2 activation (hidden layer)
    z2 = x1 * n2_w1 + x2 * n2_w2 + n2_b
    a2 = sigmoid(z2)
    
    # Calculate neuron 3 activation (output layer)
    z3 = a1 * n3_w1 + a2 * n3_w2 + n3_b
    y_pred = sigmoid(z3)
    
    # Backpropagation
    # Output layer error
    error_output = y - y_pred
    delta3 = error_output * sigmoid_derivative(z3)
    
    # Gradients for output layer
    grad_n3_w1 = delta3 * a1
    grad_n3_w2 = delta3 * a2
    grad_n3_b = delta3
    
    # Hidden layer errors
    delta1 = delta3 * n3_w1 * sigmoid_derivative(z1)
    delta2 = delta3 * n3_w2 * sigmoid_derivative(z2)
    
    # Gradients for hidden layer
    grad_n1_w1 = delta1 * x1
    grad_n1_w2 = delta1 * x2
    grad_n1_b = delta1
    
    grad_n2_w1 = delta2 * x1
    grad_n2_w2 = delta2 * x2
    grad_n2_b = delta2
    
    # Update parameters (gradient descent)
    n1_w1 = n1_w1 + eta * grad_n1_w1
    n1_w2 = n1_w2 + eta * grad_n1_w2
    n1_b = n1_b + eta * grad_n1_b
    
    n2_w1 = n2_w1 + eta * grad_n2_w1
    n2_w2 = n2_w2 + eta * grad_n2_w2
    n2_b = n2_b + eta * grad_n2_b
    
    n3_w1 = n3_w1 + eta * grad_n3_w1
    n3_w2 = n3_w2 + eta * grad_n3_w2
    n3_b = n3_b + eta * grad_n3_b
    
    return [n1_w1, n1_w2, n1_b, n2_w1, n2_w2, n2_b, n3_w1, n3_w2, n3_b]

# Helper function for sigmoid derivative
def sigmoid_derivative(z):
    sig_z = sigmoid(z)
    return sig_z * (1 - sig_z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))