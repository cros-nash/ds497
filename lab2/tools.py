import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import random

def load_craik_tulving(v=1):

    if v == 2:
        mem = {
            'score': [12, 12, 11, 10, 13, 10, 9, 11, 12, 12, 9, 8, 10, 11, 7, 11, 9, 10, 10, 8],
            'condition': ['deep'] * 10 + ['shallow'] * 10
        }
    else:
        mem = {
            'score': [12, 9, 11, 9, 13, 10, 9, 11, 10, 12, 9, 8, 10, 11, 7, 11, 9, 10, 10, 8],
            'condition': ['deep'] * 10 + ['shallow'] * 10
        }

    mem = pd.DataFrame(mem)

    return mem

def load_shepard():
    rotation_data = {
        'angle': [0, 20, 40, 60, 80, 100, 120, 140, 160, 180],
        'rt': [1023, 1167, 1382, 1578, 1842, 1976, 2198, 2445, 2583, 2791]
    }
    return pd.DataFrame(rotation_data)

def load_hick():
    # Set random seed for reproducibility
    np.random.seed(1)

    base_rts = {
        1: 180,   # fastest with just one option
        2: 250,
        3: 290,
        4: 310,
        5: 325,
        6: 335,
        7: 345,
        8: 350,
        9: 355,
        10: 360   # starts to level off with many alternatives
    }

    hick_data = {
        'n_alternatives': [],
        'rt': []
    }

    for n, base_rt in base_rts.items():
        hick_data['n_alternatives'].append(n)
        # Add noise with higher variability for longer RTs
        noise = np.random.normal(0, base_rt * 0.01)  # 10% of base RT as std dev
        rt = max(base_rt + noise, 150)  # ensure no impossibly fast RTs
        hick_data['rt'].append(round(rt, 1))

    return pd.DataFrame(hick_data)

import ipywidgets as widgets
from ipywidgets import interact

def standardize(x):
    return (x - np.mean(x)) / np.std(x)

def correlation(x, y):
    x = standardize(x)
    y = standardize(y)
    return np.mean(x * y)

def rank(x):
    return np.argsort(np.argsort(x)) + 1

def spearman(x, y):
    x = rank(x)
    y = rank(y)
    return correlation(x, y)

def plot_scorrelation(strength):
    x = np.linspace(0.1, 10, 100)
    plt.clf()
    
    # Base relationship
    y_base = np.log(x)
    
    # Scale and shift to achieve target correlation
    y = y_base * np.sign(strength)
    
    # Add noise to adjust correlation
    noise = np.random.normal(0, 1.0, len(x))
    y = abs(strength) * y + (1 - abs(strength)) * noise
    
    # Calculate actual Spearman correlation
    correlation = spearman(x, y)
    
    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 10)
    plt.ylim(-3, 3)
    plt.title(f"Spearman's rho: {correlation:.3f}")
    plt.show()
    plt.close()

def explore_spearman():
    interact(
        plot_scorrelation, 
        strength=widgets.FloatSlider(min=-1, max=1, step=0.01, value=1)
    )

def plot_pcorrelation(strength):
    x = np.linspace(-5, 5, 100)
    plt.clf()
    
    # Base linear relationship
    y = x * np.sign(strength)
    
    # Add controlled noise
    noise = np.random.normal(0, 1.0, len(x))
    y = abs(strength) * y + (1 - abs(strength)) * noise
    
    # Calculate actual Pearson correlation
    correlation = np.corrcoef(x, y)[0,1]
    
    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title(f"Pearson's r: {correlation:.3f}")
    plt.show()
    plt.close()

def explore_pearson():
    interact(
        plot_pcorrelation,
        strength=widgets.FloatSlider(min=-1, max=1, step=0.01, value=1)
    )

def levels_experiment():
    # Study items
    deep_items = [
        ("TIGER - Is this an animal?", "deep"),
        ("APPLE - Is this edible?", "deep"),
        ("HAMMER - Is this a tool?", "deep"),
        ("PIANO - Does this make music?", "deep"),
        ("CHAIR - Can you sit on this?", "deep")
    ]
    
    shallow_items = [
        ("HOUSE - Contains letter E?", "shallow"),
        ("PLANT - Five letters long?", "shallow"),
        ("CLOCK - Starts with C?", "shallow"),
        ("BOOK - All capital letters?", "shallow"),
        ("TRAIN - Contains letter A?", "shallow")
    ]

    lure_items = [
        ("DESK", "new"), ("BIRD", "new"), 
        ("LAMP", "new"), ("PHONE", "new"),
        ("SHOE", "new"), ("LAKE", "new"),
        ("BRUSH", "new"), ("CLOUD", "new"),
        ("FORK", "new"), ("RING", "new")
    ]

    # Study phase
    study_items = deep_items + shallow_items
    random.shuffle(study_items)
    
    print("Study Phase - Watch each item carefully\n")
    time.sleep(2)
    
    for item, _ in study_items:
        clear_output(wait=True)
        print(f"\n\n{item}\n\n")
        time.sleep(3)
    
    # Test phase
    clear_output(wait=True)
    print("Memory Test\n")
    time.sleep(2)
    
    test_items = [(item.split(' - ')[0], condition) for item, condition in study_items]
    test_items.extend(lure_items)
    random.shuffle(test_items)
    
    results = {'deep': [], 'shallow': [], 'new': []}
    for word, condition in test_items:
        response = input(f"Did you see {word}? (y/n): ").lower() == 'y'
        results[condition].append(response)
    
    deep_correct = sum(results['deep'])
    shallow_correct = sum(results['shallow'])
    false_alarms = sum(results['new'])

    print(f"\nDeep Processing: {deep_correct}/5 words recalled")
    print(f"Shallow Processing: {shallow_correct}/5 words recalled")
    print(f"False Alarms: {false_alarms}/10 new words incorrectly recognized")

    return deep_correct, shallow_correct, false_alarms
