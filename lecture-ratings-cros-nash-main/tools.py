import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_projection_to_x(ratings):

    # Standardize the features for visualization
    responsible = (ratings['responsible'] - ratings['responsible'].mean()) / ratings['responsible'].std()
    trustworthy = (ratings['trustworthy'] - ratings['trustworthy'].mean()) / ratings['trustworthy'].std()

    # Create plot to demonstrate projection onto x-axis (responsible dimension)
    plt.figure(figsize=(10, 6))

    # Plot original data points
    plt.scatter(responsible, trustworthy, alpha=0.7, label='Original data')

    # Project all points onto the x-axis (responsible dimension)
    # This means setting y=0 for all points
    plt.scatter(responsible, np.zeros_like(responsible), color='red', alpha=0.7, 
            label='Projection onto "responsible" axis')

    # Draw projection lines from original points to their projections
    for i in range(len(responsible)):
        plt.plot([responsible.iloc[i], responsible.iloc[i]], 
                [trustworthy.iloc[i], 0], 'k--', alpha=0.3)

    # Add axis lines
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.5)

    # # Indicate the variance along the responsible axis
    # resp_variance = np.var(responsible)
    # total_variance = np.var(responsible) + np.var(trustworthy)
    # variance_explained = resp_variance / total_variance * 100

    # Add annotation
    # plt.annotate(f'Variance = {resp_variance:.2f}\n({variance_explained:.1f}% of total)',
    #              xy=(0.05, 0.95), xycoords='axes fraction',
    #              bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

    plt.grid(True, alpha=0.3)
    plt.xlabel('responsible')
    plt.ylabel('trustworthy')
    # plt.title('Projection of Points onto the Responsible Dimension')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # # Print the variance calculation
    # print(f"Responsible variance: {resp_variance:.3f}")
    # print(f"Total variance: {total_variance:.3f}")
    # print(f"Responsible dimension explains {variance_explained:.1f}% of total variance")

def plot_single_point_projection():

    def project_point_onto_line(x, y, slope):
        x_p = (x + slope*y) / (slope**2 + 1)
        y_p = x_p * slope
        return x_p, y_p

    # Define a line: y = 0.7x + 0.2
    slope = 0.7
    intercept = 0.0 #0.2

    # Define a point
    x0, y0 = 2.0, 1.0

    # Project the point onto the line
    # x_p, y_p = project_point_onto_line(x0, y0, slope, intercept)
    x_p, y_p = project_point_onto_line(x0, y0, slope)

    # Visualization
    plt.figure(figsize=(8, 6))

    # Plot the line
    x_line = np.linspace(-1, 4, 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, 'b-', label=f'y = {slope}x')# + {intercept}')

    # Plot the original point
    plt.plot(x0, y0, 'ro', markersize=8, label='Original point ($x, y$)')

    # Plot the projected point
    plt.plot(x_p, y_p, 'go', markersize=8, label='Projected point ($x_p, y_p$)')

    # Plot the projection line (perpendicular to the original line)
    plt.plot([x0, x_p], [y0, y_p], 'k--', label='Projection line')

    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    # plt.title('Projection of a Point onto a Line')
    plt.xlabel('x')
    plt.ylabel('y')

    # Add text annotations
    plt.text(x0 + 0.1, y0 + 0.1, f'({x0}, {y0})')
    plt.text(x_p + 0.1, y_p + 0.1, f'({x_p:.2f}, {y_p:.2f})')

    plt.show()

    # print(f"Original point: ({x0}, {y0})")
    # print(f"Projected point: ({x_p:.4f}, {y_p:.4f})")

def check_face_answers(student_answers):
    """
    Check if student answers for highest-rated faces match the actual data.
    
    Parameters:
    - ratings (pandas.DataFrame): DataFrame containing face ratings
    - student_answers (dict): Dictionary with trait names as keys and face numbers (1-6) as values
    
    Returns:
    - dict: Feedback on incorrect answers with correct values
    """

    ratings = pd.read_csv('face_ratings.csv')

    # Get the subset of faces we're testing
    face_subset = ratings[ratings.stimulus.isin(['AF01NES', 'AM01NES', 'AF02NES', 'AM02NES', 'AF03NES', 'AM03NES'])]
    
    # Map stimulus names to their position (1-6)
    face_numbers = {
        'AF01NES': 1,
        'AM01NES': 2,
        'AF02NES': 3,
        'AM02NES': 4,
        'AF03NES': 5,
        'AM03NES': 6
    }
    
    # Calculate correct answers
    correct_answers = {}
    for column in face_subset.columns:
        if column != 'stimulus':
            # Get the stimulus with maximum value for this trait
            max_stimulus = face_subset.loc[face_subset[column].idxmax(), 'stimulus']
            # Map to face number (1-6)
            correct_answers[column] = face_numbers[max_stimulus]
    
    # Check student answers against correct ones
    incorrect = {}
    for trait, correct_value in correct_answers.items():
        student_key = f"most_{trait}"
        if student_key in student_answers:
            if student_answers[student_key] != correct_value:
                incorrect[student_key] = correct_value
    
    # Return feedback
    if not incorrect:
        print("All answers are correct! Great job!")
        return {}
    else:
        print(f"You got {len(incorrect)} trait(s) wrong.")
        # print(f"You got {len(incorrect)} traits wrong. Here are the correct answers:")
        # for trait, correct_value in incorrect.items():
            # print(f"{trait} should be {correct_value}")
        return incorrect

def plot_triangles():
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1, 1, 0], [0, 1, 0, 0], color='black')
    plt.plot([0, -1, -1, 0], [0, -1, 0, 0], color='black')
    plt.plot([-1, 1], [-1, 1], color='red')
    plt.scatter([1, -1], [1, -1], color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()