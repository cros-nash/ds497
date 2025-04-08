import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

def plot_utility_curve(alpha=0.5, max=1000):
    x = np.linspace(0, max, 100)
    y = np.power(x, alpha)
    plt.plot(x, y, label=f'α={alpha}')
    plt.xlabel('x - Money')
    plt.ylabel('u(x) - Utility')

def plot_utility_curve_with_losses(alpha=0.5, max=1000):
    x = np.linspace(-max, max, 100)
    y = np.sign(x) * np.abs(x) ** alpha
    plt.plot(x, y, label=f'α={alpha}')
    plt.xlabel('x - Money')
    plt.ylabel('u(x) - Utility')

def plot_utility_curves():
    for alpha in [0.4, 0.5, 0.6, 0.7, 1.0]:
        plot_utility_curve(alpha=alpha)
    plt.legend()
    plt.ylim(0, 40)

# gambles for simulation
A_outcomes = np.array([100., 20.])
A_probs = np.array([0.7, 0.3])

B_outcomes = np.array([180., 40.])
B_probs = np.array([0.4, 0.6])

class ChoiceExperiment:
    def __init__(self, num_rounds=10):
        self.num_rounds = num_rounds
        self.current_round = 0
        self.total_earnings = 0
        self.choices = []
        self.outcomes = []
        self.waiting_for_next = False
        self.last_outcome = 0
        self.last_choice = ''
        
        # Create the UI
        self.output = widgets.Output()
        self.result_output = widgets.Output()
        self.button_a = widgets.Button(description='Choose Gamble A')
        self.button_b = widgets.Button(description='Choose Gamble B')
        self.next_button = widgets.Button(
            description='Ok, next trial',
            button_style='info',
            layout=widgets.Layout(display='none')
        )
        self.reset_button = widgets.Button(
            description='Start New Game', 
            button_style='warning',
            layout=widgets.Layout(display='none')
        )
        
        # Set up button callbacks
        self.button_a.on_click(lambda b: self.make_choice('A'))
        self.button_b.on_click(lambda b: self.make_choice('B'))
        self.next_button.on_click(lambda b: self.proceed_to_next())
        self.reset_button.on_click(lambda b: self.reset_game())
        
        # Display initial UI
        self.update_display()
    
    def make_choice(self, choice):
        if self.waiting_for_next:
            return
            
        # Record the choice
        self.last_choice = choice
        
        # Sample outcome based on choice
        if choice == 'A':
            outcome_idx = np.random.choice(len(A_outcomes), p=A_probs)
            outcome = A_outcomes[outcome_idx]
        else:  # choice == 'B'
            outcome_idx = np.random.choice(len(B_outcomes), p=B_probs)
            outcome = B_outcomes[outcome_idx]
        
        # Record outcome
        self.last_outcome = outcome
        
        # Show result and wait for confirmation
        self.waiting_for_next = True
        self.show_outcome()
    
    def show_outcome(self):
        with self.output:
            clear_output(wait=True)
            display(HTML(f"<h3>Round {self.current_round + 1} of {self.num_rounds}</h3>"))
            display(HTML(f"<p>Current earnings: ${self.total_earnings:.2f}</p>"))
            
            # Show the outcome with animation or highlight
            result_html = f"""
            <div style="margin: 20px 0; padding: 15px; background-color: #f0f8ff; border-radius: 5px; text-align: center;">
                <h4>Result</h4>
                <p>You chose <b>Gamble {self.last_choice}</b> and won <span style="color: green; font-weight: bold; font-size: 1.2em;">${self.last_outcome:.2f}</span>!</p>
            </div>
            """
            display(HTML(result_html))
            
            # Show next button
            self.next_button.layout.display = 'block'
            display(self.next_button)
    
    def proceed_to_next(self):
        # Add the outcome to records
        self.choices.append(self.last_choice)
        self.outcomes.append(self.last_outcome)
        self.total_earnings += self.last_outcome
        
        # Increment round counter
        self.current_round += 1
        
        # Reset waiting state
        self.waiting_for_next = False
        self.next_button.layout.display = 'none'
        
        # Check if game is over
        if self.current_round >= self.num_rounds:
            self.end_game()
        else:
            self.update_display()
    
    def update_display(self):
        with self.output:
            clear_output(wait=True)
            display(HTML(f"<h3>Round {self.current_round + 1} of {self.num_rounds}</h3>"))
            display(HTML(f"<p>Current earnings: ${self.total_earnings:.2f}</p>"))
            display(HTML("""
            <div style="margin-bottom: 10px;">
                <b>Gamble A:</b> <span style="color:blue">70%</span> chance of <span style="color:green">$100</span>, 
                <span style="color:blue">30%</span> chance of <span style="color:green">$20</span>
            </div>
            <div style="margin-bottom: 10px;">
                <b>Gamble B:</b> <span style="color:blue">40%</span> chance of <span style="color:green">$180</span>, 
                <span style="color:blue">60%</span> chance of <span style="color:green">$40</span>
            </div>
            """))
            display(widgets.HBox([self.button_a, self.button_b]))
            
    def end_game(self):
        with self.output:
            clear_output(wait=True)
            display(HTML(f"<h3>Game Over!</h3>"))
            display(HTML(f"<p>Total earnings: ${self.total_earnings:.2f}</p>"))
            
            # Create a summary of choices and outcomes
            results_html = "<h4>Your Results:</h4><table style='width:50%; border-collapse: collapse;'>"
            results_html += "<tr><th style='border:1px solid black; padding:8px;'>Round</th>"
            results_html += "<th style='border:1px solid black; padding:8px;'>Choice</th>"
            results_html += "<th style='border:1px solid black; padding:8px;'>Outcome</th></tr>"
            
            for i in range(self.num_rounds):
                results_html += f"<tr><td style='border:1px solid black; padding:8px;'>{i+1}</td>"
                results_html += f"<td style='border:1px solid black; padding:8px;'>Gamble {self.choices[i]}</td>"
                results_html += f"<td style='border:1px solid black; padding:8px;'>${self.outcomes[i]:.2f}</td></tr>"
                
            results_html += "</table>"
            display(HTML(results_html))
            
            # Calculate statistics
            a_choices = self.choices.count('A')
            b_choices = self.choices.count('B')
            display(HTML(f"<p>You chose Gamble A: {a_choices} times, Gamble B: {b_choices} times</p>"))
            
            # Display reset button
            self.reset_button.layout.display = 'block'
            display(self.reset_button)
    
    def reset_game(self):
        self.current_round = 0
        self.total_earnings = 0
        self.choices = []
        self.outcomes = []
        self.waiting_for_next = False
        self.reset_button.layout.display = 'none'
        self.update_display()
    
    def run(self):
        display(self.output)