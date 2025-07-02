import random                     # Import Python's built-in module for generating random numbers.
import numpy as np                # Import numpy for numerical operations.
import math                       # Import math module for mathematical functions.
import pandas as pd               # Import pandas to work with data structures like DataFrame.
import matplotlib.pyplot as plt   # Import pyplot from matplotlib for plotting graphs.
import tkinter as tk              # Import tkinter for GUI related operations.

from Environment import Environment   # Import the Environment class defined in Environment.py.
from Parameters import *                # Import all constants and parameters from Parameters.py.

# Set the random seed for both numpy and random modules to ensure reproducibility.
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class SARSA(object):
    def __init__(self, env, learning_rate, gamma, epsilon):
        # Store the environment object and parameters for learning.
        self.env = env
        # List of available actions, obtained by creating a list from 0 to (num_actions - 1).
        self.actions = list(range(self.env.num_actions))
        # Set the learning rate (alpha) for updating Q-values.
        self.lr = learning_rate
        # Discount factor (gamma) used to balance immediate and future rewards.
        self.gamma = gamma
        # Epsilon value for the epsilon-greedy policy (initial exploration rate).
        self.epsilon = epsilon  # Initial epsilon for SARSA

        # Construct the Q-table for all states.
        # The states are represented as stringified integers.
        # Using pandas DataFrame with rows as states and columns as actions, initialized to 0.
        all_states = [str(s) for s in range(self.env.num_states)]
        self.q_table = pd.DataFrame(0, index=all_states, columns=self.actions)

    def state_to_label(self, state):
        """
        Convert a state number to a string representation of (x, y) coordinates.
        For example, for a 10x10 grid, state 0 corresponds to "0_0" and state 14 corresponds to "4_1".
        """
        x = state % GRID_SIZE              # Compute x-coordinate (column) using modulo.
        y = state // GRID_SIZE             # Compute y-coordinate (row) using integer division.
        return f"{x}_{y}"                  # Return the coordinate as a string "x_y".

    def print_q_table(self):
        """
        Print the complete Q-table and a mapping of state indices to (x, y) coordinates,
        which helps to understand which grid cell each row represents.
        """
        print("Full Q-table:")
        print(self.q_table)                # Print the Q-table.
        mapping = {str(s): self.state_to_label(s) for s in range(self.env.num_states)}
        print("State mapping (state index: coordinate):")
        for key, value in mapping.items():
            print(f"{key}: {value}")       # Print each state's mapping.

    def epsilon_greedy_policy(self, observation):
        """
        Choose an action based on the epsilon-greedy strategy.
        Convert the observation (state) to its string representation and either:
          - With probability epsilon, select a random action (exploration).
          - Otherwise, select the action with the highest Q-value (exploitation).
        """
        state = str(observation)           # Convert current state to string for indexing in Q-table.
        if np.random.uniform() < self.epsilon: # Check if a randomly generated number is less than epsilon.
            action = np.random.choice(self.actions)  # Choose random action if within epsilon probability.
        else:
            state_action = self.q_table.loc[state, :] # Get Q-values for the current state.
            action = np.random.choice(state_action[state_action == np.max(state_action)].index) # Choose the best action.
        return action                     # Return the selected action.

    def optimal_policy(self, observation):
        """
        Determine the optimal action for a given state based on the current Q-table.
        Returns the action with the highest Q-value.
        """
        state = str(observation)           # Convert observation to string.
        state_action = self.q_table.loc[state, :]
        # Return one of the best actions (if multiple have the same Q-value, choose randomly).
        return int(np.random.choice(state_action[state_action == np.max(state_action)].index))

    def train(self, num_epd):
        """
        Train the agent using the SARSA (State-Action-Reward-State-Action) algorithm.
        This method runs for a given number of episodes (num_epd) and performs the following steps:
          - Reset the environment at the start of each episode.
          - Select an initial action using the epsilon-greedy policy.
          - For each time step in the episode, update the Q-table using the SARSA update rule:
                Q(s,a) = Q(s,a) + alpha * (reward + gamma * Q(s',a') - Q(s,a))
          - Decay epsilon after each episode.
          - Record various metrics (time steps, cumulative cost, success rate, average reward) for plotting.
        """
        # Initialize lists and variables for tracking training progress.
        time_steps = []                   # List to store the number of steps taken in each episode.
        MSE_Q = []                        # List to store the cumulative cost (sum of Q-value updates) per episode (modified from episode_cumulative_cost)
        sucess_rate = []                  # List to store the success rate every 50 episodes.
        average_reward = []               # List to store the cumulative average reward per episode.
        Q_value = {}                      # Dictionary to store Q values at intervals (for analysis).
        goal_count = 0                    # Counter for successful episodes (e.g., reaching goal).
        rewards = 0                       # Accumulator for total rewards.
        positive_count = 0                # Counter for episodes with positive reward.
        negative_count = 0                # Counter for episodes with negative reward.

        # Loop over each episode.
        for i in range(num_epd):
            observation = self.env.reset()  # Reset the environment to the initial state.
            action = self.epsilon_greedy_policy(observation)  # Choose the initial action using epsilon-greedy.
            step = 0                      # Initialize step counter for the episode.
            cost = 0                      # Initialize cumulative cost for the episode.

            # Every 50 episodes, calculate and store the success rate, then reset the goal counter.
            if i != 0 and i % 50 == 0:
                sucess_rate.append(goal_count / 50)
                goal_count = 0

            # Every 1000 episodes, record the Q-values for a specific state (state 14) for analysis.
            if i != 0 and i % 1000 == 0:
                Q_value[i] = []
                state_label = str(14)
                for j in self.actions:
                    Q_value[i].append(self.q_table.loc[state_label, j])

            # SARSA algorithm: interact with the environment until the episode ends.
            while True:
                # Execute the current action and obtain the next state, reward, and termination flag.
                next_observation, reward, done, info = self.env.step(action) # Take a step in the environment.
                # Choose the next action using epsilon-greedy policy (this is key to SARSA's on-policy nature).
                next_action = self.epsilon_greedy_policy(next_observation) # Choose the next action.
                current_state = str(observation)   # Convert the current state to string.
                next_state = str(next_observation)   # Convert the next state to string.
                current_q = self.q_table.loc[current_state, action]  # Current Q-value for state-action pair.
                next_q = self.q_table.loc[next_state, next_action]     # Next Q-value for the next state-action pair.
                
                # SARSA update: adjust Q(current_state, action) using the formula:
                # Q(s,a) = Q(s,a) + alpha * (reward + gamma * Q(s',a') - Q(s,a))
                self.q_table.loc[current_state, action] += self.lr * (reward + self.gamma * next_q - current_q) # Update Q-value.
                
                # Accumulate the updated Q-value into the cost metric for analysis.
                cost += self.q_table.loc[current_state, action] - current_q # Accumulate the cost.
                step += 1  # Increment the step count.

                # If the maximum number of steps is reached, end the episode.
                if step >= NUM_STEPS: # If the agent exceeds the maximum number of steps, end the episode.
                    if reward > 0: # Check if the reward is positive (reaching the goal).
                        positive_count += 1 # Increment the positive count.
                    else:
                        negative_count += 1 # Increment the negative count.
                    if reward == 1: # Check if the reward is 1 (reaching the goal).
                        goal_count += 1 # Increment the goal count.
                    rewards += reward  # Accumulate the reward.
                    break

                # If the episode has terminated (either reaching goal or hitting an obstacle), update counters.
                if done: # If the episode has terminated (either reaching goal or hitting an obstacle), update counters.
                    if reward > 0: # If the reward is positive (reaching the goal), increment the positive count.
                        positive_count += 1 
                    else:
                        negative_count += 1
                    if reward == 1:
                        goal_count += 1
                    rewards += reward  # Accumulate the reward.
                    break

                # Prepare for the next iteration: update state and action.
                observation = next_observation # Update the current state.
                action = next_action          # Update the current action.

            # Record episode metrics after each episode, regardless of how the episode ended.
            time_steps.append(step)        # Record the number of steps taken in the episode.
            MSE_Q.append(np.std(self.q_table.values))  # modified: record standard deviation of Q-values
            average_reward.append(rewards / (i + 1)) # Record the average reward per episode.

            print("Episode: {}".format(i))
            # Decay the exploration rate epsilon after each episode, ensuring it does not fall below a minimum threshold.
            self.epsilon = max(SARSA_MIN_EPSILON, self.epsilon * SARSA_EPSILON_DECAY)

        # After training is complete, print out some of the recorded Q values for analysis.
        print("Q_value:", Q_value)
        self.print_q_table()  # Print the full Q-table and state mappings.

        # Prepare data for a bar plot showing the number of positive and negative outcomes.
        all_cost_bar = [positive_count, negative_count]
        # Plot the results including time steps, MSE_Q, success rate, and average reward.
        self.plot_results(time_steps, MSE_Q, sucess_rate, all_cost_bar, average_reward)
        # Save the Q-table as a txt file in the current path
        self.q_table.to_csv("SARSA_Q_table_GRID_" + str(GRID_SIZE) + ".txt", sep='\t')
        # Return the trained Q-table and collected metrics.
        return self.q_table, time_steps, MSE_Q, sucess_rate, all_cost_bar, average_reward

    def plot_results(self, time_steps, cost, sucess_rate, all_cost_bar, average_reward):
        """
        Plot the training progress metrics:
          - Time steps per episode.
          - Cumulative cost per episode.
          - Success rate measured every 50 episodes.
          - Success vs. Failure counts.
          - Cumulative average rewards per episode.
        """
        # Create a figure with three subplots to visualize different metrics.
        # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
        # ax1.plot(np.arange(len(time_steps)), time_steps, 'b')
        # ax1.set_xlabel('Episode')
        # ax1.set_ylabel('Time Steps')
        # ax1.set_title('Time Steps vs Episode')
        
        # ax2.plot(np.arange(len(cost)), cost, 'b')
        # ax2.set_xlabel('Episode')
        # ax2.set_ylabel('Episode Cumulative Cost')
        # ax2.set_title('Episode Cumulative Cost vs Episode')
        
        # ax3.plot(np.arange(len(sucess_rate)), sucess_rate, 'b')
        # ax3.set_xlabel('Per 50 Episode')
        # ax3.set_ylabel('Success Rate')
        # ax3.set_title('Success Rate vs Per 50 Episode')
        
        # plt.tight_layout()  # Adjust layout for better spacing.
        
        # Create additional figures for each individual plot.
        plt.figure()
        plt.plot(np.arange(len(average_reward)), average_reward, color="blue", linewidth=1)
        plt.grid(True) 
        plt.title('Average Rewards vs Episode')
        plt.xlabel('Number of Episodes')
        plt.ylabel(' Average Rewards')
        
        plt.figure()
        plt.plot(np.arange(len(time_steps)), time_steps, color="dodgerblue", linewidth=1)
        plt.grid(True) 
        plt.title('Time Steps vs Episode')
        plt.xlabel('Number of Episodes')
        plt.ylabel('Time Steps')
    
        plt.figure()
        plt.plot(np.arange(len(sucess_rate)), sucess_rate, color="darkorange", linewidth=1)
        plt.grid(True) 
        plt.title('Success Rate per 50 Episodes')
        plt.xlabel('Batch (every 50 episodes)')
        plt.ylabel('Success Rate')
    
        plt.figure()
        plt.plot(np.arange(len(cost)), cost, color="hotpink", linewidth=1)
        plt.grid(True) 
        plt.title('MSE of Q values vs Episode')  # Modified for Q-value standard deviation curve
        plt.xlabel('Number of Episodes')
        plt.ylabel('MSE of Q values')  # Modified for Q-value standard deviation curve
    
        plt.figure()
        labels = ['Success', 'Failure']
        color_list = ['limegreen', 'crimson']
        plt.bar(np.arange(len(all_cost_bar)), all_cost_bar, tick_label=labels, color=color_list)
        plt.title('Success vs Failure')
        plt.ylabel('Number of Episodes')
    
        plt.show(block=False)

    def draw_best_route(self):
        """
        Using the trained Q-table, start from the initial state,
        follow the optimal policy to determine the best route.
        The route is then drawn in a GUI window.
        """
        self.env.reset()  # Reset the environment, placing the agent at the starting position.
        route = {}
        # Get the starting coordinates from the GUI element (agent's position).
        start_coords = self.env.Canvas_Widget.coords(self.env.agent)
        route[0] = start_coords
        # Convert the starting coordinates into a state using the environment's transformation.
        state = self.env.transformation(start_coords[0], start_coords[1])
        print("Starting state:", state, "Starting coordinates:", start_coords)
        # Loop for a maximum number of steps.
        for j in range(NUM_STEPS):
            # Choose the optimal action based on the current Q-table.
            action = self.optimal_policy(state)
            # Take a step in the environment using the chosen action.
            next_state, reward, done, info = self.env.step(action)
            self.env.render()  # Update the GUI display.
            # Record the new coordinates.
            coords = self.env.Canvas_Widget.coords(self.env.agent)
            route[j+1] = coords
            # Update the state based on the new coordinates.
            state = self.env.transformation(coords[0], coords[1])
            print(f"Step {j+1}: State {state}, Action {action}, Coordinates {coords}")
            if done:
                print("Reached the goal or encountered an obstacle, ending simulation.")
                break
        # Store the route in the environment and trigger its drawing.
        self.env.b = route
        self.env.route()
        try:
            self.env.update()
        except Exception as e:
            print("Environment update failed:", e)

# Entry point of the script.
if __name__ == '__main__':
    # Create an environment instance with the given grid size.
    env = Environment(grid_size=GRID_SIZE)
    # Initialize the SARSA agent with parameters from Parameters.py.
    agent = SARSA(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=SARSA_EPSILON)
    # Train the agent for a specified number of episodes.
    agent.train(num_epd=NUM_EPISODES)
    # Schedule the drawing of the best route after training.
    env.after(100, agent.draw_best_route)
    env.mainloop()  # Start the GUI event loop.
