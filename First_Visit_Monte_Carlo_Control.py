# Import standard libraries for randomness, numerical computation, data handling, plotting, and math functions
import random                         # Used for random number generation (e.g., epsilon-greedy decisions)
import numpy as np                    # For numerical operations and handling arrays
import pandas as pd                   # For creating and manipulating dataframes (used for Q-table and bookkeeping)
import matplotlib.pyplot as plt       # For plotting results and visualizing training performance
import math                           # For mathematical functions (e.g., floor)

# Import custom modules that define the environment and parameters
from Environment import Environment   # The environment in which the agent operates (e.g., grid world)
from Parameters import *              # Import all parameters such as GRID_SIZE, MONTE_CARLO_EPSILON, etc.

# Set the seed for NumPy's random number generator to ensure reproducibility
np.random.seed(RANDOM_SEED)

# Define the MonteCarlo class for implementing First-Visit Monte Carlo Control
class MonteCarlo(object):
    def __init__(self, env, epsilon, gamma):
        """
        Initialize the First-Visit Monte Carlo Control agent.
        :param env: The environment object
        :param epsilon: Initial epsilon for epsilon-greedy policy (exploration rate)
        :param gamma: Discount factor for future rewards
        """
        self.env = env                           # Save the environment instance
        self.n_s = self.env.num_states           # Number of states in the environment
        self.n_a = self.env.num_actions          # Number of actions available in the environment

        self.epsilon = epsilon                   # Initialize epsilon (exploration probability)
        self.gamma = gamma                       # Discount factor for future rewards

        # Lists for storing statistics during training
        self.time_steps = []                     # Time steps taken per episode
        self.MSE_Q = []                          # Cumulative cost (or value) for each episode (modified from episode_cumulative_cost)
        self.sucess_rate = []                    # Success rate (e.g., goal-reaching rate) per 50 episodes
        self.average_reward = []                 # Average reward per episode
        self.rewards = 0                         # Accumulated reward across episodes
        self.positive_count = 0                  # Count of episodes with positive reward
        self.negative_count = 0                  # Count of episodes with negative reward
        self.goal_count = 0                      # Count of episodes where the goal is reached

        # Create Q-table, total return, and visit count tables as Pandas DataFrames.
        # States are represented as strings to maintain consistency with Q-learning implementations.
        all_states = [str(s) for s in range(self.env.num_states)]
        # Q-table: Each row is a state and each column is an action; initialized to 0.0
        self.q_table = pd.DataFrame(0.0, index=all_states, columns=list(range(self.n_a)))
        # total_return: To accumulate returns (G) for state-action pairs over episodes
        self.total_return = pd.DataFrame(0.0, index=all_states, columns=list(range(self.n_a)))
        # visit_count: To count how many times each state-action pair has been encountered (first visits)
        self.visit_count = pd.DataFrame(0, index=all_states, columns=list(range(self.n_a)))

    def epsilon_greedy_policy(self, observation):
        """
        Choose an action based on an epsilon-greedy policy.
        :param observation: The current state (as a string)
        :return: An action index
        """
        # With probability epsilon, choose a random action (exploration)
        if random.uniform(0, 1) < self.epsilon: # Randomly select a number between 0 and 1
            return np.random.randint(0, self.n_a) # Randomly select an action index
        else:
            # Otherwise, choose the action with the highest Q-value (exploitation)
            state_actions = self.q_table.loc[observation, :] # Get Q-values for the current state
            # Handle tie-breaking by randomly selecting among the best actions
            max_actions = state_actions[state_actions == state_actions.max()].index # Find the best action(s)
            return np.random.choice(max_actions) # Randomly select among the best actions

    def optimal_policy(self, observation):
        """
        Choose the optimal action (greedy) given the current state.
        :param observation: The current state (as a string)
        :return: The action with the highest Q-value
        """
        state_actions = self.q_table.loc[observation, :] # Get Q-values for the current state
        max_actions = state_actions[state_actions == state_actions.max()].index # Find the best action(s)
        return np.random.choice(max_actions) # Randomly select among the best actions

    def generate_episode(self):
        """
        Generate an episode (a sequence of state, action, reward tuples) by interacting with the environment.
        :return: A list representing the episode
        """
        episode = []                                # List to store the sequence of (state, action, reward)
        observation = str(self.env.reset())          # Reset environment to get the starting state; cast state to string
        time_steps = 0                              # Initialize step counter for the episode

        # Loop for a maximum number of steps (NUM_STEPS)
        for t in range(NUM_STEPS):                 # Loop for a maximum number of steps
            # Choose an action using the epsilon-greedy policy based on the current state
            action = self.epsilon_greedy_policy(observation) # Choose an action using epsilon-greedy policy
            # Execute the chosen action in the environment
            next_observation, reward, done, info = self.env.step(action) # Get the next state, reward, and done flag
            next_observation = str(next_observation) # Convert next state to string for consistency
            # Record the transition in the episode history
            episode.append((observation, action, reward)) # Append the tuple (state, action, reward) to the episode
            time_steps += 1                        # Increment the time step counter

            # Check if the episode is finished (terminal state reached or other stopping condition)
            if done:
                # Update counters based on reward received at termination
                if reward > 0:                    # Check if the reward is positive
                    self.positive_count += 1      # Count episodes with positive reward
                    self.goal_count += 1         # Count successful episodes (e.g., reaching goal)
                else:
                    self.negative_count += 1    # Count episodes with negative reward
                self.time_steps.append(time_steps)  # Store total time steps taken in this episode
                self.rewards += reward              # Accumulate reward from this episode
                
                break                              # Exit the loop if done

            # Update observation to the next state for the next iteration of the loop
            observation = next_observation        # Update the current state to the next state
        else:
            if reward > 0:                   # Check if the reward is positive
                self.positive_count += 1    # Count episodes with positive reward
                self.goal_count += 1         # Count successful episodes (e.g., reaching goal)
            else:
                self.negative_count += 1   # Count episodes with negative reward
            self.time_steps.append(time_steps)  # Store total time steps taken in this episode
            self.rewards += reward              # Accumulate reward from this episode

        # Return the complete episode as a list of tuples (state, action, reward)
        return episode

    def first_visit_monte_carlo_control(self, num_eps): # Modified to return Q-table and additional statistics
        """
        Perform First-Visit Monte Carlo Control over a number of episodes (num_eps) to learn the optimal Q-values.
        :param num_eps: Number of episodes to train on
        :return: Tuple of Q-table and various performance metrics
        """
        # Loop over each training episode
        for i in range(num_eps): 
            cost = 0                       # Initialize the cumulative cost for this episode
            episode = self.generate_episode()  # Generate an episode using current policy

            # A set to keep track of the first visit of state-action pairs in this episode
            visited = set()               # Initialize an empty set to store visited state-action pairs

            G = 0                          # Initialize return G, which will be computed by backtracking
            # Every 50 episodes, compute and record the success rate over those episodes
            if i != 0 and i % 50 == 0:
                self.goal_count = self.goal_count / 50
                self.sucess_rate.append(self.goal_count)
                self.goal_count = 0         # Reset the goal counter after recording

            # Record average reward so far across episodes
            self.average_reward.append(self.rewards / (i + 1))  # Average reward per episode

            # --- Begin Monte Carlo Return Calculation ---
            # The key Monte Carlo computation is done by iterating backward through the episode.
            # This computes the return G for each state-action pair encountered in the episode.
            for observation, action, reward in reversed(episode): # Iterate backward through the episode
                # Update the return G by adding the current reward and discounting the previous return
                G = reward + self.gamma * G # Discounted return for this state-action pair
                # First-visit check: update only if this is the first time the state-action pair was encountered in this episode
                if (observation, action) not in visited:  # Check if this state-action pair was visited earlier
                    # Accumulate the total return for this state-action pair
                    self.total_return.loc[observation, action] += G 
                    # Increment the visit count for this state-action pair
                    self.visit_count.loc[observation, action] += 1  
                    # Update the Q-value as the average return for this state-action pair
                    self.q_table.loc[observation, action] = (
                        self.total_return.loc[observation, action] / self.visit_count.loc[observation, action]
                    )
                    # Mark this state-action pair as visited for the current episode
                    visited.add((observation, action)) # Add to the set of visited state-action pairs
                # Accumulate cost using the updated Q-value (can be used for further analysis or plotting)
                cost += self.q_table.loc[observation, action]
            # --- End Monte Carlo Return Calculation ---

            # Append the cumulative cost of the episode to the performance tracking list (modified: record standard deviation of Q-values)
            self.MSE_Q.append(np.std(self.q_table.values))  # Modified to record standard deviation of Q-values

            # After each episode, decay epsilon to gradually reduce exploration, but ensure it does not fall below a minimum value
            self.epsilon = max(MONTE_CARLO_MIN_EPSILON, self.epsilon * MONTE_CARLO_EPSILON_DECAY)
            print("Episode: {}".format(i))

        # After training, plot the performance metrics such as time steps, MSE_Q, success rate, and rewards
        self.plot_results(self.time_steps, self.MSE_Q, self.sucess_rate,
                          [self.positive_count, self.negative_count], self.average_reward)
        # Save the Q-table as a txt file in the current path
        self.q_table.to_csv("MonteCarlo_Q_table_GRID_" + str(GRID_SIZE) + ".txt", sep='\t')
        # Return the final Q-table and other statistics for potential further analysis
        return self.q_table, self.time_steps, self.MSE_Q, self.sucess_rate, \
               [self.positive_count, self.negative_count], self.average_reward

    def plot_results(self, time_steps, cost, sucess_rate, episode_cumulative_cost_bar, average_reward):
        """
        Plot various performance metrics for the agent.
        :param time_steps: List of time steps per episode
        :param cost: Cumulative cost per episode (here MSE_Q)
        :param sucess_rate: Success rate computed every 50 episodes
        :param episode_cumulative_cost_bar: Bar chart data for success vs failure counts
        :param average_reward: Average rewards per episode
        """
        # Create a subplot with three graphs in one row
        # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
        # # Plot the number of time steps per episode
        # ax1.plot(np.arange(len(time_steps)), time_steps, 'b')
        # ax1.set_xlabel('Episode')
        # ax1.set_ylabel('Time Steps')
        # ax1.set_title('Time Steps vs Episode')
        
        # # Plot the cumulative cost per episode
        # ax2.plot(np.arange(len(cost)), cost, 'b')
        # ax2.set_xlabel('Episode')
        # ax2.set_ylabel('Episode Cumulative Cost')
        # ax2.set_title('Episode Cumulative Cost vs Episode')
        
        # # Plot the success rate (per 50 episodes)
        # ax3.plot(np.arange(len(sucess_rate)), sucess_rate, 'b')
        # ax3.set_xlabel('Per 50 Episode')
        # ax3.set_ylabel('Sucess Rate')
        # ax3.set_title('Sucess Rate vs Per 50 Episode')
        
        # plt.tight_layout()
        # # Generate additional figures for detailed visualization
        plt.figure()
        plt.plot(np.arange(len(average_reward)), average_reward, color="blue", linewidth=1)
        plt.grid(True) 
        plt.title('Average Rewards vs Episode')
        plt.xlabel('Number of Episodes')
        plt.ylabel('Average Rewards')
        
        
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
        plt.ylabel('Sucess Rate')
        
        plt.figure()
        plt.plot(np.arange(len(cost)), cost, color="hotpink", linewidth=1)
        plt.grid(True) 
        plt.title('MSE of Q values vs Episode')  # Modified for Q-value standard deviation curve
        plt.xlabel('Number of Episodes')
        plt.ylabel('MSE of Q values')  # Modified for Q-value standard deviation curve

        plt.figure()
        labels = ['Success', 'Failure']
        color_list = ['limegreen', 'crimson']
        plt.bar(np.arange(len(episode_cumulative_cost_bar)), episode_cumulative_cost_bar,
                tick_label=labels, color=color_list)
        plt.title('Success vs Failure')
        plt.ylabel('Number of Episodes')
        plt.show(block=False)
        

    def display_optimal_route(self):
        """
        Using the learned Q-table, start at the initial state and always choose the action with the highest Q-value.
        This method extracts the optimal route and displays it in the GUI window.
        """
        self.env.reset()  # Reset the environment to place the agent at the starting position
        route = {}      # Dictionary to store the coordinates of the route at each time step
        # Get the starting pixel coordinates from the GUI canvas
        start_coords = self.env.Canvas_Widget.coords(self.env.agent)
        route[0] = start_coords
        # Convert the starting coordinates to a state index using a transformation function
        state = self.env.transformation(start_coords[0], start_coords[1])
        print("Starting state:", state, "Starting coordinates:", start_coords)
        
        # Iterate for a maximum number of steps to construct the route
        for j in range(NUM_STEPS):
            # Choose the action with the highest Q-value for the current state
            action = int(self.q_table.loc[str(state), :].idxmax())
            next_state, reward, done, info = self.env.step(action)
            self.env.render()  # Refresh the GUI to show the agent's movement
            coords = self.env.Canvas_Widget.coords(self.env.agent)
            route[j+1] = coords
            # Update the current state based on new coordinates
            state = self.env.transformation(coords[0], coords[1])
            print(f"Step {j+1}: State {state}, Action {action}, Coordinates {coords}")
            if done:
                print("Reached the goal or encountered an obstacle, ending simulation.")
                break
        
        # Save the route in the environment and draw it on the GUI
        self.env.b = route
        self.env.route()
        try:
            self.env.update()
        except Exception as e:
            print("Error updating the window:", e)


# Main execution block: this runs when the script is executed directly
if __name__ == '__main__':
    # Create an instance of the Environment with a specified grid size
    env = Environment(grid_size=GRID_SIZE)
    # Initialize the Monte Carlo agent with the environment, starting epsilon, and discount factor gamma
    monte_carlo = MonteCarlo(env, epsilon=MONTE_CARLO_EPSILON, gamma=GAMMA)
    # Run the first-visit Monte Carlo control for a specified number of episodes (NUM_EPISODES)
    q_table, time_steps, MSE_Q, sucess_rate, cost_bar, average_reward = monte_carlo.first_visit_monte_carlo_control(num_eps=NUM_EPISODES)
    # Schedule the display of the optimal route using Tkinter's 'after' method (ensuring it runs in the main loop)
    env.after(100, monte_carlo.display_optimal_route)
    # Start the Tkinter main loop to run the GUI
    env.mainloop()
