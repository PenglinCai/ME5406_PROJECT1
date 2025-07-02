import numpy as np                   # Import numpy for numerical computations (e.g., random numbers, arrays)
import pandas as pd                  # Import pandas to use DataFrames for storing the Q-table
import matplotlib.pyplot as plt      # Import matplotlib for plotting training results

from Environment import Environment  # Import the custom Environment class (the simulation/maze environment)
from Parameters import *             # Import parameters (e.g., grid size, learning rate, epsilon, etc.)

np.random.seed(RANDOM_SEED)          # Set the random seed for reproducibility

# Define the Q_learning class which encapsulates the Q-learning algorithm
class Q_learning(object):
    def __init__(self, env, learning_rate, gamma, epsilon):
        self.env = env
        # Define the list of actions available in the environment.
        # Actions are represented by integers from 0 to (num_actions - 1)
        self.actions = list(range(self.env.num_actions)) # Initialized using env.num_actions
        
        # Store learning rate (alpha) for updating Q values
        self.lr = learning_rate # Initialized using LEARNING_RATE
        
        # Gamma is the discount factor to balance immediate and future rewards.
        self.gamma = gamma # Initialized using GAMMA
        
        # Epsilon for the epsilon-greedy policy: the probability of choosing a random action.
        self.epsilon = epsilon  # Initialized using QLEARNING_EPSILON
        
        # Pre-construct all possible states as strings.
        # The Q-table will have one row for each state (using a string label) and one column per action.
        all_states = [str(s) for s in range(self.env.num_states)]
        # Create a Q-table DataFrame with initial Q values set to 0.
        self.q_table = pd.DataFrame(0, index=all_states, columns=self.actions)
        # This table is used to record Q-values for each state-action pair.
        # An additional DataFrame is prepared to store Q-table snapshots if needed.
        self.q_table_final = pd.DataFrame(columns=self.actions)

    def state_to_label(self, state):
        """
        Convert a state number into a string representing the (x, y) coordinate.
        For example, in a 10×10 grid:
          - State 0 becomes "0_0"
          - State 14 becomes "4_1" (assuming numbering starts from the top-left corner)
        """
        x = state % GRID_SIZE  # Compute the x-coordinate by taking the remainder
        y = state // GRID_SIZE # Compute the y-coordinate by integer division
        return f"{x}_{y}"      # Return as a formatted string "x_y"

    def epsilon_greedy_policy(self, observation):
        # Given an observation (which is a state label in string format), decide the next action.
        # With probability epsilon, choose a random action to explore.
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            # Otherwise (with probability 1 - epsilon), exploit by choosing the best known action.
            # Look up the Q-values for the current state and select among those with maximum Q value.
            state_action = self.q_table.loc[observation, :]
            # In case multiple actions have the same maximum value, choose randomly among them.
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        return action

    def learn(self, state, action, reward, next_state): #Q Update Function
        # Update the Q-table using the Q-learning update rule.
        # Both 'state' and 'next_state' are provided as string representations.

        # q_predict is the current Q value for the state-action pair (i.e. Q(s, a))
        q_predict = self.q_table.loc[state, action] # Get the current Q value
        
        # Compute the target Q value:
        # q_target = reward + gamma * max_a' Q(next_state, a')
        # This is the reward received plus the discounted maximum future reward from the next state.
        q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()  # Compute the target Q value
        
        # Update the Q value for the state-action pair using the learning rate.
        # This is the core Q-learning update formula:
        # Q(s, a) = Q(s, a) + learning_rate * (q_target - Q(s, a))
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict) # Update the Q value
        
        # Return the updated Q value (useful for tracking cumulative cost)
        return self.q_table.loc[state, action] # Return the updated Q value

    def train(self, num_epd):
        # Initialize lists to record training statistics over episodes:
        time_steps = []                 # Records the number of steps taken per episode
        MSE_Q = []                      # Records the cumulative cost (sum of Q updates) per episode (modified from episode_cumulative_cost)
        sucess_rate = []                # Records the success rate every 50 episodes
        average_reward = []             # Records the running average reward per episode
        Q_value = {}                    # Dictionary to snapshot specific Q-values during training
        goal_count = 0                  # Counter for how many times the goal was reached in a batch of episodes
        rewards = 0                     # Cumulative reward across episodes
        positive_count = 0              # Count of episodes with positive reward (successful episodes)
        negative_count = 0              # Count of episodes with negative reward (failures or penalties)

        # Loop over the number of training episodes
        for i in range(num_epd): 
            # Reset the environment for a new episode; observation is an integer state number.
            observation = self.env.reset() # Reset the environment and get the initial state
            step = 0        # Reset step count for the current episode
            cost = 0        # Reset cumulative cost for the current episode

            # Every 50 episodes, calculate the success rate (goal_count/50)
            if i != 0 and i % 50 == 0:
                goal_count = goal_count / 50
                sucess_rate += [goal_count]
                goal_count = 0  # Reset goal count for the next batch

            # Every 1000 episodes, record the Q values for a specific state (state "14") for analysis.
            if i != 0 and i % 1000 == 0:
                Q_value[i] = []
                for j in range(self.env.num_actions):
                    Q_value[i].append(self.q_table.loc[str(14), j])

            # Inner loop: run steps within the current episode
            while True:
                # Choose an action using the epsilon-greedy policy.
                # Note: 'observation' is converted to a string since the Q-table uses string indices.
                action = self.epsilon_greedy_policy(str(observation)) 
                
                # Execute the chosen action in the environment.
                # env.step() returns:
                #   obs: next state (as an integer)
                #   reward: reward received for the action
                #   done: whether the episode is finished (goal reached or failure)
                #   info: additional info (if any)
                obs, reward, done, info = self.env.step(action)
                
                # Learn/update the Q-table using the current transition.
                # The returned value is added to the cumulative cost.
                cost += self.learn(str(observation), action, reward, str(obs))
                
                # Update current observation to the next observation.
                observation = obs
                
                # Increment the step counter for this episode.
                step += 1
                
                # If the step count exceeds the maximum allowed steps, record stats and break the inner loop.
                if step >= NUM_STEPS: # Modified for the maximum number of steps
                    if reward > 0:
                        positive_count += 1
                    else:
                        negative_count += 1
                    time_steps += [step]
                    MSE_Q += [np.std(self.q_table.values)]
                    if reward == 1:
                        goal_count += 1
                    rewards += reward
                    average_reward += [rewards / (i + 1)]
                    break
                
                # If the episode terminates (goal reached or failure):
                if done:
                    # Count how many episodes received a positive reward (goal reached)
                    if reward > 0: 
                        positive_count += 1
                    else:
                        negative_count += 1
                    # Record the number of steps taken in this episode.
                    time_steps += [step] # Record the number of steps taken in this episode.
                    # Record the cumulative cost accumulated over the episode (modified: record standard deviation of Q-values)
                    MSE_Q += [np.std(self.q_table.values)]
                    # Increment goal count if the reward signifies reaching the goal (reward equals 1).
                    if reward == 1:
                        goal_count += 1
                    # Add reward to the cumulative reward sum.
                    rewards += reward
                    # Record the running average reward (cumulative reward divided by the number of episodes so far)
                    average_reward += [rewards / (i + 1)]
                    break

            # After finishing an episode, decay epsilon so that the agent gradually shifts from exploration to exploitation.
            # The new epsilon is the larger value between the decayed epsilon and the minimum allowed epsilon.
            self.epsilon = max(QLEARNING_MIN_EPSILON, self.epsilon * QLEARNING_EPSILON_DECAY)
            # Print out the current episode number to track progress.
            print('Episode: {}'.format(i))

            # Optionally, you can also record MSE_Q here per episode if not already recorded inside the loop.
            # (In this code, MSE_Q is recorded only when the episode ends with a terminal state.)
        # After training, print any recorded snapshots of Q values.
        print("Q_value: {}".format(Q_value))
        # Print the full Q-table and a mapping of state indices to coordinates.
        self.print_q_table()
        # Plot various training results such as time steps, MSE_Q, success rate, and average reward.
        self.plot_results(time_steps, MSE_Q, sucess_rate, [positive_count, negative_count], average_reward)
        # Save the final Q-table to a text file for later analysis.
        self.q_table.to_csv("Q_learning_Q_table_GRID_" + str(GRID_SIZE) + ".txt", sep='\t')
        # Return various training statistics.
        return self.q_table, time_steps, MSE_Q, sucess_rate, [positive_count, negative_count], average_reward

    def print_q_table(self):
        # Print the full Q-table to the console.
        print("Full Q-table:")
        print(self.q_table)
        # Create a mapping from state index (as string) to (x, y) coordinates for easier interpretation.
        mapping = {str(s): self.state_to_label(s) for s in range(self.env.num_states)}
        print("State mapping (state index: coordinate):")
        # Print out each mapping in the format "state: coordinate".
        for key, value in mapping.items():
            print(f"{key}: {value}")

    def plot_results(self, time_steps, cost, sucess_rate, all_cost_bar, average_reward):
        # Create multiple plots to visualize training results.
        # The first plot: time steps vs. episode index.
        # b, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
        # ax1.plot(np.arange(len(time_steps)), time_steps, 'b')
        # ax1.set_xlabel('Episode')
        # ax1.set_ylabel('Time Steps')
        # ax1.set_title('Time Steps vs Episode')
        
        # # The second plot: episode cumulative cost vs. episode index.
        # ax2.plot(np.arange(len(cost)), cost, 'b')
        # ax2.set_xlabel('Episode')
        # ax2.set_ylabel('Episode Cumulative Cost')
        # ax2.set_title('Episode Cumulative Cost vs Episode')
        
        # # The third plot: success rate per 50 episodes.
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
        Using the trained Q-table, start from the initial state and always choose the action with the highest Q value.
        This extracts the best route and draws it in the GUI.
        """
        # Reset the environment to start from the beginning.
        self.env.reset()  
        route = {}
        # Record the starting pixel coordinates of the agent.
        start_coords = self.env.Canvas_Widget.coords(self.env.agent)
        route[0] = start_coords
        # Convert starting coordinates into a state (using the environment's transformation)
        state = self.env.transformation(start_coords[0], start_coords[1])
        print("Starting state:", state, "Starting coordinates:", start_coords)
        # Loop for a maximum number of steps to follow the best route.
        for j in range(NUM_STEPS):
            # Select the action with the highest Q value for the current state.
            action = int(self.q_table.loc[str(state), :].idxmax())
            # Take the selected action.
            next_state, reward, done, info = self.env.step(action)
            self.env.render()  # Update the GUI to reflect the new state.
            coords = self.env.Canvas_Widget.coords(self.env.agent)
            route[j+1] = coords
            # Update the state based on the new coordinates.
            state = self.env.transformation(coords[0], coords[1])
            print(f"Step {j+1}: State {state}, Action {action}, Coordinates {coords}")
            if done:
                print("Reached the goal or encountered an obstacle, ending simulation.")
                break
        
        # Assign the computed route to the environment and plot it.
        self.env.b = route
        self.env.route()
        self.env.update()

# Main execution block: Create the environment, initialize the Q-learning agent, and run training.
if __name__ == '__main__':
    # Instantiate the environment with the given grid size.
    env = Environment(grid_size=GRID_SIZE)
    # Create a Q-learning agent using the environment and parameters for learning rate, gamma, and epsilon.
    Q_learning_agent = Q_learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=QLEARNING_EPSILON)
    # Train the agent for a specified number of episodes.
    Q_table = Q_learning_agent.train(num_epd=NUM_EPISODES)
    # Schedule the drawing of the best route in the main GUI loop after a short delay.
    env.after(100, Q_learning_agent.draw_best_route)
    # Start the GUI main loop.
    env.mainloop()
