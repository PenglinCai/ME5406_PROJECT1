# Environment parameters

RANDOM_SEED   = 31      # Random seed for reproducibility
PIXELS        = 40          # Pixel size for each grid cell
GRID_SIZE     = 10     # Grid size of the environment; valid values are 4 (fixed obstacles) or 10 (random obstacles with 25% density)
GRID_HEIGHT    = GRID_SIZE   # Height of the grid (number of rows)
GRID_WIDTH     = GRID_SIZE   # Width of the grid (number of columns)

# Training parameters

NUM_STEPS     = 2000         # Maximum number of steps allowed per episode
NUM_EPISODES  =20000      # Total number of training episodes
LEARNING_RATE = 0.01        # Learning rate used in Q-value updates (SARSA and Q-learning)
GAMMA         = 0.99         # Discount factor for future rewards

# For Monte Carlo
MONTE_CARLO_EPSILON        = 0.5   # Initial exploration rate for Monte Carlo
MONTE_CARLO_MIN_EPSILON    = 0.01   # Minimum epsilon for Monte Carlo
MONTE_CARLO_EPSILON_DECAY  = 0.999  # Epsilon decay factor for Monte Carlo

# For SARSA
SARSA_EPSILON        = 0.4   # Initial exploration rate for SARSA
SARSA_MIN_EPSILON    = 0.01  # Minimum epsilon for SARSA
SARSA_EPSILON_DECAY  = 0.9 # Epsilon decay factor for SARSA

# For Q-learning
QLEARNING_EPSILON        = 0.4   # Initial exploration rate for Q-learning
QLEARNING_MIN_EPSILON    = 0.01  # Minimum epsilon for Q-learning
QLEARNING_EPSILON_DECAY  = 0.9 # Epsilon decay factor for Q-learning
