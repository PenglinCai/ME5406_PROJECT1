import random
import numpy as np
import tkinter as tk
import time

from PIL import Image, ImageTk
from Parameters import PIXELS, RANDOM_SEED  # Retain the imports for PIXELS and RANDOM_SEED

class Environment(tk.Tk, object):
    def __init__(self, grid_size):
        # Initialize random seeds for reproducibility.
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
        # Set the grid size for dynamic environment layout.
        self.grid_size = grid_size
        self.grid_width = grid_size
        self.grid_height = grid_size

        super(Environment, self).__init__()
        
        # Define the possible actions and compute the total number of states based on grid dimensions.
        self.action_space = ['left', 'right', 'up', 'down']
        self.num_actions = len(self.action_space)
        self.num_states = self.grid_width * self.grid_height

        self.title('Project 1: Frozen Lake')
        self.geometry('{}x{}'.format(self.grid_width * PIXELS, self.grid_height * PIXELS))

        # Dictionaries and variables for recording the agent's path:
        #   self.a: Records the pixel coordinates of the agent at every step.
        #   self.b: Stores the first successful path (or the best route) for later analysis or drawing.
        #   self.c: Intended for storing the optimal route for further processing.
        #   self.d: A step counter to index each movement.
        #   self.e: A flag indicating whether the first successful path has been recorded.
        #   self.leastoptimal: Tracks the longest path length encountered.
        #   self.optimal: Tracks the shortest (optimal) path length encountered.
        self.a = {}
        self.b = {}
        self.c = {}
        self.d = 0
        self.e = True
        self.leastoptimal = 0
        self.optimal = 0

        # Lists to hold the positions of obstacles (holes) and the goal.
        self.holes_positions = []
        self.goal_position = None

        # Create the environment based on the grid size.
        self.create_environments()

    def create_environments(self):
        """
        Creates the environment layout based on the grid size.
        For a grid size of 4, fixed obstacle positions are used.
        For a grid size of 10, obstacles are placed randomly with a fixed density.
        """
        if self.grid_size == 4:
            self.generate_4x4_environment()
            print('Created 4x4 environment.')
        elif self.grid_size == 10:
            self.generate_10x10_environment()
            print('Created 10x10 environment.')
        else:
            print("Please input the correct map size (4 or 10)")

    def generate_4x4_environment(self):
        """
        Sets up a 4x4 environment with a canvas:
          - Draws grid lines.
          - Places holes (obstacles) at fixed grid cells.
          - Places the goal at a fixed position.
          - Places the robot (agent) at the starting position (top-left corner).
          - Records the pixel coordinates of holes and the goal for later collision and goal detection.
        """
        # Create the canvas for drawing.
        self.Canvas_Widget = tk.Canvas(self, bg='white',
                                       height=self.grid_height * PIXELS,
                                       width=self.grid_width * PIXELS)
        # Draw vertical grid lines.
        for column in range(0, self.grid_width * PIXELS, PIXELS):
            self.Canvas_Widget.create_line(column, 0, column, self.grid_height * PIXELS, fill='grey', tags='grid_line')
        # Draw horizontal grid lines.
        for row in range(0, self.grid_height * PIXELS, PIXELS):
            self.Canvas_Widget.create_line(0, row, self.grid_width * PIXELS, row, fill='grey', tags='grid_line')

        # Load the hole image and place holes at predetermined positions.
        img_hole1 = Image.open("images for GUI/hole.png")
        self.hole1_object = ImageTk.PhotoImage(img_hole1)

        # Place hole images at specific grid cell positions (by multiplying grid index with PIXELS).
        self.hole1 = self.Canvas_Widget.create_image(PIXELS * 0, PIXELS * 3, anchor='nw', image=self.hole1_object)
        self.hole2 = self.Canvas_Widget.create_image(PIXELS * 1, PIXELS * 1, anchor='nw', image=self.hole1_object)
        self.hole3 = self.Canvas_Widget.create_image(PIXELS * 3, PIXELS * 1, anchor='nw', image=self.hole1_object)
        self.hole4 = self.Canvas_Widget.create_image(PIXELS * 3, PIXELS * 2, anchor='nw', image=self.hole1_object)

        # Load the goal image and place it at a fixed position.
        img_goal = Image.open("images for GUI/goal.png")
        self.goal_object = ImageTk.PhotoImage(img_goal)
        self.goal = self.Canvas_Widget.create_image(PIXELS * 3, PIXELS * 3, anchor='nw', image=self.goal_object)

        # Load the robot image and place the agent at the starting position.
        img_robot = Image.open("images for GUI/robot.png")
        self.robot = ImageTk.PhotoImage(img_robot)
        self.agent = self.Canvas_Widget.create_image(0, 0, anchor='nw', image=self.robot)

        self.Canvas_Widget.tag_raise('grid_line')
        self.Canvas_Widget.pack()

        # Record the pixel coordinates of each hole and the goal for collision and goal detection.
        self.holes_positions = [self.Canvas_Widget.coords(self.hole1),
                                self.Canvas_Widget.coords(self.hole2),
                                self.Canvas_Widget.coords(self.hole3),
                                self.Canvas_Widget.coords(self.hole4)]
        self.goal_position = self.Canvas_Widget.coords(self.goal)

    def generate_10x10_environment(self):
        """
        Sets up a 10x10 environment with a canvas:
          - Draws grid lines.
          - Randomly places 25 obstacles (holes) at unique grid cells excluding the start and goal.
          - Places the goal at the bottom-right corner.
          - Places the robot (agent) at the starting position.
          - Records the pixel coordinate of the goal.
        """
        # Create the canvas.
        self.Canvas_Widget = tk.Canvas(self, bg='white',
                                       height=self.grid_height * PIXELS,
                                       width=self.grid_width * PIXELS)
        # Draw grid lines.
        for column in range(0, self.grid_width * PIXELS, PIXELS):
            self.Canvas_Widget.create_line(column, 0, column, self.grid_height * PIXELS, fill='grey', tags='grid_line')
        for row in range(0, self.grid_height * PIXELS, PIXELS):
            self.Canvas_Widget.create_line(0, row, self.grid_width * PIXELS, row, fill='grey', tags='grid_line')

        # Load the hole image.
        img_hole1 = Image.open("images for GUI/hole.png")
        self.hole1_object = ImageTk.PhotoImage(img_hole1)

        # Generate a list of possible positions excluding the start (0,0) and goal (bottom-right).
        possible_positions = [(col, row) for col in range(self.grid_width) for row in range(self.grid_height)
                              if (col, row) not in [(0, 0), (self.grid_width-1, self.grid_height-1)]]
        # Randomly select 25 unique positions for holes.
        hole_indices = np.random.choice(len(possible_positions), 25, replace=False)
        self.hole_positions_grid = [possible_positions[i] for i in hole_indices]

        self.holes_positions = []
        # Place each hole on the canvas and record its pixel coordinates.
        for (col, row) in self.hole_positions_grid:
            x = col * PIXELS
            y = row * PIXELS
            obs_id = self.Canvas_Widget.create_image(x, y, anchor='nw', image=self.hole1_object)
            self.holes_positions.append(self.Canvas_Widget.coords(obs_id))

        # Load and place the goal image at the bottom-right corner.
        img_goal = Image.open("images for GUI/goal.png")
        self.goal_object = ImageTk.PhotoImage(img_goal)
        self.goal = self.Canvas_Widget.create_image((self.grid_width - 1) * PIXELS, (self.grid_height - 1) * PIXELS,
                                                     anchor='nw', image=self.goal_object)

        # Load and place the robot image at the starting position.
        img_robot = Image.open("images for GUI/robot.png")
        self.robot = ImageTk.PhotoImage(img_robot)
        self.agent = self.Canvas_Widget.create_image(0, 0, anchor='nw', image=self.robot)

        self.Canvas_Widget.tag_raise('grid_line')
        self.Canvas_Widget.pack()
        self.goal_position = self.Canvas_Widget.coords(self.goal)

    def reset(self):
        """
        Resets the environment:
          - Refreshes the GUI.
          - Resets the agent to the starting position (top-left corner).
          - Clears the recorded path and step counter.
          - Returns the initial state as a unique index after converting from pixel coordinates.
        """
        self.update()
        self.Canvas_Widget.delete(self.agent)
        self.agent = self.Canvas_Widget.create_image(0, 0, anchor='nw', image=self.robot)
        self.Canvas_Widget.tag_raise('grid_line')
        self.a = {}
        self.d = 0
        s = self.Canvas_Widget.coords(self.agent)
        s = self.transformation(s[0], s[1])
        return s

    def step(self, action):
        """
        Executes one time step within the environment given an action.
        
        Calculation Process:
          1. Retrieve the current pixel coordinates of the agent.
          2. Initialize a two-element array (agent_actions) to record the change in position.
          3. For the given action (0: left, 1: right, 2: up, 3: down), compute the pixel offset:
              - For left (action==0): subtract PIXELS from the x-coordinate if not at the left edge.
              - For right (action==1): add PIXELS to the x-coordinate if not at the right edge.
              - For up (action==2): subtract PIXELS from the y-coordinate if not at the top edge.
              - For down (action==3): add PIXELS to the y-coordinate if not at the bottom edge.
          4. Move the agent image by the computed offset.
          5. Record the new pixel coordinates in the path dictionary (self.a) and update the step counter.
          6. Check for terminal states:
              - If the new position equals the goal position, assign a reward of +1 and mark the episode as done.
                Also, update the recorded path if it is the first or a shorter successful route.
              - If the new position is in the list of hole positions, assign a reward of -1, mark the episode as done, and reset the path tracking.
              - Otherwise, assign a reward of 0 and mark the episode as not done.
          7. Convert the new pixel coordinates into a unique state index using the transformation function.
          8. Return the new state, reward, done flag, and an empty info dictionary.
        """
        # Get current pixel coordinates of the agent.
        state = self.Canvas_Widget.coords(self.agent)
        agent_actions = np.array([0, 0])
        
        # Determine pixel movement based on action.
        if action == 0:  # left
            if state[0] >= PIXELS:
                agent_actions[0] -= PIXELS
        elif action == 1:  # right
            if state[0] < (self.grid_width - 1) * PIXELS:
                agent_actions[0] += PIXELS
        elif action == 2:  # up
            if state[1] >= PIXELS:
                agent_actions[1] -= PIXELS
        elif action == 3:  # down
            if state[1] < (self.grid_height - 1) * PIXELS:
                agent_actions[1] += PIXELS

        # Move the agent image by the computed pixel offset.
        self.Canvas_Widget.move(self.agent, agent_actions[0], agent_actions[1])
        self.Canvas_Widget.tag_raise('grid_line')
        # Record the new pixel coordinates in the path dictionary.
        self.a[self.d] = self.Canvas_Widget.coords(self.agent)
        next_state = self.a[self.d]
        self.d += 1

        # Check if the agent has reached the goal.
        if next_state == self.goal_position:
            reward = 1
            done = True
            # Record the first successful path or update if a new shorter path is found.
            if self.e:
                for j in range(len(self.a)):
                    self.b[j] = self.a[j]
                self.e = False
                self.leastoptimal = len(self.a)
                self.optimal = len(self.a)
            if len(self.a) < len(self.b):
                self.optimal = len(self.a)
                self.b = {}
                for j in range(len(self.a)):
                    self.b[j] = self.a[j]
            if len(self.a) > self.leastoptimal:
                self.leastoptimal = len(self.a)
        # Check if the agent has stepped into a hole (obstacle).
        elif next_state in self.holes_positions:
            reward = -1
            done = True
            # Reset the path tracking upon hitting an obstacle.
            self.a = {}
            self.d = 0
        else:
            reward = 0
            done = False

        # Convert the new pixel coordinates to a state index.
        next_state = self.transformation(next_state[0], next_state[1])
        return next_state, reward, done, {}

    def render(self):
        """
        Pauses briefly to allow visualization and then refreshes the GUI window.
        """
        time.sleep(0.05)
        self.update()

    def route(self):
        """
        Draws the optimal route on the canvas:
          1. Converts the stored optimal path's pixel coordinates into offset points.
          2. Draws connecting lines between these points, adding an arrow at the final segment to indicate direction.
          3. Resets the robot image to the starting point for clear visualization of the path.
        """
        print('The optimal route:', self.optimal)
        print('The leastoptimal route:', self.leastoptimal)
        origin = np.array([20, 20])
        route_points = []
        # Convert each recorded pixel coordinate in the optimal path to an offset point.
        for j in range(len(self.b)):
            x = self.b[j][0] + origin[0]
            y = self.b[j][1] + origin[1]
            route_points.append((x, y))
        # Draw the route by connecting consecutive points.
        if len(route_points) >= 2:
            for i in range(len(route_points) - 1):
                if i == len(route_points) - 2:
                    self.Canvas_Widget.create_line(
                        route_points[i][0], route_points[i][1],
                        route_points[i+1][0], route_points[i+1][1],
                        arrow=tk.LAST, fill='green', width=2
                    )
                else:
                    self.Canvas_Widget.create_line(
                        route_points[i][0], route_points[i][1],
                        route_points[i+1][0], route_points[i+1][1],
                        fill='green', width=2
                    )
        else:
            # If only one point exists, draw a small circle at that point.
            self.Canvas_Widget.create_oval(
                route_points[0][0] - 5, route_points[0][1] - 5,
                route_points[0][0] + 5, route_points[0][1] + 5,
                fill='green', outline='green'
            )
        # Reposition the robot image to the starting point.
        start_coords = self.b[0]
        self.Canvas_Widget.delete(self.agent)
        self.agent = self.Canvas_Widget.create_image(start_coords[0], start_coords[1], anchor='nw', image=self.robot)
        # self.Canvas_Widget.tag_raise(self.agent)
        self.Canvas_Widget.tag_raise('grid_line')
    def store_route(self):
        """
        Returns the stored optimal route for further analysis or visualization.
        """
        return self.c

    def transformation(self, x, y):
        """
        Converts pixel coordinates (x, y) to a unique state index:
          - Computes the column index as int(x / PIXELS).
          - Computes the row index as int(y / PIXELS).
          - Returns the state index computed as: (row_index * grid_size) + column_index.
        This mapping ensures that each grid cell has a unique identifier.
        """
        width = self.grid_size
        s = int(x / PIXELS) + int(y / PIXELS * width)
        return s

def update():
    """
    A simple function that repeatedly resets and renders the environment.
    Demonstrates the agent taking random actions over multiple episodes.
    """
    for t in range(100):
        s = env.reset()
        while True:
            env.render()
            c = random.randint(0, 3)
            s_, r, done, info = env.step(c)
            if done:
                break

if __name__ == '__main__':
    env = Environment(grid_size=4)
    env.mainloop()
