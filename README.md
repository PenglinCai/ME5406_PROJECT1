# Project 1: The Froze Lake Problem and Variations

## Create a Conda Virtual Environment with Python 3.6 under Windows 11 OS.
```sh
conda create -n Project1_venv python=3.6 
conda activate Project1_venv
```

## Install Dependencies
```sh
pip install -r requirements.txt
```

## Running different functions in `main.py`
(All the training results are given in the `Results` folder and Q tables are saved in the `Q Table` folder.)

### Usage examples
(You can change `--grid_size` to 4 or 10 and `--episodes` number as you like):

- Run Monte Carlo individually with default parameters:
  ```sh
  python main.py --mc --grid_size 10 --episodes 20000
  ```
- Run SARSA individually with default parameters:
  ```sh
  python main.py --sarsa --grid_size 10 --episodes 20000
  ```
- Run Q-learning individually with default parameters:
  ```sh
  python main.py --qlearning --grid_size 10 --episodes 20000
  ```

### Tasks
- **Task 1**: Average Reward curves for 3 methods in the same graph
  ```sh
  python main.py --task1 --grid_size 10 --episodes 20000
  ```
- **Task 2**: Success Rate curves for 3 methods in the same graph
  ```sh
  python main.py --task2 --grid_size 10 --episodes 20000
  ```
- **Task 3**: Time Steps curves for 3 methods in the same graph
  ```sh
  python main.py --task3 --grid_size 10 --episodes 20000
  ```
- **Task 4**: MSE of Q Values curves for 3 methods in the same graph
  ```sh
  python main.py --task4 --grid_size 10 --episodes 20000
  ```
- **Task 5**: Using 6 different gamma values for Monte Carlo
  ```sh
  python main.py --task5 --grid_size 10 --episodes 10000
  ```
- **Task 6**: Using 6 different epsilon values for Monte Carlo
  ```sh
  python main.py --task6 --grid_size 10 --episodes 20000
  ```
- **Task 7**: Using 6 different gamma values for SARSA
  ```sh
  python main.py --task7 --grid_size 10 --episodes 20000
  ```
- **Task 8**: Using 6 different learning rates for SARSA
  ```sh
  python main.py --task8 --grid_size 10 --episodes 20000
  ```
- **Task 9**: Using 6 different epsilon values for SARSA
  ```sh
  python main.py --task9 --grid_size 10 --episodes 20000
  ```
- **Task 10**: Using 6 different epsilon values for Q-learning
  ```sh
  python main.py --task10 --grid_size 10 --episodes 20000
  ```
- **Task 11**: Using 6 different learning rates for Q-learning
  ```sh
  python main.py --task11 --grid_size 10 --episodes 20000
  ```
- **Task 12**: Using 6 different gamma values for Q-learning
  ```sh
  python main.py --task12 --grid_size 10 --episodes 20000
  
#
