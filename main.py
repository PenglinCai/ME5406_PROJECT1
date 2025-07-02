"""
Main runner for RL algorithms.
Usage examples:
  Run Monte Carlo individually:
    python main.py --mc --grid_size 10 --episodes 20000
  Run SARSA individually:
    python main.py --sarsa --grid_size 10 --episodes 20000
  Run Q-learning individually:
    python main.py --qlearning --grid_size 10 --episodes 20000
  Task 1 (Average Reward curves):
    python main.py --task1 --grid_size 10 --episodes 20000
  Task 2 (Success Rate curves):
    python main.py --task2 --grid_size 10 --episodes 20000
  Task 3 (Time Steps curves):
    python main.py --task3 --grid_size 10 --episodes 20000
  Task 4 (MSE of Q Values curves):
    python main.py --task4 --grid_size 10 --episodes 20000
  Task 5 (Using 6 different gamma values for Monte Carlo):
    python main.py --task5 --grid_size 10 --episodes 20000
  Task 6 (Using 6 different epsilon values for Monte Carlo):
    python main.py --task6 --grid_size 10 --episodes 20000
  Task 7 (Using 6 different gamma values for SARSA):
    python main.py --task7 --grid_size 10 --episodes 20000
  Task 8 (Using 6 different learning rates for SARSA):
    python main.py --task8 --grid_size 10 --episodes 20000
  Task 9 (Using 6 different epsilon values for SARSA):
    python main.py --task9 --grid_size 10 --episodes 20000
  Task 10 (Using 6 different epsilon values for Q-learning):
    python main.py --task10 --grid_size 10 --episodes 20000
  Task 11 (Using 6 different learning rates for Q-learning):
    python main.py --task11 --grid_size 10 --episodes 20000
  Task 12 (Using 6 different gamma values for Q-learning):
    python main.py --task12 --grid_size 10 --episodes 20000
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run RL algorithms with specified parameters")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--mc', action='store_true', help="Run Monte Carlo algorithm")
group.add_argument('--sarsa', action='store_true', help="Run SARSA algorithm")
group.add_argument('--qlearning', action='store_true', help="Run Q-learning algorithm")
group.add_argument('--task1', action='store_true', help="Task 1: Plot Average Reward curves for all algorithms")
group.add_argument('--task2', action='store_true', help="Task 2: Plot Success Rate curves for all algorithms")
group.add_argument('--task3', action='store_true', help="Task 3: Plot Time Steps curves for all algorithms")
group.add_argument('--task4', action='store_true', help="Task 4: Plot MSE of Q Values curves for all algorithms")
group.add_argument('--task5', action='store_true', help="Task 5: Run Monte Carlo with 6 different gamma values and plot curves")
group.add_argument('--task6', action='store_true', help="Task 6: Run Monte Carlo with 6 different epsilon values and plot curves")
group.add_argument('--task7', action='store_true', help="Task 7: Run SARSA with 6 different gamma values and plot curves")
group.add_argument('--task8', action='store_true', help="Task 8: Run SARSA with 6 different learning rates and plot curves")
group.add_argument('--task9', action='store_true', help="Task 9: Run SARSA with 6 different epsilon values and plot curves")
group.add_argument('--task10', action='store_true', help="Task 10: Run Q-learning with 6 different epsilon values and plot curves")
group.add_argument('--task11', action='store_true', help="Task 11: Run Q-learning with 6 different learning rates and plot curves")
group.add_argument('--task12', action='store_true', help="Task 12: Run Q-learning with 6 different gamma values and plot curves")
parser.add_argument('--grid_size', type=int, choices=[4, 10, 12], required=True,
                    help="Grid size of the environment (4, 10 or 12)")
parser.add_argument('--episodes', type=int, required=True,
                    help="Number of training episodes")
args = parser.parse_args()

# Update global parameters based on command-line input
import Parameters
Parameters.GRID_SIZE = args.grid_size
Parameters.GRID_HEIGHT = args.grid_size
Parameters.GRID_WIDTH = args.grid_size

# -------------------------------------------------------------------
# Run individual algorithms
# -------------------------------------------------------------------
if args.mc:
    # Run Monte Carlo algorithm individually
    from Environment import Environment
    from First_Visit_Monte_Carlo_Control import MonteCarlo

    env = Environment(grid_size=args.grid_size)
    mc_agent = MonteCarlo(env, epsilon=Parameters.MONTE_CARLO_EPSILON, gamma=Parameters.GAMMA)
    mc_agent.first_visit_monte_carlo_control(num_eps=args.episodes)
    mc_agent.display_optimal_route()
    env.mainloop()

elif args.sarsa:
    # Run SARSA algorithm individually
    from Environment import Environment
    from SARSA import SARSA

    env = Environment(grid_size=args.grid_size)
    sarsa_agent = SARSA(env, learning_rate=Parameters.LEARNING_RATE, gamma=Parameters.GAMMA,
                        epsilon=Parameters.SARSA_EPSILON)
    sarsa_agent.train(num_epd=args.episodes)
    sarsa_agent.draw_best_route()
    env.mainloop()

elif args.qlearning:
    # Run Q-learning algorithm individually
    from Environment import Environment
    from Q_learning import Q_learning

    env = Environment(grid_size=args.grid_size)
    ql_agent = Q_learning(env, learning_rate=Parameters.LEARNING_RATE, gamma=Parameters.GAMMA,
                          epsilon=Parameters.QLEARNING_EPSILON)
    ql_agent.train(num_epd=args.episodes)
    ql_agent.draw_best_route()
    env.mainloop()

# -------------------------------------------------------------------
# Run tasks that combine all three algorithms and plot one combined metric
# -------------------------------------------------------------------
elif args.task1 or args.task2 or args.task3 or args.task4:
    from Environment import Environment
    from First_Visit_Monte_Carlo_Control import MonteCarlo
    from SARSA import SARSA
    from Q_learning import Q_learning

    def run_monte_carlo():
        env_mc = Environment(grid_size=args.grid_size)
        mc_agent = MonteCarlo(env_mc, epsilon=Parameters.MONTE_CARLO_EPSILON, gamma=Parameters.GAMMA)
        mc_agent.plot_results = lambda *args, **kwargs: None
        results = mc_agent.first_visit_monte_carlo_control(num_eps=args.episodes)
        env_mc.destroy()
        return results  # (q_table, time_steps, mse_q, success_rate, cost_bar, average_reward)

    def run_sarsa():
        env_sarsa = Environment(grid_size=args.grid_size)
        sarsa_agent = SARSA(env_sarsa, learning_rate=Parameters.LEARNING_RATE, gamma=Parameters.GAMMA,
                            epsilon=Parameters.SARSA_EPSILON)
        sarsa_agent.plot_results = lambda *args, **kwargs: None
        results = sarsa_agent.train(num_epd=args.episodes)
        env_sarsa.destroy()
        return results

    def run_qlearning():
        env_ql = Environment(grid_size=args.grid_size)
        ql_agent = Q_learning(env_ql, learning_rate=Parameters.LEARNING_RATE, gamma=Parameters.GAMMA,
                              epsilon=Parameters.QLEARNING_EPSILON)
        ql_agent.plot_results = lambda *args, **kwargs: None
        results = ql_agent.train(num_epd=args.episodes)
        env_ql.destroy()
        return results

    results_mc = run_monte_carlo()       # (q_table, time_steps, mse_q, success_rate, cost_bar, average_reward)
    results_sarsa = run_sarsa()
    results_ql = run_qlearning()

    _, time_steps_mc, mse_q_mc, success_rate_mc, _, avg_reward_mc = results_mc
    _, time_steps_sarsa, mse_q_sarsa, success_rate_sarsa, _, avg_reward_sarsa = results_sarsa
    _, time_steps_ql, mse_q_ql, success_rate_ql, _, avg_reward_ql = results_ql

    if args.task1:
        plt.figure()
        plt.plot(np.arange(len(avg_reward_mc)), avg_reward_mc, label="Monte Carlo", linewidth=1)
        plt.plot(np.arange(len(avg_reward_sarsa)), avg_reward_sarsa, label="SARSA", linewidth=1)
        plt.plot(np.arange(len(avg_reward_ql)), avg_reward_ql, label="Q-learning", linewidth=1)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Average Rewards")
        plt.title("Average Rewards vs Episodes")
        plt.legend()
        plt.grid(True)
        plt.show()

    elif args.task2:
        plt.figure()
        plt.plot(np.arange(len(success_rate_mc)), success_rate_mc, label="Monte Carlo", linewidth=1)
        plt.plot(np.arange(len(success_rate_sarsa)), success_rate_sarsa, label="SARSA", linewidth=1)
        plt.plot(np.arange(len(success_rate_ql)), success_rate_ql, label="Q-learning", linewidth=1)
        plt.xlabel("Batch (every 50 episodes)")
        plt.ylabel("Success Rate")
        plt.title("Success Rate per 50 Episodes")
        plt.legend()
        plt.grid(True)
        plt.show()

    elif args.task3:
        plt.figure()
        plt.plot(np.arange(len(time_steps_mc)), time_steps_mc, label="Monte Carlo", linewidth=1)
        plt.plot(np.arange(len(time_steps_sarsa)), time_steps_sarsa, label="SARSA", linewidth=1)
        plt.plot(np.arange(len(time_steps_ql)), time_steps_ql, label="Q-learning", linewidth=1)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Time Steps")
        plt.title("Time Steps vs Episode")
        plt.legend()
        plt.grid(True)
        plt.show()

    elif args.task4:
        plt.figure()
        plt.plot(np.arange(len(mse_q_mc)), mse_q_mc, label="Monte Carlo", linewidth=1)
        plt.plot(np.arange(len(mse_q_sarsa)), mse_q_sarsa, label="SARSA", linewidth=1)
        plt.plot(np.arange(len(mse_q_ql)), mse_q_ql, label="Q-learning", linewidth=1)
        plt.xlabel("Number of Episodes")
        plt.ylabel("MSE of Q Values")
        plt.title("MSE of Q Values vs Episode")
        plt.legend()
        plt.grid(True)
        plt.show()

elif args.task5:
    # Task 5: Run Monte Carlo algorithm with 6 different gamma values,
    # and plot the Average Reward, Success Rate, and MSE of Q Values curves.
    # Display the three plots in a single row subplot, each subplot containing curves for different gamma values.
    from Environment import Environment
    from First_Visit_Monte_Carlo_Control import MonteCarlo
    import Parameters
    import matplotlib.pyplot as plt
    import numpy as np

    gamma_list = [0.1, 0.2, 0.5, 0.7, 0.9, 0.99]
    avg_rewards_dict = {}
    success_rates_dict = {}
    mse_q_dict = {}

    for gamma_val in gamma_list:
        env = Environment(grid_size=args.grid_size)
        mc_agent = MonteCarlo(env, epsilon=Parameters.MONTE_CARLO_EPSILON, gamma=gamma_val)
        mc_agent.plot_results = lambda *args, **kwargs: None
        _, _, mse_q, success_rate, _, avg_reward = mc_agent.first_visit_monte_carlo_control(num_eps=args.episodes)
        avg_rewards_dict[gamma_val] = avg_reward
        success_rates_dict[gamma_val] = success_rate
        mse_q_dict[gamma_val] = mse_q
        env.destroy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for gamma_val in gamma_list:
        axes[0].plot(np.arange(len(avg_rewards_dict[gamma_val])), avg_rewards_dict[gamma_val],
                     label=f"gamma={gamma_val}", linewidth=1)
    axes[0].set_xlabel("Number of Episodes")
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title("Average Reward vs Episodes")
    axes[0].legend()
    axes[0].grid(True)

    for gamma_val in gamma_list:
        axes[1].plot(np.arange(len(success_rates_dict[gamma_val])), success_rates_dict[gamma_val],
                     label=f"gamma={gamma_val}", linewidth=1)
    axes[1].set_xlabel("Batch (every 50 episodes)")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Success Rate vs Episodes")
    axes[1].legend()
    axes[1].grid(True)

    for gamma_val in gamma_list:
        axes[2].plot(np.arange(len(mse_q_dict[gamma_val])), mse_q_dict[gamma_val],
                     label=f"gamma={gamma_val}", linewidth=1)
    axes[2].set_xlabel("Number of Episodes")
    axes[2].set_ylabel("MSE of Q Values")
    axes[2].set_title("MSE of Q Values vs Episodes")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

elif args.task6:
    # Task 6: Run Monte Carlo algorithm with 6 different epsilon values,
    # and plot the Average Reward, Success Rate, and MSE of Q Values curves.
    # Display the three plots in a single row subplot, each subplot containing curves for different epsilon values.
    from Environment import Environment
    from First_Visit_Monte_Carlo_Control import MonteCarlo
    import Parameters
    import matplotlib.pyplot as plt
    import numpy as np

    epsilon_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    avg_rewards_dict = {}
    success_rates_dict = {}
    mse_q_dict = {}

    for eps in epsilon_list:
        env = Environment(grid_size=args.grid_size)
        mc_agent = MonteCarlo(env, epsilon=eps, gamma=Parameters.GAMMA)
        mc_agent.plot_results = lambda *args, **kwargs: None
        _, _, mse_q, success_rate, _, avg_reward = mc_agent.first_visit_monte_carlo_control(num_eps=args.episodes)
        avg_rewards_dict[eps] = avg_reward
        success_rates_dict[eps] = success_rate
        mse_q_dict[eps] = mse_q
        env.destroy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for eps in epsilon_list:
        axes[0].plot(np.arange(len(avg_rewards_dict[eps])), avg_rewards_dict[eps],
                     label=f"epsilon={eps}", linewidth=1)
    axes[0].set_xlabel("Number of Episodes")
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title("Average Reward vs Episodes")
    axes[0].legend()
    axes[0].grid(True)

    for eps in epsilon_list:
        axes[1].plot(np.arange(len(success_rates_dict[eps])), success_rates_dict[eps],
                     label=f"epsilon={eps}", linewidth=1)
    axes[1].set_xlabel("Batch (every 50 episodes)")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Success Rate vs Episodes")
    axes[1].legend()
    axes[1].grid(True)

    for eps in epsilon_list:
        axes[2].plot(np.arange(len(mse_q_dict[eps])), mse_q_dict[eps],
                     label=f"epsilon={eps}", linewidth=1)
    axes[2].set_xlabel("Number of Episodes")
    axes[2].set_ylabel("MSE of Q Values")
    axes[2].set_title("MSE of Q Values vs Episodes")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

elif args.task7:
    # Task 7: Run SARSA algorithm with 6 different gamma values,
    # and plot the Average Reward, Success Rate, and MSE of Q Values curves.
    # Display the three plots in a single row subplot, each subplot containing curves for different gamma values.
    from Environment import Environment
    from SARSA import SARSA
    import Parameters
    import matplotlib.pyplot as plt
    import numpy as np

    gamma_list = [0.1, 0.2, 0.5, 0.7, 0.9, 0.99]
    avg_rewards_dict = {}
    success_rates_dict = {}
    mse_q_dict = {}

    for gamma_val in gamma_list:
        env = Environment(grid_size=args.grid_size)
        sarsa_agent = SARSA(env, learning_rate=Parameters.LEARNING_RATE, gamma=gamma_val,
                            epsilon=Parameters.SARSA_EPSILON)
        sarsa_agent.plot_results = lambda *args, **kwargs: None
        _, _, mse_q, success_rate, _, avg_reward = sarsa_agent.train(num_epd=args.episodes)
        avg_rewards_dict[gamma_val] = avg_reward
        success_rates_dict[gamma_val] = success_rate
        mse_q_dict[gamma_val] = mse_q
        env.destroy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for gamma_val in gamma_list:
        axes[0].plot(np.arange(len(avg_rewards_dict[gamma_val])), avg_rewards_dict[gamma_val],
                     label=f"gamma={gamma_val}", linewidth=1)
    axes[0].set_xlabel("Number of Episodes")
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title("Average Reward vs Episodes")
    axes[0].legend()
    axes[0].grid(True)

    for gamma_val in gamma_list:
        axes[1].plot(np.arange(len(success_rates_dict[gamma_val])), success_rates_dict[gamma_val],
                     label=f"gamma={gamma_val}", linewidth=1)
    axes[1].set_xlabel("Batch (every 50 episodes)")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Success Rate vs Episodes")
    axes[1].legend()
    axes[1].grid(True)

    for gamma_val in gamma_list:
        axes[2].plot(np.arange(len(mse_q_dict[gamma_val])), mse_q_dict[gamma_val],
                     label=f"gamma={gamma_val}", linewidth=1)
    axes[2].set_xlabel("Number of Episodes")
    axes[2].set_ylabel("MSE of Q Values")
    axes[2].set_title("MSE of Q Values vs Episodes")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

elif args.task8:
    # Task 8: Run SARSA algorithm with 6 different learning rates,
    # and plot the Average Reward, Success Rate, and MSE of Q Values curves.
    # Display the three plots in a single row subplot, each subplot containing curves for different learning rates.
    from Environment import Environment
    from SARSA import SARSA
    import Parameters
    import matplotlib.pyplot as plt
    import numpy as np

    learning_rate_list = [0.001,0.01, 0.05, 0.1, 0.5, 0.9]
    avg_rewards_dict = {}
    success_rates_dict = {}
    mse_q_dict = {}

    for lr in learning_rate_list:
        env = Environment(grid_size=args.grid_size)
        sarsa_agent = SARSA(env, learning_rate=lr, gamma=Parameters.GAMMA,
                            epsilon=Parameters.SARSA_EPSILON)
        sarsa_agent.plot_results = lambda *args, **kwargs: None
        _, _, mse_q, success_rate, _, avg_reward = sarsa_agent.train(num_epd=args.episodes)
        avg_rewards_dict[lr] = avg_reward
        success_rates_dict[lr] = success_rate
        mse_q_dict[lr] = mse_q
        env.destroy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for lr in learning_rate_list:
        axes[0].plot(np.arange(len(avg_rewards_dict[lr])), avg_rewards_dict[lr],
                     label=f"lr={lr}", linewidth=1)
    axes[0].set_xlabel("Number of Episodes")
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title("Average Reward vs Episodes")
    axes[0].legend()
    axes[0].grid(True)

    for lr in learning_rate_list:
        axes[1].plot(np.arange(len(success_rates_dict[lr])), success_rates_dict[lr],
                     label=f"lr={lr}", linewidth=1)
    axes[1].set_xlabel("Batch (every 50 episodes)")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Success Rate vs Episodes")
    axes[1].legend()
    axes[1].grid(True)

    for lr in learning_rate_list:
        axes[2].plot(np.arange(len(mse_q_dict[lr])), mse_q_dict[lr],
                     label=f"lr={lr}", linewidth=1)
    axes[2].set_xlabel("Number of Episodes")
    axes[2].set_ylabel("MSE of Q Values")
    axes[2].set_title("MSE of Q Values vs Episodes")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

elif args.task9:
    # Task 9: Run SARSA algorithm with 6 different epsilon values,
    # and plot the Average Reward, Success Rate, and MSE of Q Values curves.
    # Display the three plots in a single row subplot, each subplot containing curves for different epsilon values.
    from Environment import Environment
    from SARSA import SARSA
    import Parameters
    import matplotlib.pyplot as plt
    import numpy as np

    epsilon_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    avg_rewards_dict = {}
    success_rates_dict = {}
    mse_q_dict = {}

    for eps in epsilon_list:
        env = Environment(grid_size=args.grid_size)
        sarsa_agent = SARSA(env, learning_rate=Parameters.LEARNING_RATE, gamma=Parameters.GAMMA,
                            epsilon=eps)
        sarsa_agent.plot_results = lambda *args, **kwargs: None
        _, _, mse_q, success_rate, _, avg_reward = sarsa_agent.train(num_epd=args.episodes)
        avg_rewards_dict[eps] = avg_reward
        success_rates_dict[eps] = success_rate
        mse_q_dict[eps] = mse_q
        env.destroy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for eps in epsilon_list:
        axes[0].plot(np.arange(len(avg_rewards_dict[eps])), avg_rewards_dict[eps],
                     label=f"epsilon={eps}", linewidth=1)
    axes[0].set_xlabel("Number of Episodes")
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title("Average Reward vs Episodes")
    axes[0].legend()
    axes[0].grid(True)

    for eps in epsilon_list:
        axes[1].plot(np.arange(len(success_rates_dict[eps])), success_rates_dict[eps],
                     label=f"epsilon={eps}", linewidth=1)
    axes[1].set_xlabel("Batch (every 50 episodes)")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Success Rate vs Episodes")
    axes[1].legend()
    axes[1].grid(True)

    for eps in epsilon_list:
        axes[2].plot(np.arange(len(mse_q_dict[eps])), mse_q_dict[eps],
                     label=f"epsilon={eps}", linewidth=1)
    axes[2].set_xlabel("Number of Episodes")
    axes[2].set_ylabel("MSE of Q Values")
    axes[2].set_title("MSE of Q Values vs Episodes")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

elif args.task10:
    # Task 10: Run Q-learning algorithm with 6 different epsilon values,
    # and plot the Average Reward, Success Rate, and MSE of Q Values curves.
    # Display the three plots in a single row subplot, each subplot containing curves for different epsilon values.
    from Environment import Environment
    from Q_learning import Q_learning
    import Parameters
    import matplotlib.pyplot as plt
    import numpy as np

    epsilon_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    avg_rewards_dict = {}
    success_rates_dict = {}
    mse_q_dict = {}

    for eps in epsilon_list:
        env = Environment(grid_size=args.grid_size)
        ql_agent = Q_learning(env, learning_rate=Parameters.LEARNING_RATE, gamma=Parameters.GAMMA,
                              epsilon=eps)
        ql_agent.plot_results = lambda *args, **kwargs: None
        _, _, mse_q, success_rate, _, avg_reward = ql_agent.train(num_epd=args.episodes)
        avg_rewards_dict[eps] = avg_reward
        success_rates_dict[eps] = success_rate
        mse_q_dict[eps] = mse_q
        env.destroy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for eps in epsilon_list:
        axes[0].plot(np.arange(len(avg_rewards_dict[eps])), avg_rewards_dict[eps],
                     label=f"epsilon={eps}", linewidth=1)
    axes[0].set_xlabel("Number of Episodes")
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title("Average Reward vs Episodes")
    axes[0].legend()
    axes[0].grid(True)

    for eps in epsilon_list:
        axes[1].plot(np.arange(len(success_rates_dict[eps])), success_rates_dict[eps],
                     label=f"epsilon={eps}", linewidth=1)
    axes[1].set_xlabel("Batch (every 50 episodes)")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Success Rate vs Episodes")
    axes[1].legend()
    axes[1].grid(True)

    for eps in epsilon_list:
        axes[2].plot(np.arange(len(mse_q_dict[eps])), mse_q_dict[eps],
                     label=f"epsilon={eps}", linewidth=1)
    axes[2].set_xlabel("Number of Episodes")
    axes[2].set_ylabel("MSE of Q Values")
    axes[2].set_title("MSE of Q Values vs Episodes")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

elif args.task11:
    # Task 11: Run Q-learning algorithm with 6 different learning rates,
    # and plot the Average Reward, Success Rate, and MSE of Q Values curves.
    # Display the three plots in a single row subplot, each subplot containing curves for different learning rates.
    from Environment import Environment
    from Q_learning import Q_learning
    import Parameters
    import matplotlib.pyplot as plt
    import numpy as np

    learning_rate_list = [0.001,0.01, 0.05, 0.1, 0.5, 0.9]
    avg_rewards_dict = {}
    success_rates_dict = {}
    mse_q_dict = {}

    for lr in learning_rate_list:
        env = Environment(grid_size=args.grid_size)
        ql_agent = Q_learning(env, learning_rate=lr, gamma=Parameters.GAMMA,
                              epsilon=Parameters.QLEARNING_EPSILON)
        ql_agent.plot_results = lambda *args, **kwargs: None
        _, _, mse_q, success_rate, _, avg_reward = ql_agent.train(num_epd=args.episodes)
        avg_rewards_dict[lr] = avg_reward
        success_rates_dict[lr] = success_rate
        mse_q_dict[lr] = mse_q
        env.destroy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for lr in learning_rate_list:
        axes[0].plot(np.arange(len(avg_rewards_dict[lr])), avg_rewards_dict[lr],
                     label=f"lr={lr}", linewidth=1)
    axes[0].set_xlabel("Number of Episodes")
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title("Average Reward vs Episodes")
    axes[0].legend()
    axes[0].grid(True)

    for lr in learning_rate_list:
        axes[1].plot(np.arange(len(success_rates_dict[lr])), success_rates_dict[lr],
                     label=f"lr={lr}", linewidth=1)
    axes[1].set_xlabel("Batch (every 50 episodes)")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Success Rate vs Episodes")
    axes[1].legend()
    axes[1].grid(True)

    for lr in learning_rate_list:
        axes[2].plot(np.arange(len(mse_q_dict[lr])), mse_q_dict[lr],
                     label=f"lr={lr}", linewidth=1)
    axes[2].set_xlabel("Number of Episodes")
    axes[2].set_ylabel("MSE of Q Values")
    axes[2].set_title("MSE of Q Values vs Episodes")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

elif args.task12:
    # Task 12: Run Q-learning algorithm with 6 different gamma values,
    # and plot the Average Reward, Success Rate, and MSE of Q Values curves.
    # Display the three plots in a single row subplot, each subplot containing only curves for different gamma values.
    from Environment import Environment
    from Q_learning import Q_learning
    import Parameters
    import matplotlib.pyplot as plt
    import numpy as np

    gamma_list = [0.1, 0.2, 0.5, 0.7, 0.9, 0.99]
    avg_rewards_dict = {}
    success_rates_dict = {}
    mse_q_dict = {}

    for gamma_val in gamma_list:
        env = Environment(grid_size=args.grid_size)
        ql_agent = Q_learning(env, learning_rate=Parameters.LEARNING_RATE, gamma=gamma_val,
                              epsilon=Parameters.QLEARNING_EPSILON)
        # Disable internal plotting
        ql_agent.plot_results = lambda *args, **kwargs: None
        _, time_steps, mse_q, success_rate, _, avg_reward = ql_agent.train(num_epd=args.episodes)
        avg_rewards_dict[gamma_val] = avg_reward
        success_rates_dict[gamma_val] = success_rate
        mse_q_dict[gamma_val] = mse_q
        env.destroy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for gamma_val in gamma_list:
        axes[0].plot(np.arange(len(avg_rewards_dict[gamma_val])), avg_rewards_dict[gamma_val],
                     label=f"gamma={gamma_val}", linewidth=1)
    axes[0].set_xlabel("Number of Episodes")
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title("Average Reward vs Episodes")
    axes[0].legend()
    axes[0].grid(True)

    for gamma_val in gamma_list:
        axes[1].plot(np.arange(len(success_rates_dict[gamma_val])), success_rates_dict[gamma_val],
                     label=f"gamma={gamma_val}", linewidth=1)
    axes[1].set_xlabel("Batch (every 50 episodes)")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Success Rate vs Episodes")
    axes[1].legend()
    axes[1].grid(True)

    for gamma_val in gamma_list:
        axes[2].plot(np.arange(len(mse_q_dict[gamma_val])), mse_q_dict[gamma_val],
                     label=f"gamma={gamma_val}", linewidth=1)
    axes[2].set_xlabel("Number of Episodes")
    axes[2].set_ylabel("MSE of Q Values")
    axes[2].set_title("MSE of Q Values vs Episodes")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
