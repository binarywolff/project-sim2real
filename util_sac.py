import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from env.humanoid import *

def create_model(env, seed, lr=0.0003):
    return SAC("MlpPolicy", env, verbose=1, learning_rate=lr, seed=seed, device='auto')

def load_model(file, env):
    return SAC.load(file, env=env)

def train(env, seed, lr=0.0003, total_timesteps=200_000, log_dir='./tmp/gym', env_id='Humanoid-v3'):
    env = Monitor(env, log_dir) # Logs will be saved in log_dir/monitor.csv
    model = create_model(env, seed, lr)

    model.learn(total_timesteps=total_timesteps)
    model.save(f"models/SAC_{env_id}")
    plot_results(log_dir)

    # timestep_interval = 1000
    # for _ in range(0, total_timesteps, timestep_interval):
    #     model.learn(total_timesteps=timestep_interval)
    #     test(model, env)  # Assuming test_model is a function that tests the model

    # model.save(f"models/SAC_{env_id}")
    # plot_results(log_dir)

def test(env, model_file, render=True, test_episodes=50, log_dir='./tmp/gym'):
    model = load_model(model_file, env)
    env = Monitor(env, log_dir)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=test_episodes, render=render)
    print(f"Test reward (avg +/- std): ({mean_reward} +/- {std_reward}) - Num episodes: {test_episodes}")

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show(block=False)
    plt.savefig(f"reward_plot/sac_reward.png", dpi=300, bbox_inches="tight")