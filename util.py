import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy

from env.custom_hopper import *
import random
import torch

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_model(env, seed, verbose = 0, algo="ppo"):
    if algo == "ppo":
        return PPO("MlpPolicy", env, verbose=verbose, seed=seed)
    elif algo == "sac":
        return SAC("MlpPolicy", env, verbose=verbose, seed=seed)

def load_model(file, env, algo):
    if algo == "ppo":
        return PPO.load(file, env=env)
    elif algo == "sac":
        return SAC.load(file, env=env)
    

def train(env, seed, total_timesteps, log_dir, env_id, verbose = 0, algo="ppo"):
    model = create_model(env, seed, verbose, algo)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(f"models/{algo.upper()}_{env_id}")
    plot_train_results(log_dir, env_id)

def test(env, model_file, render=False, test_episodes=50, algo="ppo", scenario="source_source"):
    model = load_model(model_file, env, algo)
    rewards, _= evaluate_policy(model, env, n_eval_episodes=test_episodes, render=render, return_episode_rewards=True)
    print(f"Test reward (avg +/- std): ({np.mean(rewards):.3f} +/- {np.std(rewards):.3f}) - Num episodes: {test_episodes}")
    plot_test_results(rewards, scenario)

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

def plot_train_results(log_folder, env_id, title="Learning Curve"):
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
    plt.title(title)
    plt.savefig(f"training_results/{env_id}_reward.png", dpi=300, bbox_inches="tight")

def plot_test_results(rewards, scenario):
    plt.figure()
    plt.plot(range(len(rewards)), rewards, label="Episode Reward")
    plt.axhline(np.mean(rewards), color="red", label="Average Reward")
    plt.title(f"Test over {len(rewards)} episodes")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.legend()
    plt.savefig(f"test_results/{scenario}_test_reward.png", dpi=300, bbox_inches="tight")