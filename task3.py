from stable_baselines3.common.monitor import Monitor
import numpy as np

import gym
from env.custom_hopper import *
from util import train, set_seed, test
import os

def main():
    set_seed(42)
    # Training on SOURCE domain
    print("Training on source")
    source_env = gym.make('CustomHopper-source-v0')
    log_dir = f"./tmp/gym/train/CustomHopper-source-v0"
    os.makedirs(log_dir, exist_ok=True)
    source_env = Monitor(source_env, log_dir) # Logs will be saved in log_dir/monitor.csv
    train(source_env, seed=42, total_timesteps=200_000, log_dir=log_dir, env_id='CustomHopper-source-v0', algo='ppo')

    # Training on TARGET domain
    print("Training on target")
    target_env = gym.make('CustomHopper-target-v0')
    log_dir = f"./tmp/gym/train/CustomHopper-target-v0"
    os.makedirs(log_dir, exist_ok=True)
    target_env = Monitor(target_env, log_dir) # Logs will be saved in log_dir/monitor.csv
    train(target_env, seed=42, total_timesteps=200_000, log_dir=log_dir, env_id='CustomHopper-target-v0', algo='ppo')

    # Testing SOURCE model on SOURCE domain
    print("Testing source->source configuration")
    source_env = gym.make('CustomHopper-source-v0')
    log_dir = f"./tmp/gym/test/CustomHopper-source-v0"
    os.makedirs(log_dir, exist_ok=True)
    source_env = Monitor(source_env, log_dir)
    test(source_env, model_file="models/PPO_CustomHopper-source-v0.zip", render=False, test_episodes=50, algo='ppo', scenario="source_source")

    # Testing SOURCE model on TARGET domain (lower bound)
    print("Testing source->target configuration")
    target_env = gym.make('CustomHopper-target-v0')
    log_dir = f"./tmp/gym/test/CustomHopper-target-v0"
    os.makedirs(log_dir, exist_ok=True)
    target_env = Monitor(target_env, log_dir)
    test(target_env, model_file="models/PPO_CustomHopper-source-v0.zip", render=False, test_episodes=50, algo='ppo', scenario="source_target")

    # Testing TARGET model on TARGET domain (upper bound)
    print("Testing target->target configuration")
    target_env = gym.make('CustomHopper-target-v0')
    log_dir = f"./tmp/gym/test/CustomHopper-target-v0"
    os.makedirs(log_dir, exist_ok=True)
    target_env = Monitor(target_env, log_dir)
    test(target_env, model_file="models/PPO_CustomHopper-target-v0.zip", render=False, test_episodes=50, algo='ppo', scenario="target_target")

if __name__ == "__main__":
    main()