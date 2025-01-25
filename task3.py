from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

import gym
from env.custom_hopper import *


def train(env, total_timesteps=10000, **kwargs):
    model = PPO('MlpPolicy', env, verbose=1, **kwargs)
    model.learn(total_timesteps=total_timesteps)
    return model, env

def test(model, env, n_episodes=50):
    rewards, lengths = evaluate_policy(model, env, n_eval_episodes=n_episodes, return_episode_rewards=True)
    return rewards, lengths

def test_plot(rewards, lengths, title=""):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title(f'{title} - Rewards')
    plt.subplot(1, 2, 2)
    plt.plot(lengths)
    plt.title(f'{title} - Episode Lengths')
    plt.show()

def main():
    # Training on source domain
    source_env = gym.make('CustomHopper-source-v0')
    source_model, _ = train(source_env, total_timesteps=10000)
    source_model.save("source_model.mdl")

    # Training on target domain
    target_env = gym.make('CustomHopper-target-v0')
    target_model, _ = train(target_env, total_timesteps=10000)
    target_model.save("target_model.mdl")

    # Testing source model on source domain
    source_env = Monitor(gym.make('CustomHopper-source-v0'))
    source_model = PPO.load("source_model.mdl")
    source_rewards, source_lengths = test(source_model, source_env)
    test_plot(source_rewards, source_lengths, title="source->source")

    # Testing source model on target domain (lower bound)
    target_env = Monitor(gym.make('CustomHopper-target-v0'))
    source_model = PPO.load("source_model.mdl")
    source_on_target_rewards, source_on_target_lengths = test(source_model, target_env)
    test_plot(source_on_target_rewards, source_on_target_lengths, title="source->target")

    # Testing target model on target domain (upper bound)
    target_env = Monitor(gym.make('CustomHopper-target-v0'))
    target_model = PPO.load("target_model.mdl")
    target_rewards, target_lengths = test(target_model, target_env)
    test_plot(target_rewards, target_lengths, title="target->target")

    # Reporting results
    print("source->source average return:", np.mean(source_rewards))
    print("source->target average return:", np.mean(source_on_target_rewards))
    print("target->target average return:", np.mean(target_rewards))

if __name__ == "__main__":
    main()