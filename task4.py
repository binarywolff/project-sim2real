import argparse
import gym 
import numpy as np
import sys
import matplotlib.pyplot as plt
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="CustomHopper-udr-v0", help="Environment to use")
    parser.add_argument("--total_timesteps", type=int, default=200000, help="The total number of samples to train on")
    parser.add_argument("--render_test", action='store_false', help="Render test")
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning rate')
    parser.add_argument('--test_episodes', default=50, type=int, help='# episodes for test evaluations')
    parser.add_argument('--thigh_range', type=float, nargs=2, default=[0.5, 1.5], help='Range for thigh mass randomization')
    parser.add_argument('--leg_range', type=float, nargs=2, default=[0.5, 1.5], help='Range for leg mass randomization')
    parser.add_argument('--foot_range', type=float, nargs=2, default=[0.5, 1.5], help='Range for foot mass randomization')
    return parser.parse_args(args)
def main():
    args = parse_args()
    np.random.seed(args.seed)

    env = gym.make(args.env, thigh_range=args.thigh_range, leg_range=args.leg_range, foot_range=args.foot_range)
    env.seed(args.seed)

    if args.test is not None:
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=args.lr, seed=args.seed, device='auto')
        model.learn(total_timesteps=args.total_timesteps)
        model.save(f"models_{args.env}_{args.total_timesteps}")
    else:
        target_env = gym.make("CustomHopper-target-v0")
        target_env.seed(args.seed)
        model = PPO.load(f"env/models_{args.env}_{args.total_timesteps}", env=target_env, verbose=1)
        mean_reward, std_reward = evaluate_policy(model, target_env, n_eval_episodes=args.test_episodes, render=args.render_test)
        print(f"Test reward (avg +/- std): ({mean_reward} +/- {std_reward}) - Num episodes: {args.test_episodes}")

if __name__ == "__main__":
    main()