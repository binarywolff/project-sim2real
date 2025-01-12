import gym
import numpy as np
import argparse

import sys

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from util import train, test

from env.custom_hopper import *

def set_seed(seed):
    if seed > 0:
        np.random.seed(seed)

# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="CustomHopper-source-v0", help="Environment to use")
    parser.add_argument("--total_timesteps", type=int, default=200_000, help="The total number of samples to train on")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning rate')
    parser.add_argument('--test_episodes', default=50, type=int, help='# episodes for test evaluations')
    return parser.parse_args(args)

def main():
    args = parse_args()
    set_seed(args.seed)

    env_id = args.env

    if env_id not in ['CustomHopper-source-v0', 'CustomHopper-target-v0', 'CustomHopper-v0']:
        print("Wrong environment specified, expected: CustomHopper-source-v0, CustomHopper-target-v0 or CustomHopper-v0")
        sys.exit()
    
    env = gym.make(args.env)

    # If no model was passed, train a policy from scratch.
    # Otherwise load the model from the file and go directly to testing.
    if args.test is None:
        log_dir = f"./tmp/gym/train/{env_id}"
        os.makedirs(log_dir, exist_ok=True)
        train(env, seed=args.seed, lr=args.lr, total_timesteps=args.total_timesteps, log_dir=log_dir, env_id=env_id)
    else:
        log_dir = f"./tmp/gym/test/{env_id}"
        os.makedirs(log_dir, exist_ok=True)
        test(env, model_file=args.test, render=args.render_test, test_episodes=args.test_episodes, log_dir=log_dir)

    env.close() 
if __name__ == "__main__":
    main()
       
