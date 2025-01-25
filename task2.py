import gym
import argparse
import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from util import train, test, set_seed

from env.custom_hopper import *
from env.custom_half_cheetah import *
from env.custom_hopper_adr import *
from stable_baselines3.common.monitor import Monitor

# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="CustomHopper-source-v0", help="Environment to use")
    parser.add_argument("--total_timesteps", type=int, default=200_000, help="The total number of samples to train on")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--test_episodes', default=50, type=int, help='# episodes for test evaluations')
    parser.add_argument('--algo', default="ppo", type=str, help='Algorithm to use. Possible values: ppo or sac')
    return parser.parse_args(args)

def main():
    args = parse_args()
    
    if args.algo not in ["ppo", "sac"]:
        print("ERROR: possible algorithms ppo or sac")
        sys.exit()

    env_id = args.env
    env = gym.make(args.env)
    
    # If no model was passed, train a policy from scratch.
    # Otherwise load the model from the file and go directly to testing.
    if args.test is None:
        #Set seed for reproducibility
        set_seed(args.seed)
        log_dir = f"./tmp/gym/train/{env_id}"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir) # Logs will be saved in log_dir/monitor.csv
        train(env, seed=args.seed, total_timesteps=args.total_timesteps, log_dir=log_dir, env_id=env_id, algo=args.algo)
    else:
        log_dir = f"./tmp/gym/test/{env_id}"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir) # Logs will be saved in log_dir/monitor.csv
        test(env, model_file=args.test, render=args.render_test, test_episodes=args.test_episodes, algo=args.algo)

    env.close() 

if __name__ == "__main__":
    main()
       

