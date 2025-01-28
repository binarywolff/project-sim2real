import gym
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from util import moving_average
from stable_baselines3.common.results_plotter import ts2xy, load_results

from env.custom_hopper import *
from env.custom_hopper_adr import *
from env.custom_half_cheetah import *
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from automatic_domain_randomization import ADR, ADRCallback

import os
os.environ["OMP_NUM_THREADS"] = "1"

# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, help="Domain to be trained. Possible values source, target, udr, adr", required=True)
    parser.add_argument("--env", type=str, help="Environment of training. Possible values hopper, cheetah", required=True)
    parser.add_argument("--total_timesteps", type=int, default=None, help="Number of episodes to train for")
    parser.add_argument("--num_runs", type=int, default=10, help="How many independent training runs to conduct")
    return parser.parse_args(args)

def train_PPO(env_id, trainer_id, domain, environment, total_timesteps, log_dir):
    env = gym.make(env_id)
    env = Monitor(env, log_dir)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(f"models/{domain}_{environment}/PPO_{env_id}_{trainer_id}")

def train_SAC(env_id, trainer_id, domain, environment, total_timesteps, log_dir):
    env = gym.make(env_id)
    env = Monitor(env, log_dir)
    model = SAC("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(f"models/{domain}_{environment}/SAC_{env_id}_{trainer_id}")

def train_PPO_ADR(env_id, trainer_id, domain, environment, total_timesteps, log_dir):
    env = gym.make(env_id)
    #Initial ADR parameters
    min_max_bounds = [(1, 10) for _ in env.get_parameters()]
    masses_bounds = [(0.95*mass, 1.05*mass) for mass in env.get_parameters()] 
    thresholds = (550, 1150)
    delta = 0.1 #Update step size 
    m = 10 #Buffer size
    fixed_torso_mass = env.get_parameters()[0]
    
    env = Monitor(env, log_dir)

    adr_env = gym.make(env_id)
    adr_env = DummyVecEnv([lambda: adr_env])

    adr = ADR(masses_bounds, thresholds, delta, m, min_max_bounds, adr_env, fixed_torso_mass)
    adr_callback = ADRCallback(adr, env, f'entropy_log_{trainer_id}.csv')

    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps, callback=adr_callback, progress_bar=True)
    model.save(f"models/{domain}_{environment}/PPO_ADR_{env_id}_{trainer_id}")

def train_SAC_ADR(env_id, trainer_id, domain, environment, total_timesteps, log_dir):
    env = gym.make(env_id)
    #Initial ADR parameters
    min_max_bounds = [(1, 10) for _ in env.get_parameters()]
    masses_bounds = [(0.95 * mass, 1.05 * mass) for mass in env.get_parameters()]
    thresholds = (600, 1450)
    delta = 0.1 #Update step size 
    m = 1 #Buffer size
    fixed_torso_mass = env.get_parameters()[0]
    
    env = Monitor(env, log_dir)

    adr_env = gym.make(env_id)
    adr_env = DummyVecEnv([lambda: adr_env])

    adr = ADR(masses_bounds, thresholds, delta, m, min_max_bounds, adr_env, fixed_torso_mass)
    adr_callback = ADRCallback(adr, env, f'entropy_log_{trainer_id}.csv')

    model = SAC("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps, callback=adr_callback, progress_bar=True)
    model.save(f"models/{domain}_{environment}/SAC_ADR_{env_id}_{trainer_id}")

def trainer(domain, environment, total_timesteps, num_runs):
    env_id = get_env_id(domain, environment)
    results = []
    for trainer_id in range(0,num_runs):
        print("Trainer id", trainer_id, "started")
        log_dir = f"./tmp/gym/multi_train/{env_id}_{trainer_id}"
        os.makedirs(log_dir, exist_ok=True)
        if environment == "hopper":
            if domain != "adr":
                train_PPO(env_id, trainer_id, domain, environment, total_timesteps, log_dir)
            else:
                train_PPO_ADR(env_id, trainer_id, domain, environment, total_timesteps, log_dir)
        elif environment == "cheetah":
            if domain != "adr":
                train_SAC(env_id, trainer_id, domain, environment, total_timesteps, log_dir)
            else: 
                train_SAC_ADR(env_id, trainer_id, domain, environment, total_timesteps, log_dir)
        x, y = ts2xy(load_results(log_dir), "timesteps")
        y = moving_average(y, window=50)
        x = x[len(x) - len(y) :]
        data = pd.DataFrame({"timesteps": x,
                             "train_run_id": [trainer_id]*len(x),
                             "reward": y})
        results.append(data)
        print("Trainer id", trainer_id, "ended")
    
    plt.figure(figsize=(12, 8))
    all_results = pd.concat(results)
    sns.lineplot(x="timesteps", y="reward", hue="train_run_id", data=all_results, dashes=[(2,2)]*10, palette="Set2", style="train_run_id")
    plt.title("Training Performance over 10 different trained models")
    plt.savefig(f"training_results/{environment}_{domain}_training.png")

def get_env_id(domain, env):

    hopper_dict = {
        "source" : "CustomHopper-source-v0",
        "target" : "CustomHopper-target-v0", 
        "udr" : "CustomHopper-udr-v0", 
        "adr" : "CustomHopper-source-adr-v0",
    }

    cheetah_dict = {
        "udr" : "HalfCheetah-source-udr-v3", 
        "adr" : "HalfCheetah-source-adr-v3",
    }

    if env == "hopper":
        return hopper_dict[domain]
    elif env == "cheetah":
        return cheetah_dict[domain]

# Entry point of the script
if __name__ == "__main__":
    args = parse_args()

    if args.domain not in ["source", "target", "udr", "adr"]:
        print("ERROR: domain accepted values: source, target, udr, adr")
        sys.exit()
    if args.env not in ["hopper", "cheetah"]:
        print("ERROR: env accepted values: hopper, cheetah")
        sys.exit()
    if args.env == "cheetah" and (args.domain == "source" or args.domain == "target"):
        print("ERROR: unsupported combination only cheetah-adr, cheetah-udr")
        sys.exit()
    if args.total_timesteps == None:
        total_timesteps = 1_000_000 if args.env == "hopper" else 200_000
    else:
        total_timesteps = args.total_timesteps 

    trainer(args.domain, args.env, total_timesteps, args.num_runs)