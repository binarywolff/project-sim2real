import gym
import os
import pandas as pd
import argparse
import sys

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from stable_baselines3 import PPO, SAC
from env.custom_half_cheetah import *
from env.custom_hopper import *
from env.custom_hopper_adr import *
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Trained Model for the test. Possible values source, target, udr, adr", required=True)
    parser.add_argument("--env", type=str, help="Environment of the test. Possible values hopper, cheetah", required=True)
    parser.add_argument("--num_runs", type=int, default=10, help="Number of models to be tested")
    parser.add_argument("--test_episodes", type=int, default=5, help="Number of test episodes")
    return parser.parse_args(args)

def tester(model_type, environment, test_episodes, num_runs):
    all_rewards = []
    all_eps = []

    #Always target env during testing
    env_id = "CustomHopper-target-v0" if environment == "hopper" else "HalfCheetah-target-v3"

    if model_type == "source":
        print("SOURCE -> TARGET")
    elif model_type == "adr":
        print("SOURCE-ADR -> TARGET")
    elif model_type == "target": 
        print("TARGET -> TARGET")
    elif model_type == "udr": 
        print("SOURCE-UDR -> TARGET")
    
    for i in range(0,num_runs):
        env = gym.make(env_id)
        log_dir = f"./tmp/gym/test/{env_id}_{i}"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        if environment == "hopper":
            if model_type == "source":
                
                model = PPO.load(f"models/{model_type}_{environment}/PPO_CustomHopper-source-v0_{i}.zip", env=env) 
            elif model_type == "adr":
                model = PPO.load(f"models/{model_type}_{environment}/PPO_ADR_CustomHopper-source-adr-v0_{i}.zip", env=env)
            elif model_type == "target": 
                model = PPO.load(f"models/{model_type}_{environment}/PPO_CustomHopper-target-v0_{i}.zip", env=env)
            elif model_type == "udr":
                print("models/{model_type}_{environment}/CustomHopper-udr-v0_{i}.zip")
                model = PPO.load(f"models/{model_type}_{environment}/CustomHopper-udr-v0_{i}.zip", env=env)
        elif environment == "cheetah":
            if model_type == "adr":
                model = SAC.load(f"models/{model_type}_{environment}/CustomHalfCheetah-adr-source-v3_{i}.zip", env=env)
            elif model_type == "udr":
                model = SAC.load(f"models/{model_type}_{environment}/CustomHalfCheetah-source-v3_{i}.zip", env=env)
        
        rewards, eps_length = evaluate_policy(model, env, n_eval_episodes=test_episodes, return_episode_rewards=True)
        #Print stats about single model performance
        print(f"Model: {i} Test reward (avg +/- std): ({np.mean(rewards):.2f} +/- {np.std(rewards):.2f}) - Num episodes: {test_episodes} - Eps. length {np.mean(eps_length):.2f}")
        all_rewards.extend(rewards)
        all_eps.extend(eps_length)
        env.close() 

    # Create a DataFrame
    data = {
        "reward": all_rewards,
        "eps_length": all_eps,
    }
    df = pd.DataFrame(data)

    if model_type == "source":
        output_file = f"test_results/{environment}_source_target_performance_data.csv"
    elif model_type == "adr":
        output_file = f"test_results/{environment}_source_adr_target_performance_data.csv"
    elif model_type == "target": 
        output_file = f"test_results/{environment}_target_target_performance_data.csv"
    elif model_type == "udr": 
        output_file = f"test_results/{environment}_source_udr_performance_data.csv"
    df.to_csv(output_file,index=False)

    # Calculate performance metrics
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    median_reward = np.median(all_rewards)
    best_reward = np.max(all_rewards)
    worst_reward = np.min(all_rewards)

    mean_episode_length = np.mean(all_eps)
    std_episode_length = np.std(all_eps)
    median_episode_length = np.median(all_eps)
    best_episode_length = np.max(all_eps)
    worst_episode_length = np.min(all_eps) 

    # Print results
    print(f"Reward Metrics:")
    print(f"- Mean: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"- Median: {median_reward:.2f}")
    print(f"- Best: {best_reward:.2f}")
    print(f"- Worst: {worst_reward:.2f}")

    print(f"Episode Length Metrics:")
    print(f"- Mean: {mean_episode_length:.2f} +/- {std_episode_length:.2f}")
    print(f"- Median: {median_episode_length:.2f}")
    print(f"- Best (Longest): {best_episode_length:.2f}")
    print(f"- Worst (Shortest): {worst_episode_length:.2f}")

if __name__ == "__main__":
    args = parse_args()

    if args.model not in ["source", "target", "udr", "adr"]:
        print("ERROR: model accepted values: source, target, udr, adr")
        sys.exit()
    if args.env not in ["hopper", "cheetah"]:
        print("ERROR: env accepted values: hopper, cheetah")
        sys.exit()
    if args.env == "cheetah" and (args.model == "source" or args.model == "target"):
        print("ERROR: unsupported combination only cheetah-adr, cheetah-udr")
        sys.exit()

    tester(args.model, args.env, args.test_episodes, args.num_runs)
       

