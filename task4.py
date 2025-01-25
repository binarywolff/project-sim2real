import gym 
from env.custom_hopper import *
import os 
from stable_baselines3.common.monitor import Monitor


from util import train, test

def main():

    #Training on SOURCE using UDR
    print("Training on SOURCE using UDR")
    adr_env = gym.make('CustomHopper-udr-v0')
    log_dir = f"./tmp/gym/train/CustomHopper-udr-v0"
    os.makedirs(log_dir, exist_ok=True)
    adr_env = Monitor(adr_env, log_dir)
    train(adr_env, seed=None, total_timesteps=20_000, log_dir=log_dir, env_id='CustomHopper-udr-v0', algo='ppo')

    # Testing SOURCE+UDR model on SOURCE domain
    print("Testing source_udr->source configuration")
    source_env = gym.make('CustomHopper-source-v0')
    log_dir = f"./tmp/gym/test/CustomHopper-source-v0"
    os.makedirs(log_dir, exist_ok=True)
    source_env = Monitor(source_env, log_dir)
    test(source_env, model_file="models/PPO_CustomHopper-udr-v0.zip", render=False, test_episodes=50, algo='ppo', scenario="source_udr_source")

    # Testing SOURCE+UDR model on TARGET domain
    print("Testing source_udr->target configuration")
    target_env = gym.make('CustomHopper-target-v0')
    log_dir = f"./tmp/gym/test/CustomHopper-target-v0"
    os.makedirs(log_dir, exist_ok=True)
    target_env = Monitor(target_env, log_dir)
    test(target_env, model_file="models/PPO_CustomHopper-udr-v0.zip", render=False, test_episodes=50, algo='ppo', scenario="source_udr_target")

if __name__ == "__main__":
    main()