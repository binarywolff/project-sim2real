from env.custom_half_cheetah import *
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from automatic_domain_randomization import ADR, ADRCallback
from stable_baselines3.common.monitor import Monitor
from util import plot_train_results
import os

if __name__ == "__main__":
    env_id = "HalfCheetah-source-adr-v3"
    env = gym.make(env_id)
    log_dir = f"./tmp/gym/train/{env_id}"
    os.makedirs(log_dir, exist_ok=True)

    #Initial ADR parameters
    min_max_bounds = [(1, 10) for _ in env.get_parameters()]
    masses_bounds = [(0.5 * mass, 1.5 * mass) for mass in env.get_parameters()] #Initial range between 50% and 150% of original masses
    thresholds = (700, 1600)
    delta = 0.01 #Update step size 
    m = 20 #Buffer size
    fixed_torso_mass = env.get_parameters()[0]
    
    env = Monitor(env, log_dir)

    adr_env = gym.make(env_id)
    adr_env = DummyVecEnv([lambda: adr_env])

    adr = ADR(masses_bounds, thresholds, delta, m, min_max_bounds, adr_env, fixed_torso_mass)
    adr_callback = ADRCallback(adr, env, f'entropy_log_cheetah_adr.csv')

    model = SAC("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=200_000, callback=adr_callback, progress_bar=True)
    model.save(f"models/SAC_ADR_{env_id}")
    plot_train_results(log_dir, env_id, save_path_figure="training_results/SAC_ADR_")
    