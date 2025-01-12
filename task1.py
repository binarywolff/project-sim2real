"""Test a random policy on the Gym Hopper environment


    Play around with this code to get familiar with the
    Hopper environment.

    For example, what happens if you don't reset the environment
    even after the episode is over?
    When exactly is the episode over?
    What is an action here?
"""



import gym
from env.custom_hopper import *
env = gym.make('CustomHopper-source-v0')
# env = gym.make('CustomHopper-target-v0')

print('State space:', env.observation_space)  # state-space
print('Action space:', env.action_space)  # action-space
print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper

## TASK 1

n_episodes = 5

for episode in range(n_episodes):
  done = False
  observation = env.reset()	 # Reset environment to initial state

  while not done:  # Until the episode is over

    action = env.action_space.sample()	# Sample random action

    observation, reward, done, info = env.step(action)	# Step the simulator to the next timestep