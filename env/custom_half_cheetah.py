"""Implementation of the HalfCheetah environment supporting
domain randomization optimization."""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from scipy.stats import truncnorm

class HalfCheetah(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None, ADR = False, UDR = False, xml_file = "assets/half_cheetah.xml"):
        self.ctrl_cost_weight=0.1
        self.forward_reward_weight=1.0
        MujocoEnv.__init__(self, 1, xml_file)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses
        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.sim.model.body_mass[1] -= 1.0
        self.ADR = ADR
        self.UDR = UDR
        self.masses_bounds = [(0.5 * mass, 1.5 * mass) for mass in self.get_parameters()]
        
        


    def update_bounds(self, i, bound):
        """
        Update the bounds for the mass at the specified index.

        Args:
            i (int): The index of the mass whose bounds are to be updated.
            bound (tuple): A tuple representing the new bounds for the mass.
        """
        self.masses_bounds[i] = bound 
    
    def set_random_parameters(self):
        """ Set random masses """
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """ Sample masses according to a domain randomization distribution """
        sampled_parameters = np.zeros(len(self.masses_bounds))
        for i in range(len(self.masses_bounds)):
            if i != 0:
                lower = self.masses_bounds[i][0]
                upper = self.masses_bounds[i][1]
                sampled_parameters[i] = np.random.uniform(lower, upper)
            else: 
                sampled_parameters[i] = self.sim.model.body_mass[1] #Keep the torso mass fixed
        return sampled_parameters

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses

    def set_parameters(self, task):
        """Set each HalfCheetah link's mass to a new value"""
        self.sim.model.body_mass[1:] = task

    def control_cost(self, action):
        control_cost = self.ctrl_cost_weight * np.sum(np.square(action))
        return control_cost
    
    def step(self, action):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self.forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }
        return observation, reward, done, {}

        


    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        if self.ADR == True or self.UDR == True:
            self.set_random_parameters()
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
    

"""
    Registered environments
"""
gym.envs.register(
        id="HalfCheetah-v0",
        entry_point="%s:HalfCheetah" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="HalfCheetah-source-v3",
        entry_point="%s:HalfCheetah" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="HalfCheetah-target-v3",
        entry_point="%s:HalfCheetah" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

gym.envs.register(
        id="HalfCheetah-source-udr-v3",
        entry_point="%s:HalfCheetah" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source", "UDR" : True, "ADR" : False}
)

gym.envs.register(
        id="HalfCheetah-source-adr-v3",
        entry_point="%s:HalfCheetah" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source", "UDR" : False, "ADR" : True}
)

