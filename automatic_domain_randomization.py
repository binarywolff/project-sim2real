from env.custom_hopper import *
from stable_baselines3.common.callbacks import BaseCallback

# Class to implement Automatic Domain Randomization (ADR) for adaptive parameter adjustment
class ADR:
    def __init__(self, d_bounds, thresholds, delta, m, min_max_bounds, adr_env):
        """
        Initialize the ADR class with necessary parameters and buffers.

        Args:
            d_bounds (list): Initial bounds for the domain parameters.
            thresholds (tuple): Performance thresholds (low and high).
            delta (float): Increment or decrement for the parameter bounds.
            m (int): Number of evaluations required before adjusting bounds.
            min_max_bounds (list): Absolute min and max bounds for parameters.
            adr_env: The environment instance for which ADR is applied.
        """
        self.d_bounds = d_bounds
        self.th_low, self.th_high = thresholds
        self.delta = delta
        self.m = m
        self.lower_buffers = {i: [] for i in range(len(d_bounds))}
        self.upper_buffers = {i: [] for i in range(len(d_bounds))}
        self.min_max_bounds = min_max_bounds
        self.env = adr_env

    def sample_parameters(self):
        """
        Randomly sample parameters, applying boundary sampling to one of the parameter at a time.

        Returns:
            i (int): Index of the selected parameter.
            updated_parameters (list): New parameter values.
            bound (str): Boundary type ('lower' or 'upper').
        """
        i = np.random.randint(1, len(self.d_bounds)) # Change 1 to 0 if you want to randomize all parameters, in this case the torso mass (index 0) is always fixed.
        x = np.random.uniform(0, 1)
        if x < 0.5:
            boundary_value =  self.d_bounds[i][0] #Lower boundary 
            bound = "lower"
        else: 
            boundary_value = self.d_bounds[i][1] #Upper boundary
            bound = "upper"

        # Generate updated parameters with the selected boundary
        updated_parameters = np.zeros(len(self.d_bounds))
        updated_parameters[i] = boundary_value
        for j in range(len(self.d_bounds)):
            if j != i:
                low = self.d_bounds[j][0]
                high = self.d_bounds[j][1]
                updated_parameters[j] = np.random.uniform(low, high)

        return i, updated_parameters, bound
        
    def evaluate_performance(self, model, updated_parameters):
        """
        Evaluate the performance of the model with the updated parameters.

        Args:
            model: The trained model.
            updated_parameters (list): Updated environment parameters.

        Returns:
            total_reward (float): Total reward achieved during evaluation.
        """
        obs = self.env.reset()
        self.env.env_method('set_parameters',updated_parameters)
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)  # Predict action using the model
            obs, reward, done, _ = self.env.step(action) # Take a step in the environment
            total_reward += reward # Accumulate reward
        return total_reward 
    
    def update_phi(self, i, performance, bound):
        """
        Update the parameter bounds based on performance evaluation.

        Args:
            i (int): Index of the parameter.
            performance (float): Performance score.
            bound (str): Boundary type ('lower' or 'upper').

        Returns:
            updated (bool): Whether the bounds were updated.
            i (int): Index of the parameter updated.
            new_bound (list): Updated bounds for the parameter.
        """
        updated = False
        new_bound = []
        buffer = self.lower_buffers[i] if bound == "lower" else self.upper_buffers[i]
        buffer.append(performance)
        
        # Check if enough evaluations are collected
        if len(buffer) > self.m:
            avg_p = np.mean(buffer) # Calculate average performance
            buffer.clear()

            # Update bounds if average performance crosses thresholds
            if avg_p > self.th_high:
                updated = True
                if bound == "lower": 
                    updated_value = self.d_bounds[i][0] - self.delta #Increase lower bound 
                    updated_value = updated_value if updated_value > self.min_max_bounds[i][0] else self.min_max_bounds[i][0] #Check if the lower bound is higher than the minimum accepted value
                    self.d_bounds[i] = [updated_value, self.d_bounds[i][1]]
                else: 
                    updated_value = self.d_bounds[i][1] + self.delta #Increase upper bound 
                    updated_value = updated_value if updated_value < self.min_max_bounds[i][1] else self.min_max_bounds[i][1] #Check if the upper bound is lower than the maximum accepted value
                    self.d_bounds[i] = [self.d_bounds[i][0], updated_value]
                new_bound = [self.d_bounds[i][0], self.d_bounds[i][1]]
                self.env.env_method('update_bounds', i, new_bound)
            elif avg_p < self.th_low:
                updated = True 
                if bound == "lower": 
                    updated_value = self.d_bounds[i][0] + self.delta #Decrease lower bound 
                    updated_value = updated_value if updated_value < self.d_bounds[i][1] else self.d_bounds[i][1] # Check that low_b < up_b 
                    self.d_bounds[i] = [updated_value, self.d_bounds[i][1]]
                else: 
                    updated_value = self.d_bounds[i][1] - self.delta #Decrease upper bound 
                    updated_value = updated_value if updated_value > self.d_bounds[i][0] else self.d_bounds[i][0] # Check that up_b > low_b 
                    self.d_bounds[i] = [self.d_bounds[i][0], updated_value]
                new_bound = [self.d_bounds[i][0], self.d_bounds[i][1]]
                self.env.env_method('update_bounds', i, new_bound)
                
        return updated, i, new_bound

# Custom callback for integrating ADR with Stable-Baselines3 
class ADRCallback(BaseCallback):
    def __init__(self, adr, env, boundary_sampling_probability, verbose=0):
        super().__init__(verbose)
        """
        Initialize the ADR callback.

        Args:
            adr (ADR): ADR instance.
            env: Environment instance.
            boundary_sampling_probability (float): Probability of sampling a boundary.
            verbose (int): Verbosity level (0 for silent, 1 for logging).
        """
        self.adr = adr
        self.env = env
        self.boundary_sampling_probability = boundary_sampling_probability
        self.verbose = verbose
        self.bound_to_update = None
        self.need_to_update = False
        self.index_to_update = None 

    def _on_step(self) -> bool:
        """
        Custom logic executed at each training step.

        Returns:
            bool: True to continue training.
        """
        infos = self.locals.get("infos", None)
        if "episode" in infos[0]:
            episode_end = True 
            if self.need_to_update: #Update bounds at the end of the episode if necessary
                self.env.update_bounds(self.index_to_update, self.bound_to_update)
                self.need_to_update = False
                self.bound_to_update = None
                self.index_to_update = None
        else:
            episode_end = False
        if np.random.rand() < self.boundary_sampling_probability:   
            i, updated_parameters, bound = self.adr.sample_parameters()
            performance = self.adr.evaluate_performance(self.model, updated_parameters)
            updated, j, new_bound = self.adr.update_phi(i, performance, bound)
            if updated:
                if episode_end: #Episode ended, update the bound
                    self.env.update_bounds(j, new_bound)
                else:   #Episode NOT ended, save the bound to be updated when the episode ends
                    self.need_to_update = True
                    self.bound_to_update = new_bound
                    self.index_to_update = j
            if self.verbose == 1:
                print(f"Performance Evaluation - Step: {self.num_timesteps}, Performance: {performance}")
        return True