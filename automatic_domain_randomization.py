from env.custom_hopper import *
from stable_baselines3.common.callbacks import BaseCallback

# Class to implement Automatic Domain Randomization (ADR) for adaptive parameter adjustment
class ADR:
    def __init__(self, d_bounds, thresholds, delta, m, min_max_bounds, adr_env, fixed_torso_mass):
        """
        Initialize the ADR class with necessary parameters and buffers.

        Args:
            d_bounds (list): Initial bounds for the domain parameters.
            thresholds (tuple): Performance thresholds (low and high) to adjust parameter bound.
            delta (float): Step size for adjusting parameter bounds.
            m (int): Number of evaluations required before adjusting bounds.
            min_max_bounds (list): Absolute min and max bounds for parameters.
            adr_env: The environment instance for which ADR is applied.
            fixed_torso_mass (float): Fixed value for the torso mass, which is not randomized.
        """
        self.d_bounds = d_bounds
        self.th_low, self.th_high = thresholds
        self.delta = delta
        self.m = m
        self.lower_buffers = {i: [] for i in range(len(d_bounds))}
        self.upper_buffers = {i: [] for i in range(len(d_bounds))}
        self.min_max_bounds = min_max_bounds
        self.env = adr_env
        self.fixed_torso_mass = fixed_torso_mass

    def sample_parameters(self):
        """
        Randomly sample parameters for the environment. Only one parameter is sampled at its boundary,
        while others are randomized within their current bounds.

        Returns:
            i (int): Index of the selected parameter to sample at its boundary.
            updated_parameters (list): Parameter values for the environment.
            bound (str): Boundary type ('lower' or 'upper') for the sampled parameter.
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
        updated_parameters[0] = self.fixed_torso_mass # Torso mass remains fixed
        updated_parameters[i] = boundary_value
        for j in range(len(self.d_bounds)):
            if j != i and j != 0: # Randomize other parameters within their bounds
                low = self.d_bounds[j][0]
                high = self.d_bounds[j][1]
                updated_parameters[j] = np.random.uniform(low, high)
        return i, updated_parameters, bound
        
    def evaluate_performance(self, model, updated_parameters, eps_rewards):
        """
        Evaluate the agent's performance on the environment with updated parameters.

        Args:
            model: The trained agent.
            updated_parameters (list): Parameters to set in the environment.
            eps_rewards (list): List to store episode rewards, containing the rewards from 5 training episodes. 

        Returns:
            float: Average episode reward across evaluations.
        """
        obs = self.env.reset()
        self.env.env_method('set_parameters',updated_parameters)
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)  # Predict action using the model
            obs, reward, done, _ = self.env.step(action) # Take a step in the environment
            total_reward += reward # Accumulate reward
        eps_rewards.append(total_reward.item())
        return np.mean(eps_rewards) 
    
    def update_phi(self, i, performance, bound):
        """
        Update the bounds of the selected parameter based on the agent's performance.

        Args:
            i (int): Index of the parameter to adjust.
            performance (float): Performance score of the agent.
            bound (str): Boundary type ('lower' or 'upper') being evaluated.
        
        Returns:
            bool: True if the bounds were updated, False otherwise.
        """
        buffer = self.lower_buffers[i] if bound == "lower" else self.upper_buffers[i]
        buffer.append(performance)
        
        # Check if enough evaluations are collected
        if len(buffer) > self.m:
            avg_p = np.mean(buffer) # Calculate average performance
            if bound == "lower":
                self.lower_buffers[i].clear()
            else:
                self.upper_buffers[i].clear()

            # Adjust bounds based on average performance relative to thresholds
            if avg_p > self.th_high:
                if bound == "lower": 
                    updated_value = self.d_bounds[i][0] - self.delta #Increase lower bound 
                    updated_value = updated_value if updated_value > self.min_max_bounds[i][0] else self.min_max_bounds[i][0] #Check if the lower bound is higher than the minimum accepted value
                    self.d_bounds[i] = [updated_value, self.d_bounds[i][1]]
                else: 
                    updated_value = self.d_bounds[i][1] + self.delta #Increase upper bound 
                    updated_value = updated_value if updated_value < self.min_max_bounds[i][1] else self.min_max_bounds[i][1] #Check if the upper bound is lower than the maximum accepted value
                    self.d_bounds[i] = [self.d_bounds[i][0], updated_value]
                return True
            elif avg_p < self.th_low:
                if bound == "lower": 
                    updated_value = self.d_bounds[i][0] + self.delta #Decrease lower bound 
                    updated_value = updated_value if updated_value < self.d_bounds[i][1] else self.d_bounds[i][1] # Check that low_b < up_b 
                    self.d_bounds[i] = [updated_value, self.d_bounds[i][1]]
                else: 
                    updated_value = self.d_bounds[i][1] - self.delta #Decrease upper bound 
                    updated_value = updated_value if updated_value > self.d_bounds[i][0] else self.d_bounds[i][0] # Check that up_b > low_b 
                    self.d_bounds[i] = [self.d_bounds[i][0], updated_value]
                return True
        return False 
    
    def entropy(self):
        """
        Calculate the average entropy of the domain parameter bounds.

        Returns:
            float: Average entropy across all parameters.
        """
        eps = 1e-12 #used to prevent numerical issue
        ranges = [np.log10(bound[1] - bound[0] + eps) for bound in self.d_bounds]
        return np.mean(ranges)

# Custom callback for integrating ADR with Stable-Baselines3 
class ADRCallback(BaseCallback):
    def __init__(self, adr, env, log_dir, verbose=0):
        super().__init__(verbose)
        """
        Initialize the ADR callback.

        Args:
            adr (ADR): Instance of the ADR class.
            env: Environment instance.
            log_dir (str): Directory path to save logs.
            verbose (int): Verbosity level (0 for silent, 1 for logging).
        """
        self.adr = adr
        self.env = env
        self.eps_count = 0
        self.eps_rewards = []

        self.current_parameter = None
        self.current_bound = None
        self.current_sampled_parameters = None 

        self.prev_parameter = None
        self.prev_bound = None
        self.prev_sampled_parameters = None 

        self.log_file = log_dir
        
        # Initialize CSV writer for logging entropy
        self.csv_file = open(self.log_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['episode', 'entropy'])  # Headers for the CSV
        self.count = 0 

    def _on_step(self) -> bool:
        """
        Custom logic executed at each training step.

        Returns:
            bool: True to continue training.
        """
        infos = self.locals.get("infos", None) 
        
        # Execute logic only at the end of an episode
        if "episode" in infos[0]:
            self.count += 1
            if self.count == 1:
                # Initialize environment with sampled parameters for the first episode
                i, updated_parameters, bound = self.adr.sample_parameters()
                self.env.set_parameters(updated_parameters)
                self.current_parameter = i
                self.current_bound = bound
                self.current_sampled_parameters = updated_parameters
            else: 
                self.eps_count += 1 
                if self.eps_count >= 5: #Boundary sampling probability (every 5 episodes)
                    self.eps_count = 0
                    self.eps_rewards.append(infos[0]["episode"]['r'])

                    # Sample new environment parameters
                    i, updated_parameters, bound = self.adr.sample_parameters()
                    self.env.set_parameters(updated_parameters)

                     # Update previous state information
                    self.prev_bound = self.current_bound
                    self.prev_parameter = self.current_parameter
                    self.prev_sampled_parameters = self.current_sampled_parameters
                    
                    # Update current state information
                    self.current_parameter = i
                    self.current_bound = bound
                    self.current_sampled_parameters = updated_parameters

                    # Evaluate performance and update bounds
                    performance = self.adr.evaluate_performance(self.model, self.prev_sampled_parameters, self.eps_rewards.copy())
                    self.eps_rewards.clear()
                    updated = self.adr.update_phi(self.prev_parameter, performance, self.prev_bound)
                    if updated: 
                        entropy = self.adr.entropy()
                        self.csv_writer.writerow([self.count, entropy])

                else: 
                    self.eps_rewards.append(infos[0]["episode"]['r'])
            
        return True