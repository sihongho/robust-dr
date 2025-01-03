import numpy as np

class Environment:
    def __init__(self, state_count, action_count, seed=None):
        self.state_count = state_count
        self.action_count = action_count

        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Generate nominal MDP
        self.kernels = self._generate_transition_kernels()
        self.rewards = self._generate_rewards()

        # Save nominal MDP
        self.nominal_kernels = self.kernels.copy()
        self.nominal_rewards = self.rewards.copy()

    def _generate_transition_kernels(self):
        """Generate transition probability kernels for all states and actions."""
        kernels = np.zeros((self.state_count, self.action_count, self.state_count))
        for s in range(self.state_count):
            for a in range(self.action_count):
                # Generate positive probabilities and normalize
                raw_probs = np.abs(np.random.normal(1, np.random.random()**2, self.state_count))
                kernels[s, a] = self._normalize(raw_probs)
        return kernels

    def _generate_rewards(self):
        """Generate reward matrix for all states and actions."""
        rewards = np.zeros((self.state_count, self.action_count))
        for s in range(self.state_count):
            for a in range(self.action_count):
                # Clip rewards to a reasonable range for stability
                rewards[s, a] = np.clip(np.random.normal(1, np.random.random()**2), 0, 5)
        return rewards

    @staticmethod
    def _normalize(vector):
        """Normalize a vector to sum to 1."""
        vector = np.abs(vector)
        total = vector.sum()
        if total == 0:
            raise ValueError("Normalization error: Vector sum is zero.")
        return vector / total

    def step(self, state, action):
        """Return next state based on current state and action."""
        try:
            return np.random.choice(self.state_count, p=self.kernels[state, action])
        except ValueError as e:
            raise ValueError(f"Invalid transition probabilities for state {state} and action {action}: {e}")

    def get_reward(self, state, action):
        """Get reward for a given state and action."""
        return self.rewards[state, action]
    
    # Ziying
    def copy(self):
        new_env = Environment(self.state_count, self.action_count)
        new_env.kernels = np.copy(self.kernels)
        new_env.rewards = np.copy(self.rewards)
        new_env.nominal_kernels = np.copy(self.nominal_kernels)
        new_env.nominal_rewards = np.copy(self.nominal_rewards)
        return new_env
    
    def _add_perturbation(self, nominal_probs, R, bias=0):
        """
        Add a bounded, biased perturbation to the nominal probabilities.

        Args:
            nominal_probs (np.ndarray): The nominal probability distribution (1D array).
            R (float): Maximum perturbation radius for each probability element.
            bias (float): Bias to be added to the noise, introducing asymmetry.

        Returns:
            np.ndarray: Perturbed probability distribution.
        """
        # Generate noise with symmetric distribution
        noise = np.random.uniform(-R, R, nominal_probs.shape)

        # Add bias to introduce asymmetry
        biased_noise = noise + bias

        # Apply the biased noise to nominal probabilities
        perturbed_probs = nominal_probs + biased_noise

        # Ensure probabilities are non-negative and renormalize
        perturbed_probs = np.clip(perturbed_probs, 0, None)  # Clip to non-negative
        total = perturbed_probs.sum()
        if total == 0:
            # Fallback to nominal_probs if all probabilities are clipped
            perturbed_probs = nominal_probs
        else:
            perturbed_probs /= total  # Normalize to ensure sum = 1

        return perturbed_probs

    def create_uncertainty_set(self, R, bias=0, num_mdps=10):
        """
        Create an uncertainty set of MDPs based on the nominal MDP, 
        and compute the average perturbed kernels and rewards as a new MDP.

        Args:
            R (float): Maximum perturbation radius.
            bias (float): Bias to be added to the noise, introducing asymmetry.
            num_mdps (int): Number of MDPs in the uncertainty set.

        Returns:
            tuple: A tuple containing:
                - list[Environment]: A list of Environment objects representing the uncertainty set.
                - Environment: An Environment object representing the MDP with average kernels and rewards.
        """
        uncertainty_set = []
        
        # Initialize accumulators for kernels and rewards
        total_kernels = np.zeros_like(self.nominal_kernels)
        total_rewards = np.zeros_like(self.nominal_rewards)

        for _ in range(num_mdps):
            # Create a new perturbed kernel
            uncertain_kernels = np.zeros_like(self.nominal_kernels)
            for s in range(self.state_count):
                for a in range(self.action_count):
                    uncertain_kernels[s, a] = self._add_perturbation(self.nominal_kernels[s, a], R, bias)
            
            # Accumulate the perturbed kernels and rewards
            total_kernels += uncertain_kernels
            total_rewards += self.nominal_rewards  # Rewards are not perturbed
            
            # Create a new Environment with the perturbed kernel and original rewards
            new_env = Environment(self.state_count, self.action_count)
            new_env.kernels = uncertain_kernels  # Assign the perturbed kernels
            new_env.rewards = self.nominal_rewards.copy()  # Use the same reward structure
            uncertainty_set.append(new_env)
        
        # Compute the averages
        average_kernels = total_kernels / num_mdps
        average_rewards = total_rewards / num_mdps

        # Create an Environment object for the average MDP
        average_env = Environment(self.state_count, self.action_count)
        average_env.kernels = average_kernels
        average_env.rewards = average_rewards

        return uncertainty_set, average_env