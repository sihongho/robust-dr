import numpy as np
import logging

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
        # Ensure NumPy arrays are printed in full
        np.set_printoptions(threshold=np.inf, suppress=True, precision=6)

        uncertainty_set = []
        
        # Initialize accumulators for kernels and rewards
        total_kernels = np.zeros_like(self.nominal_kernels)
        total_rewards = np.zeros_like(self.nominal_rewards)

        for i in range(num_mdps):
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

            # Log details of the current MDP in the uncertainty set
            logging.info(f"Uncertainty Set MDP {i+1}:")
            logging.info(f"Kernels:\n{uncertain_kernels}")
            logging.info(f"Rewards:\n{self.nominal_rewards}")
            logging.info("-" * 50)
        
        # Compute the averages
        average_kernels = total_kernels / num_mdps
        average_rewards = total_rewards / num_mdps

        # Create an Environment object for the average MDP
        average_env = Environment(self.state_count, self.action_count)
        average_env.kernels = average_kernels
        average_env.rewards = average_rewards

        # Log the average MDP details
        logging.info("Average MDP:")
        logging.info(f"Kernels:\n{average_kernels}")
        logging.info(f"Rewards:\n{average_rewards}")
        logging.info("=" * 50)

        return uncertainty_set, average_env
    
class RobotEnvironment(Environment):
    def __init__(self, alpha, beta, seed=None):
        # Constants for number of states and actions
        self.state_count = 3  # States: 'high', 'low', 'dead'
        self.action_count = 2  # Actions: 'search', 'wait'

        # Constants for transition probabilities
        self.alpha = alpha  # Pr(stay at high charge if searching | now have high charge)
        self.beta = beta   # Pr(stay at low charge if searching | now have low charge)

        # Constants for rewards
        self.r_search = 50   # reward for searching
        self.r_wait = 10     # reward for waiting
        self.r_dead = 0      # reward (actually penalty) for dead

        super().__init__(self.state_count, self.action_count, seed)

    def _generate_transition_kernels(self):
        """Generate transition probability kernels for all states and actions."""
        kernels = np.zeros((self.state_count, self.action_count, self.state_count))

        # Action 'search' (action index 0)
        kernels[0, 0, 0] = self.alpha     # High to High
        kernels[0, 0, 1] = 1 - self.alpha # High to Low
        kernels[0, 0, 2] = 0              # High to Dead
        kernels[1, 0, 0] = 0              # Low to High
        kernels[1, 0, 1] = self.beta      # Low to Low
        kernels[1, 0, 2] = 1 - self.beta  # Low to Dead
        kernels[2, 0, 2] = 1              # Dead to Dead

        # Action 'wait' (action index 1)
        kernels[0, 1, 0] = 1  # High stays High
        kernels[1, 1, 1] = 1  # Low stays Low
        kernels[2, 1, 2] = 1  # Dead stays Dead

        return kernels
    
    def _generate_rewards(self):
        """Generate reward matrix for all states and actions."""
        rewards = np.zeros((self.state_count, self.action_count))

        # Rewards for actions
        rewards[0, 0] = self.r_search  # High state, search
        rewards[1, 0] = self.r_search  # Low state, search
        rewards[0, 1] = self.r_wait    # High state, wait
        rewards[1, 1] = self.r_wait    # Low state, wait
        rewards[2, 0] = self.r_dead    # Dead state, search
        rewards[2, 1] = self.r_dead    # Dead state, wait
        return rewards
    
    def copy(self):
        new_env = RobotEnvironment(self.alpha, self.beta)
        new_env.kernels = np.copy(self.kernels)
        new_env.rewards = np.copy(self.rewards)
        new_env.nominal_kernels = np.copy(self.nominal_kernels)
        new_env.nominal_rewards = np.copy(self.nominal_rewards)
        return new_env
    
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
        # Ensure NumPy arrays are printed in full
        np.set_printoptions(threshold=np.inf, suppress=True, precision=6)

        uncertainty_set = []
        
        # Initialize accumulators for kernels and rewards
        total_kernels = np.zeros_like(self.nominal_kernels)
        total_rewards = np.zeros_like(self.nominal_rewards)
        total_alpha = 0
        total_beta = 0

        for i in range(num_mdps):
            # Create a new perturbed kernel
            uncertain_kernels = np.zeros_like(self.nominal_kernels)
            new_alpha = R + np.random.uniform(-bias, bias)
            new_beta = R + np.random.uniform(-bias, bias)
            # Action 'search' (action index 0)
            uncertain_kernels[0, 0, 0] = new_alpha     # High to High
            uncertain_kernels[0, 0, 1] = 1 - new_alpha # High to Low
            uncertain_kernels[0, 0, 2] = 0              # High to Dead
            uncertain_kernels[1, 0, 0] = 0              # Low to High
            uncertain_kernels[1, 0, 1] = new_beta      # Low to Low
            uncertain_kernels[1, 0, 2] = 1 - new_beta  # Low to Dead
            uncertain_kernels[2, 0, 2] = 1              # Dead to Dead
            # Action 'wait' (action index 1)
            uncertain_kernels[0, 1, 0] = 1  # High stays High
            uncertain_kernels[1, 1, 1] = 1  # Low stays Low
            uncertain_kernels[2, 1, 2] = 1  # Dead stays Dead
            
            # Accumulate the perturbed kernels and rewards
            total_kernels += uncertain_kernels
            total_rewards += self.nominal_rewards  # Rewards are not perturbed
            total_alpha += new_alpha
            total_beta += new_beta
            
            # Create a new Environment with the perturbed kernel and original rewards
            new_env = RobotEnvironment(new_alpha, new_beta)
            new_env.kernels = uncertain_kernels  # Assign the perturbed kernels
            new_env.rewards = self.nominal_rewards.copy()  # Use the same reward structure
            uncertainty_set.append(new_env)

            # Log details of the current MDP in the uncertainty set
            logging.info(f"Uncertainty Set MDP {i+1}:")
            logging.info(f"Kernels:\n{uncertain_kernels}")
            logging.info(f"Rewards:\n{self.nominal_rewards}")
            logging.info("-" * 50)
        
        # Compute the averages
        average_kernels = total_kernels / num_mdps
        average_rewards = total_rewards / num_mdps
        average_alpha = total_alpha / num_mdps
        average_beta = total_beta / num_mdps

        # Create an Environment object for the average MDP
        average_env = RobotEnvironment(average_alpha, average_beta)
        average_env.kernels = average_kernels
        average_env.rewards = average_rewards

        # Log the average MDP details
        logging.info("Average MDP:")
        logging.info(f"Kernels:\n{average_kernels}")
        logging.info(f"Rewards:\n{average_rewards}")
        logging.info("=" * 50)

        return uncertainty_set, average_env
    
class InventoryEnvironment(Environment):
    def __init__(self, state_count, action_count, max_demand=29, demand_prob=None, seed=None):
        self.states = np.arange(0, state_count)     # Inventory levels from 0 to state_count.
        self.actions = np.arange(0, action_count)   # Possible orders from 0 to action_count units.

        self.max_demand = max_demand
        if demand_prob is not None:
            self.demand_prob = demand_prob
        else:
            self.demand_prob = self._generate_demand_distribution()

        # Costs and rewards
        self.order_cost = 3
        self.holding_cost = 3
        self.sale_price = 5
        self.penalty_cost = -15

        super().__init__(state_count, action_count, seed)

    def _generate_transition_kernels(self):
        """Generate transition probability kernels for all states and actions."""
        kernels = np.zeros((self.state_count, self.action_count, self.state_count))
        for s in range(self.state_count):
            for a in range(self.action_count):
                for d in range(self.max_demand + 1):
                    next_state = max(0, s + a - d)  # Calculate the next state after demand is met
                    if next_state < self.state_count:
                        kernels[s, a, next_state] += self.demand_prob[d]
        return kernels
    
    def _generate_rewards(self):
        """Generate reward matrix for all states and actions."""
        rewards = np.zeros((self.state_count, self.action_count))
        for s in range(self.state_count):
            for a in range(self.action_count):
                expected_reward = 0
                for d in range (self.max_demand + 1):
                    if d <= s + a:
                        reward = (self.sale_price * d) - (self.holding_cost * (s + a))
                    else:
                        reward = self.penalty_cost
                    expected_reward += reward * self.demand_prob[d]
                expected_reward -= self.order_cost * a
                rewards[s, a] = expected_reward
        return rewards

    def _generate_demand_distribution(self):
        """Generate a random demand probability distribution and normalize it."""
        demand_prob = np.random.rand(self.max_demand + 1)
        return demand_prob / demand_prob.sum()  # Normalize to make a valid probability distribution

    def copy(self):
        new_env = InventoryEnvironment(self.state_count, self.action_count, self.max_demand, self.demand_prob)
        new_env.kernels = np.copy(self.kernels)
        new_env.rewards = np.copy(self.rewards)
        new_env.nominal_kernels = np.copy(self.nominal_kernels)
        new_env.nominal_rewards = np.copy(self.nominal_rewards)
        return new_env
    
    def create_uncertainty_set(self, R, bias=0, num_mdps=10):
        uncertainty_set = []
        total_demand_prob = np.zeros_like(self.demand_prob)
        for i in range(num_mdps):
            uncertain_demand_prob = self._add_perturbation(self.demand_prob, R, bias)
            total_demand_prob += uncertain_demand_prob
            new_env = InventoryEnvironment(self.state_count, self.action_count, self.max_demand, uncertain_demand_prob)
            uncertainty_set.append(new_env)

            # Log details of the current MDP in the uncertainty set
            logging.info(f"Uncertainty Set MDP {i+1}:")
            logging.info(f"Demand prob:\n{uncertain_demand_prob}")
            logging.info(f"Kernels:\n{new_env.kernels}")
            logging.info(f"Rewards:\n{new_env.rewards}")
            logging.info("-" * 50)

        average_demand_prob = total_demand_prob / num_mdps
        average_env = InventoryEnvironment(self.state_count, self.action_count, self.max_demand, average_demand_prob)

        # Log the average MDP details
        logging.info("Average MDP:")
        logging.info(f"Demand prob:\n{average_demand_prob}")
        logging.info(f"Kernels:\n{average_env.kernels}")
        logging.info(f"Rewards:\n{average_env.rewards}")
        logging.info("=" * 50)

        return uncertainty_set, average_env
    
class GamblerEnvironment(Environment):
    def __init__(self, state_count, head_prob, seed=None):
        self.state_count = state_count
        self.action_count = state_count
        self.goal = state_count - 1
        self.head_prob = head_prob

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
        kernels = np.zeros((self.state_count, self.state_count, self.state_count))
        for s in range(1, self.goal):
            max_bet = min(s, self.goal - s)
            for a in range(1, max_bet + 1):
                win_state = s + a
                lose_state = s - a
                kernels[s, a, win_state] = self.head_prob
                kernels[s, a, lose_state] = 1 - self.head_prob

        kernels[0, :, 0] = 1.0  # If bankrupt, stay bankrupt
        kernels[self.goal, :, self.goal] = 1.0  # If reached goal, stay
        return kernels
    
    def _generate_rewards(self):
        rewards = np.zeros((self.state_count, self.action_count))
        rewards[self.goal, :] = 1
        return rewards
    
    def step(self, state, action):
        if state == 0 or state == self.goal:
            return state, 0  # Terminal states
        
        max_bet = min(state, self.goal_state - state)
        if action < 1 or action > max_bet:
            raise ValueError(f"Invalid bet amount: {action} at state {state}. Must be in [1, {max_bet}].")

        win = np.random.rand() < self.head_prob
        next_state = state + action if win else state - action
        next_state = max(0, min(next_state, self.goal))  # Ensure within bounds

        return next_state, self.rewards[state, action]
    
    def _add_perturbation(self, kernels, R, bias=0):
        # Generate noise with symmetric distribution
        noise = np.random.uniform(-R, R, kernels.shape)

        # Add bias to introduce asymmetry
        biased_noise = noise + bias

        # Apply the biased noise to nominal probabilities
        perturbed_kernels = kernels + biased_noise

        # Ensure probabilities are non-negative and renormalize
        perturbed_kernels = np.clip(perturbed_kernels, 0, None)  # Clip to non-negative

        for s in range(self.state_count):
            for a in range(self.state_count):
                if a > min(s, self.goal - s):
                    perturbed_kernels[s, a] = 0
                else:
                    if perturbed_kernels[s, a].sum() > 0:
                        perturbed_kernels[s, a] /= perturbed_kernels[s, a].sum()

        return perturbed_kernels
    
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
        # Ensure NumPy arrays are printed in full
        np.set_printoptions(threshold=np.inf, suppress=True, precision=6)

        uncertainty_set = []
        
        # Initialize accumulators for kernels and rewards
        total_kernels = np.zeros_like(self.nominal_kernels)
        total_rewards = np.zeros_like(self.nominal_rewards)

        for i in range(num_mdps):
            # Create a new perturbed kernel
            uncertain_kernels = self._add_perturbation(self.nominal_kernels, R, bias)
            
            # Accumulate the perturbed kernels and rewards
            total_kernels += uncertain_kernels
            
            # Create a new Environment with the perturbed kernel and original rewards
            new_env = GamblerEnvironment(self.state_count, self.head_prob)
            new_env.kernels = uncertain_kernels  # Assign the perturbed kernels
            uncertainty_set.append(new_env)

            # Log details of the current MDP in the uncertainty set
            logging.info(f"Uncertainty Set MDP {i+1}:")
            logging.info(f"Kernels:\n{uncertain_kernels}")
            logging.info(f"Rewards:\n{self.nominal_rewards}")
            logging.info("-" * 50)
        
        # Compute the averages
        average_kernels = total_kernels / num_mdps

        # Create an Environment object for the average MDP
        average_env = GamblerEnvironment(self.state_count, self.head_prob)
        average_env.kernels = average_kernels

        # Log the average MDP details
        logging.info("Average MDP:")
        logging.info(f"Kernels:\n{average_kernels}")
        logging.info(f"Rewards:\n{average_env.rewards}")
        logging.info("=" * 50)

        return uncertainty_set, average_env

