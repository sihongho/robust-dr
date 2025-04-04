import numpy as np
import logging

class Environment:
    def __init__(self, state_count, action_count, seed=None):
        self.state_count = state_count
        self.action_count = action_count
        self.seed = seed

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
                # rewards[s, a] = np.clip(np.random.normal(1, np.random.random()**2), 0, 5)
                if s % 2 != 0:
                    rewards[s, a] = np.random.uniform(0, 0.2)  # Low reward
                else:
                    rewards[s, a] = np.random.uniform(0.8, 1.0)  # High reward
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
        new_env = Environment(self.state_count, self.action_count, self.seed)
        new_env.kernels = np.copy(self.kernels)
        new_env.rewards = np.copy(self.rewards)
        new_env.nominal_kernels = np.copy(self.nominal_kernels)
        new_env.nominal_rewards = np.copy(self.nominal_rewards)
        return new_env
    
    def _perturb_parameter(self, old_param, R, bias):
        lower_limit = max(0, old_param - R)
        higher_limit = min(1, old_param + R)
        while True:
            new_param = np.random.uniform(lower_limit, higher_limit)
            if new_param >= old_param + bias or new_param <= old_param - bias:
                break
        return new_param
    
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
        self.r_wait = 12     # reward for waiting
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
        new_env = RobotEnvironment(self.alpha, self.beta, self.seed)
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
        
        total_alpha = 0
        total_beta = 0

        for i in range(num_mdps):
            new_alpha = self._perturb_parameter(self.alpha, R, bias)
            new_beta = self._perturb_parameter(self.beta, R, bias)

            if i == num_mdps - 1:
                new_alpha = self.alpha
                new_beta = self.beta
            
            total_alpha += new_alpha
            total_beta += new_beta
            
            # Create a new Environment with the perturbed kernel and original rewards
            new_env = RobotEnvironment(new_alpha, new_beta, self.seed)
            uncertainty_set.append(new_env)

            # Log details of the current MDP in the uncertainty set
            logging.info(f"Uncertainty Set MDP {i+1}:")
            logging.info(f"Alpha: {new_alpha}")
            logging.info(f"Beta: {new_beta}")
            logging.info(f"Kernels:\n{new_env.kernels}")
            logging.info(f"Rewards:\n{new_env.rewards}")
            logging.info("-" * 50)
        
        # Compute the averages
        average_alpha = total_alpha / num_mdps
        average_beta = total_beta / num_mdps

        # Create an Environment object for the average MDP
        average_env = RobotEnvironment(average_alpha, average_beta, self.seed)

        # Log the average MDP details
        logging.info("Average MDP:")
        logging.info(f"Alpha: {average_alpha}")
        logging.info(f"Beta: {average_beta}")
        logging.info(f"Kernels:\n{average_env.kernels}")
        logging.info(f"Rewards:\n{average_env.rewards}")
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
        new_env = InventoryEnvironment(self.state_count, self.action_count, self.max_demand, self.demand_prob, self.seed)
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
        self.seed = seed
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
    
    def copy(self):
        new_env = GamblerEnvironment(self.state_count, self.head_prob, self.seed)
        new_env.kernels = np.copy(self.kernels)
        new_env.rewards = np.copy(self.rewards)
        new_env.nominal_kernels = np.copy(self.nominal_kernels)
        new_env.nominal_rewards = np.copy(self.nominal_rewards)
        return new_env
    
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
        
        total_head_prob = 0

        for i in range(num_mdps):
            uncertain_head_prob = self._perturb_parameter(self.head_prob, R, bias)
            
            total_head_prob += uncertain_head_prob
            
            # Create a new Environment with the perturbed kernel and original rewards
            new_env = GamblerEnvironment(self.state_count, uncertain_head_prob)
            uncertainty_set.append(new_env)

            # Log details of the current MDP in the uncertainty set
            logging.info(f"Uncertainty Set MDP {i+1}:")
            logging.info(f"Head prob: {uncertain_head_prob}")
            logging.info(f"Kernels:\n{new_env.kernels}")
            logging.info(f"Rewards:\n{new_env.rewards}")
            logging.info("-" * 50)
        
        # Compute the averages
        average_head_prob = total_head_prob / num_mdps

        # Create an Environment object for the average MDP
        average_env = GamblerEnvironment(self.state_count, average_head_prob)

        # Log the average MDP details
        logging.info("Average MDP:")
        logging.info(f"Head prob: {average_head_prob}")
        logging.info(f"Kernels:\n{average_env.kernels}")
        logging.info(f"Rewards:\n{average_env.rewards}")
        logging.info("=" * 50)

        return uncertainty_set, average_env
    
class DataCenterEnvironment(Environment):
    def __init__(self, alpha=0.6, beta=0.4, seed=None):
        """
        数据中心耗能优化环境
        :param alpha: 'normal' 状态转移到 'overloaded' 的概率
        :param beta: 'overloaded' 状态转移到 'failed' 的概率
        :param seed: 随机数种子
        """
        self.seed = seed
        self.state_count = 3  # 状态: 'normal', 'overloaded', 'failed'
        self.action_count = 2  # 动作: 'allocate', 'idle'
        # 状态转移概率
        self.alpha = alpha  # 正常运行转为过载的概率
        self.beta = beta    # 过载转为宕机的概率
        # 奖励
        self.r_allocate_normal = 100  # 正常状态下分配任务的奖励
        self.r_allocate_overloaded = 80  # 过载状态下分配任务的奖励
        self.r_idle = 20  # 待机奖励（低收益但保护服务器）
        self.r_failed = 0  # 宕机状态的奖励

        self.rng = np.random.default_rng(seed)  # 随机数生成器
        self.current_state = 0  # 初始状态为 'normal'

        # Generate nominal MDP
        self.kernels = self._generate_transition_kernels()
        self.rewards = self._generate_rewards()

        # Save nominal MDP
        self.nominal_kernels = self.kernels.copy()
        self.nominal_rewards = self.rewards.copy()

    def _generate_transition_kernels(self):
        """Generate transition probability kernels for all states and actions."""
        kernels = np.zeros((self.state_count, self.action_count, self.state_count))

        # Action 'allocate' (action index 0)
        kernels[0, 0, 0] = 1 - self.alpha   # Normal to Normal
        kernels[0, 0, 1] = self.alpha       # Normal to Overloaded
        kernels[0, 0, 2] = 0                # Normal to Failed
        kernels[1, 0, 0] = 0                # Overloaded to Normal
        kernels[1, 0, 1] = 1 - self.beta    # Overloaded to Overloaded
        kernels[1, 0, 2] = self.beta        # Overloaded to Failed
        kernels[2, 0, 2] = 1                # Failed to Failed

        # Action 'idle' (action index 1)
        kernels[0, 1, 0] = 1  # High stays High
        kernels[1, 1, 1] = 1  # Low stays Low
        kernels[2, 1, 2] = 1  # Dead stays Dead

        return kernels
    
    def _generate_rewards(self):
        """Generate reward matrix for all states and actions."""
        rewards = np.zeros((self.state_count, self.action_count))

        # Rewards for actions
        rewards[0, 0] = self.r_allocate_normal          # Normal state, allocate
        rewards[1, 0] = self.r_allocate_overloaded      # Overloaded state, allocate
        rewards[0, 1] = self.r_idle                     # Normal state, idle
        rewards[1, 1] = self.r_idle                     # Overloaded state, idle
        rewards[2, 0] = self.r_failed                   # Failed state, allocate
        rewards[2, 1] = self.r_failed                   # Failed state, idle
        return rewards

    def step(self, action):
        """
        执行动作并返回新状态、奖励和是否结束
        :param action: 0 表示 'allocate', 1 表示 'idle'
        :return: (新状态, 奖励, 是否结束)
        """
        if self.current_state == 2:  # 当前状态为 'failed'
            return self.current_state, self.r_failed, True  # 保持终止状态
        reward = 0
        if action == 0:  # 'allocate' 动作
            if self.current_state == 0:  # 'normal' 状态
                reward = self.r_allocate_normal
                self.current_state = 1 if self.rng.random() < self.alpha else 0
            elif self.current_state == 1:  # 'overloaded' 状态
                reward = self.r_allocate_overloaded
                self.current_state = 2 if self.rng.random() < self.beta else 1
        elif action == 1:  # 'idle' 动作
            reward = self.r_idle  # 待机的固定奖励
        done = self.current_state == 2  # 宕机状态为终止
        return self.current_state, reward, done
    
    def reset(self):
        """重置环境到初始状态"""
        self.current_state = 0  # 初始状态为 'normal'
        return self.current_state
    
    def copy(self):
        new_env = DataCenterEnvironment(self.alpha, self.beta, self.seed)
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
        
        total_alpha = 0
        total_beta = 0

        for i in range(num_mdps):
            new_alpha = self._perturb_parameter(self.alpha, R, bias)
            new_beta = self._perturb_parameter(self.beta, R, bias)
            
            total_alpha += new_alpha
            total_beta += new_beta
            
            # Create a new Environment with the perturbed kernel and original rewards
            new_env = DataCenterEnvironment(new_alpha, new_beta, self.seed)
            uncertainty_set.append(new_env)

            # Log details of the current MDP in the uncertainty set
            logging.info(f"Uncertainty Set MDP {i+1}:")
            logging.info(f"Alpha: {new_alpha}")
            logging.info(f"Beta: {new_beta}")
            logging.info(f"Kernels:\n{new_env.kernels}")
            logging.info(f"Rewards:\n{new_env.rewards}")
            logging.info("-" * 50)
        
        # Compute the averages
        average_alpha = total_alpha / num_mdps
        average_beta = total_beta / num_mdps

        # Create an Environment object for the average MDP
        average_env = DataCenterEnvironment(average_alpha, average_beta, self.seed)

        # Log the average MDP details
        logging.info("Average MDP:")
        logging.info(f"Alpha: {average_alpha}")
        logging.info(f"Beta: {average_beta}")
        logging.info(f"Kernels:\n{average_env.kernels}")
        logging.info(f"Rewards:\n{average_env.rewards}")
        logging.info("=" * 50)

        return uncertainty_set, average_env
    
class RandomMDPEnvironment(Environment):
    def __init__(self, state_count, action_count, bottom_states, low_reward, max_reward, seed=None):
        self.bottom_states = bottom_states
        self.low_reward = low_reward
        self.max_reward = max_reward
        
        super().__init__(state_count, action_count, seed)
    
    def _generate_transition_kernels(self):
        kernels = np.zeros((self.state_count, self.action_count, self.state_count))
        for s in range(self.state_count):
            for a in range(self.action_count):
                # 动作 a 对应的转移概率权重
                weights = np.random.rand(s + 1) + a  # 动作对状态转移的影响
                transition_probs = weights / weights.sum()  # 归一化
                kernels[s, a, :s + 1] = transition_probs
        return kernels
    
    def _generate_rewards(self):
        rewards = np.zeros((self.state_count, self.action_count))
        for s in range(self.bottom_states):
            rewards[s, :] = self.low_reward
        for s in range(self.bottom_states, self.state_count):
            rewards[s, :] = np.random.uniform(low=0, high=self.max_reward, size=(self.action_count,))
        return rewards
    
    def copy(self):
        new_env = RandomMDPEnvironment(self.state_count, self.action_count, self.bottom_states, self.low_reward, self.max_reward, self.seed)
        new_env.kernels = np.copy(self.kernels)
        new_env.rewards = np.copy(self.rewards)
        new_env.nominal_kernels = np.copy(self.nominal_kernels)
        new_env.nominal_rewards = np.copy(self.nominal_rewards)
        return new_env
    
    def _add_perturbation(self, nominal_probs, R, bias=0):
        """
        Add a bounded, biased perturbation to the last element of each row in the nominal probabilities.

        Args:
            nominal_probs (np.ndarray): The nominal probability distribution (2D array).
            R (float): Maximum perturbation radius for each probability element.
            bias (float): Bias to be added to the noise, introducing asymmetry.

        Returns:
            np.ndarray: Perturbed probability distribution.
        """
        # Create a copy to avoid modifying the original array
        perturbed_probs = nominal_probs.copy()

        # Generate noise for the last element in each row
        noise = np.random.uniform(-R, R) + bias

        # Add noise to the last element of each row
        perturbed_probs[-1] += noise
    
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
            new_env = RandomMDPEnvironment(self.state_count, self.action_count, self.bottom_states, self.low_reward, self.max_reward, self.seed)
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
        average_env = RandomMDPEnvironment(self.state_count, self.action_count, self.bottom_states, self.low_reward, self.max_reward, self.seed)
        average_env.kernels = average_kernels
        average_env.rewards = average_rewards

        # Log the average MDP details
        logging.info("Average MDP:")
        logging.info(f"Kernels:\n{average_kernels}")
        logging.info(f"Rewards:\n{average_rewards}")
        logging.info("=" * 50)

        return uncertainty_set, average_env
    
class VehicleRoutingEnvironment:
    """
    A simplified MDP environment for a Vehicle Routing Problem (VRP)
    that automatically creates its own cost matrix of dimension
    (num_customers+1) x (num_customers+1).

    - We treat node 0 as the depot.
    - Nodes 1..num_customers are the customers.
    - State = (current_node, tuple_of_remaining_customers).
    - Action = next_node_to_visit (an integer from [0..num_customers]).
    - Rewards = negative of the travel cost.
    """

    def __init__(self, num_customers, seed=None):
        """
        :param num_customers: Number of customers (excluding depot).
        :param seed: Random seed for reproducibility.
        """
        self.num_customers = num_customers
        self.dimension = num_customers + 1  # total nodes (0=depot + customers)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Step 1: Build the cost matrix internally
        self.cost_matrix = self._build_cost_matrix()
        
        # Step 2: Build the state and action spaces
        self.state_map, self.reverse_state_map = self._build_state_space()
        self.state_count = len(self.state_map)
        self.action_count = self.dimension  # We allow choosing any next_node in [0..num_customers]

        # Step 3: Build the MDP dynamics (transition probabilities + reward)
        self.kernels = np.zeros((self.state_count, self.action_count, self.state_count))
        self.rewards = np.zeros((self.state_count, self.action_count))
        self._build_mdp_dynamics()

        # Save nominal MDP
        self.nominal_kernels = self.kernels.copy()
        self.nominal_rewards = self.rewards.copy()

    def _build_cost_matrix(self):
        """
        Create a random cost matrix of shape (dimension x dimension).
        Diagonal = 0 (no cost to stay in same node).
        Off-diagonal = random uniform [1..10] or whatever you prefer.
        """
        cost_matrix = np.zeros((self.dimension, self.dimension))
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i == j:
                    cost_matrix[i, j] = 0.0
                else:
                    # Example: random cost in [1..10]
                    cost_matrix[i, j] = np.random.uniform(1, 10)
        return cost_matrix

    def _build_state_space(self):
        """
        Build:
         - state_map: (current_node, (remaining...)) -> state_index
         - reverse_state_map: state_index -> (current_node, (remaining...))

        We systematically generate all subsets of [1..num_customers]
        for the "remaining" set, and pair them with each possible current node.
        """
        from itertools import combinations

        # All subsets of customers
        customers = range(1, self.num_customers + 1)
        subsets = []
        for r in range(self.num_customers + 1):
            for combo in combinations(customers, r):
                subsets.append(tuple(sorted(combo)))

        all_states = []
        for remaining_tuple in subsets:
            for current_node in range(self.dimension):
                all_states.append((current_node, remaining_tuple))

        state_map = {s: idx for idx, s in enumerate(all_states)}
        reverse_state_map = {idx: s for s, idx in state_map.items()}
        return state_map, reverse_state_map

    def _build_mdp_dynamics(self):
        """
        Build deterministic transitions + immediate reward = - cost_matrix.
        If an action is invalid (choosing a node not in 'remaining' 
        except returning to depot at the end), we penalize heavily.
        """
        for s_idx in range(self.state_count):
            current_node, remaining = self.reverse_state_map[s_idx]
            remaining_set = set(remaining)

            for a in range(self.action_count):
                # 'a' is the node we attempt to move to next
                if len(remaining) == 0:
                    # If no customers remain, the "valid" action is to go to depot (0).
                    # If we're not already at depot and choose a=0, do that. 
                    if a == 0 and current_node != 0:
                        next_state = (0, ())
                        reward = -self.cost_matrix[current_node][0]
                    else:
                        # Doing nothing or invalid => no real movement
                        next_state = (current_node, remaining)
                        reward = 0
                else:
                    # If 'a' is in the remaining set, we move there.
                    if a in remaining_set:
                        next_remaining = tuple(sorted(x for x in remaining if x != a))
                        next_state = (a, next_remaining)
                        reward = -self.cost_matrix[current_node][a]
                    else:
                        # Invalid => large negative penalty
                        next_state = (current_node, remaining)
                        reward = -9999

                # Deterministic transition => set probability = 1
                next_idx = self.state_map[next_state]
                self.kernels[s_idx, a, next_idx] = 1.0
                self.rewards[s_idx, a] = reward

    def step(self, state_idx, action):
        """
        Return the next state index (deterministic).
        """
        probs = self.kernels[state_idx, action]
        next_state_idx = np.random.choice(self.state_count, p=probs)
        return next_state_idx

    def get_reward(self, state_idx, action):
        """
        Return immediate reward for (state, action).
        """
        return self.rewards[state_idx, action]

    def copy(self):
        """
        Return a deep copy of the environment.
        """
        new_env = VehicleRoutingEnvironment(self.num_customers)
        new_env.cost_matrix = np.copy(self.cost_matrix)
        new_env.state_map = self.state_map.copy()
        new_env.reverse_state_map = self.reverse_state_map.copy()
        new_env.state_count = self.state_count
        new_env.action_count = self.action_count
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
    
# 测试环境
if __name__ == "__main__":
   env = DataCenterEnvironment(alpha=0.7, beta=0.5, seed=42)
   state = env.reset()
   done = False
   total_reward = 0
   print("开始数据中心优化模拟")
   while not done:
       action = np.random.choice(2)  # 随机选择动作 (0: 'allocate', 1: 'idle')
       next_state, reward, done = env.step(action)
       total_reward += reward
       print(f"状态: {state}, 动作: {action}, 新状态: {next_state}, 奖励: {reward}, 是否结束: {done}")
       state = next_state
   print(f"总奖励: {total_reward}")

