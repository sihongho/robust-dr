import numpy as np
from environment import *
from mdtl import *

def get_deterministic_policy_from_Q(Q):
    """
    Derive a deterministic policy from Q values.


    Parameters:
    - Q: A 2D NumPy array of shape (num_states, num_actions).


    Returns:
    - policy: A 2D NumPy array of shape (num_states, num_actions), where policy[s, a] is 1 if action a is the best action for state s, and 0 otherwise.
    """
    num_states, num_actions = Q.shape
    policy = np.zeros_like(Q)
    best_actions = np.argmax(Q, axis=1)  # Get the index of the best action for each state
    policy[np.arange(num_states), best_actions] = 1  # Set the best action's probability to 1
    return policy

def get_deterministic_policy_with_adversary(Q):
    """
    Derive a policy from Q values that, for each state, assigns a 50% chance
    to the best action and a 50% chance to a random action among the rest.
    
    Parameters:
    - Q: A 2D NumPy array of shape (num_states, num_actions).

    Returns:
    - policy: A 2D NumPy array of shape (num_states, num_actions) where for each state,
      policy[s, a] is 0.5 if a is the best action or the chosen adversarial action, and 0 otherwise.
    """
    num_states, num_actions = Q.shape
    policy = np.zeros_like(Q)
    best_actions = np.argmax(Q, axis=1)  # Best action index for each state
    
    for s in range(num_states):
        best = best_actions[s]
        # If there is only one action, assign full probability to it.
        if num_actions == 1:
            policy[s, best] = 1.0
        else:
            # Assign 50% to the best action
            policy[s, best] = 0.5
            
            # Identify all actions except the best
            non_best_actions = [a for a in range(num_actions) if a != best]
            # Choose one random action among the non-best actions
            random_action = np.random.choice(non_best_actions)
            policy[s, random_action] = 0.5
            
    return policy

def get_stochastic_policy_from_Q(Q, temperature=1.0):
    """
    Derive a stochastic policy using softmax from Q values.

    Parameters:
    - Q: A 2D NumPy array of shape (num_states, num_actions).
    - temperature: A float, the temperature parameter for softmax.

    Returns:
    - policy: A 2D NumPy array of shape (num_states, num_actions), where policy[s, a] is the probability of taking action a in state s.
    """
    exp_Q = np.exp(Q / temperature)  # Apply softmax to Q values
    policy = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)  # Normalize to get probabilities
    return policy

def policy_evaluation(Q, env, R, gamma=0.99, theta=1e-6, max_iterations=1000, policy_type="deterministic"):
    # Derive policy from Q
    if policy_type == "deterministic":
        policy = get_deterministic_policy_from_Q(Q)
    elif policy_type == "adversary":
        policy = get_deterministic_policy_with_adversary(Q)
    else:
        policy = get_stochastic_policy_from_Q(Q)
    state_count = env.state_count
    action_count = env.action_count

    # Extract kernels and rewards
    kernels = env.kernels  # Shape: (state_count, action_count, state_count)
    rewards = env.rewards  # Shape: (state_count, action_count)

    # Initialize value function
    # V = np.zeros(state_count)
    V = np.random.normal(loc=0.0, scale=1.0, size=state_count)
    all_V = []
    all_V.append(V.copy())

    for i in range(max_iterations):
        delta = 0
        # Update V(s) for each state
        for s in range(state_count):
            v = V[s]
            V[s] = sum(
                policy[s, a] * (rewards[s, a] + gamma * sum(kernels[s, a, s_next] * calculate_optimal_value(V, kernels[s, a], R) for s_next in range(state_count)))
                for a in range(action_count)
            )
            delta = max(delta, abs(v - V[s]))

        # Save all_V value
        all_V.append(V.copy())

        # Check for convergence
        # if delta < theta:
        #     print(f"Policy evaluation converged after {i + 1} iterations.")
        #     break
        # else:
        #     print("Policy evaluation reached the maximum number of iterations without full convergence.")

    return V, all_V

if __name__ == "__main__":
    # Initialize environment and uncertainty set
    env = Environment(state_count=10, action_count=3, seed=42)
    uncertainty_set = env.create_uncertainty_set(R=0, bias=0, num_mdps=3)

    # Define shared parameters
    params = {
        "uncertainty_set": uncertainty_set,
        "T": 10,
        "K": 3,
        "gamma": 0.99,
        "lambdas": [0.1] * 10,
        "E": 2,
        "state_count": 10,
        "action_count": 3,
        "R": 0
    }

    # Compute Q-values for "avg" mode
    Q, _ = MDTL_Periodic(**params, mode="avg")
    V_eva_non_robust = policy_evaluation(Q, env, R=0, gamma=0.99)
    print(V_eva_non_robust)
    V_eva_robust = policy_evaluation(Q, env, R=0.2, gamma=0.99)
    print(V_eva_robust)