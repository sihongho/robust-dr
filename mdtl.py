import numpy as np
import logging
from environment import *
import cvxpy as cp

def calculate_optimal_value(V, sample_distribution, R):
    """Solve the optimization problem for robust average reward."""
    x = cp.Variable(len(V))
    objective = cp.Maximize((sample_distribution @ (V - x) - R * (cp.max(V - x) - cp.min(V - x))))
    constraints = [0 <= x]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return problem.value

def MDTL_Periodic(uncertainty_set, T, K, gamma, lambdas, E, state_count, action_count, R, mode="avg", tolerance=1e-5):
    if mode not in {"avg", "max"}:
        raise ValueError("Mode must be 'avg' or 'max'.")

    # Initialize Q-values for each MDP
    Q_k = {k: np.zeros((state_count, action_count)) for k in range(K)}
    # Record all V values for each iteration
    all_Q = []

    # Convergence flag
    converged = False

    for t in range(T):
        logging.info(f"Step {t + 1}/{T}.")
        V_k = {k: np.zeros(state_count) for k in range(K)}

        # Update Q_k for each MDP
        for k in range(K):
            mdp = uncertainty_set[k]

            # Compute V_k(s)
            for s in range(state_count):
                V_k[k][s] = np.max(Q_k[k][s, :])

            # Update Q_k(s, a)
            for s in range(state_count):
                for a in range(action_count):
                    next_state_distribution = mdp.kernels[s, a]  # Transition probabilities
                    reward = mdp.get_reward(s, a)
                    if a == 0:
                        opt_all = calculate_optimal_value(V_k[k], next_state_distribution, R)
                    else:
                        opt_all = np.inner(V_k[k], next_state_distribution)
                    Q_k[k][s, a] = (1 - lambdas[t]) * Q_k[k][s, a] + lambdas[t] * (reward + gamma * opt_all)

        # Periodic aggregation: if t mod E == 0
        if (t + 1) % E == 0:
            if mode == "avg":
                Q_result = np.mean([Q_k[k] for k in range(K)], axis=0)
            elif mode == "max":
                Q_result = np.max(np.stack([Q_k[k] for k in range(K)]), axis=0)
            
            # Store current V in the record
            all_Q.append(Q_result.copy())

            # Check for convergence
            # max_change = 0
            for k in range(K):
                # max_change = max(max_change, np.max(np.abs(Q_k[k] - Q_result)))
                Q_k[k] = Q_result.copy()

            # If the maximum change is below the tolerance, stop iterations
            # if max_change < tolerance:
            #     logging.info(f"Algorithm converged after {t + 1} iterations.")
            #     converged = True
            #     break

    # Final aggregation
    if mode == "avg":
        Q_result = np.mean([Q_k[k] for k in range(K)], axis=0)
    elif mode == "max":
        Q_result = np.max(np.stack([Q_k[k] for k in range(K)]), axis=0)

    return Q_result, all_Q

if __name__ == "__main__":
    # Initialize environment and uncertainty set
    env = Environment(state_count=5, action_count=3, seed=42)
    uncertainty_set = env.create_uncertainty_set(R=0, bias=0, num_mdps=3)

    # Define shared parameters
    params = {
        "uncertainty_set": uncertainty_set,
        "T": 10,
        "K": 3,
        "gamma": 0.99,
        "lambdas": [0.1] * 10,
        "E": 2,
        "state_count": 5,
        "action_count": 3,
        "R": 0
    }

    # Compute Q-values for "avg" mode
    Q_avg, _ = MDTL_Periodic(**params, mode="avg")
    print("Final Averaged Q-values:", Q_avg)

    # Compute Q-values for "max" mode
    Q_max, _ = MDTL_Periodic(**params, mode="max")
    print("Final Maximum Q-values:", Q_max)