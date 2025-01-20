import argparse
import pickle
import json
import os
import logging
from datetime import datetime
from environment import Environment, RobotEnvironment, InventoryEnvironment
# from config import CONFIG, MODE
from mdtl import MDTL_Periodic
from policy_eva import policy_evaluation
import numpy as np
from utils import *

def main(args):
    # Load parameters from command-line arguments or CONFIG
    state_count = args.state_count 
    action_count = args.action_count
    alpha = args.alpha
    beta = args.beta
    max_demand = args.max_demand
    env_type = args.env_type 
    total_step = args.total_step 
    learning_rate = args.learning_rate 
    discount_rate = args.discount_rate 
    aggregation_mode = args.aggregation_mode 
    eva_max_iterations = args.eva_max_iterations
    bias = args.bias
    R = args.R 
    R_test = args.R_test
    E = args.E
    num_mdps = args.num_mdps 
    random_seed = args.random_seed 
    policy_type = args.policy_type 
    learn_domain = args.learn_domain

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Set up logging and directories
    experiment_dir = setup_logging_and_dirs()

    # Initialize or load environment and uncertainty set
    if args.load_env_path:
        logging.info(f"Loading environment and uncertainty set from {args.load_env_path}...")
        with open(args.load_env_path, "rb") as f:
            data = pickle.load(f)  # Assume the file is in pickle format
        env = data["env"]
        uncertainty_set = data["uncertainty_set"]
        average_env = data["average_env"]
        if num_mdps == 1:
            uncertainty_set, average_env = [env.copy()], env.copy()
            if learn_domain == 'avg':
                uncertainty_set = [average_env.copy()]
    else:
        if env_type == 'robot':
            env = RobotEnvironment(alpha=alpha, beta=beta)
        elif env_type == 'inventory':
            env = InventoryEnvironment(state_count=state_count, action_count=action_count, max_demand=max_demand)
        else:
            env = Environment(state_count=state_count, action_count=action_count)
        if num_mdps > 1:
            uncertainty_set, average_env = env.create_uncertainty_set(num_mdps=num_mdps, R=R, bias=bias)
        else:
            uncertainty_set, average_env = [env.copy()], env.copy()
        if learn_domain == 'avg':
            uncertainty_set = [average_env.copy()]

        # Save environment and uncertainty set for this run
        env_save_path = os.path.join(experiment_dir, "env_data.pkl")
        with open(env_save_path, "wb") as f:
            pickle.dump({"env": env, "uncertainty_set": uncertainty_set, "average_env":average_env}, f)
        logging.info(f"Environment and uncertainty set saved to {env_save_path}")

    # Save configuration to experiment_dir
    save_config(experiment_dir, vars(args))
    # Run MDTL_Periodic
    logging.info("Running MDTL_Periodic...")
    lambdas = [learning_rate] * total_step  # Use fixed learning rate for MDTL
    Q_result, all_Q = MDTL_Periodic(
        uncertainty_set=uncertainty_set,
        T=total_step,
        K=num_mdps,
        gamma=discount_rate,
        lambdas=lambdas,
        E=E,  # Aggregation interval fixed as an example
        state_count=state_count,
        action_count=action_count,
        R=R,
        mode=aggregation_mode
    )
    logging.info("MDTL_Periodic completed.")

    # Policy evaluation
    logging.info("Running policy evaluation...")
    V_nominal_nonrobust = policy_evaluation(Q_result, env, R=0, gamma=discount_rate, theta=1e-5, max_iterations=eva_max_iterations, policy_type=policy_type)
    logging.info(f"Nominal Non-Robust Value Function: {V_nominal_nonrobust}")

    logging.info("Running robust policy evaluation...")
    V_nominal_robust = policy_evaluation(Q_result, env, R=R_test, gamma=discount_rate, theta=1e-5, max_iterations=eva_max_iterations, policy_type=policy_type)
    logging.info(f"Nominal Robust Value Function: {V_nominal_robust}")

    logging.info("Running avg policy evaluation...")
    V_avg_nonrobust = policy_evaluation(Q_result, average_env, R=0, gamma=discount_rate, theta=1e-5, max_iterations=eva_max_iterations, policy_type=policy_type)
    logging.info(f"Average Non-Robust Value Function: {V_avg_nonrobust}")

    logging.info("Running avg robust policy evaluation...")
    V_avg_robust = policy_evaluation(Q_result, average_env, R=R_test, gamma=discount_rate, theta=1e-5, max_iterations=eva_max_iterations, policy_type=policy_type)
    logging.info(f"Average Robust Value Function: {V_avg_robust}")

    logging.info("Policy evaluation completed.")

    # Save final results
    results_path = os.path.join(experiment_dir, "results.json")
    all_Q_serializable = [q.tolist() for q in all_Q]
    with open(results_path, "w") as f:
        json.dump({
            "Q_result": Q_result.tolist(),
            "V_nominal_nonrobust": V_nominal_nonrobust.tolist(),
            "V_nominal_robust": V_nominal_robust.tolist(),
            "V_avg_nonrobust": V_avg_nonrobust.tolist(),
            "V_avg_robust": V_avg_robust.tolist(),
            "all_Q": all_Q_serializable
        }, f, indent=4)
    logging.info(f"Results saved to {results_path}")

    # Print results
    logging.info("\nFinal Results:")
    logging.info(f"Optimal Q-values from MDTL_Periodic:\n{Q_result}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MDTL and policy evaluation experiments.")
    # 跑实验，控制变量num_mdps, R, aggregation_mode, bias即可. 可比较robust vs non-robust，multi-learn vs single learn.
    # Add arguments with default values
    parser.add_argument("--state_count", type=int, default=3, help="Number of states in the environment (default: 15)")
    parser.add_argument("--action_count", type=int, default=2, help="Number of actions in the environment (default: 30)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Pr(stay at high charge if searching | now have high charge) (default: 0.1)")
    parser.add_argument("--beta", type=float, default=0.1, help="Pr(stay at low charge if searching | now have low charge) (default: 0.1)")
    parser.add_argument("--max_demand", type=int, default=29, help="Max demand in Inventory Environment (default: 29)")
    parser.add_argument("--env_type", type=str, choices=["robot", "inventory", "env"], default="env", help="Environment type: RobotEnvironment, InventoryEnvironment, or Environment (default: Environment)")
    parser.add_argument("--total_step", type=int, default=5000, help="Total number of steps to run MDTL (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for MDTL (default: 0.01)")
    parser.add_argument("--discount_rate", type=float, default=0.95, help="Discount factor for rewards (default: 0.95)")
    parser.add_argument("--aggregation_mode", type=str, choices=["avg", "max"], default="avg", help="Aggregation mode: 'avg' or 'max' (default: 'avg')")
    parser.add_argument("--eva_max_iterations", type=int, default=5000)
    parser.add_argument("--policy_type", type=str, default="deterministic")
    parser.add_argument("--learn_domain", type=str, default="nominal", help="avg")
    parser.add_argument("--R", type=float, default=0.4, help="Radius for uncertainty set (default: 0.4)")
    parser.add_argument("--R_test", type=float, default=0.4, help="Radius for uncertainty set (default: 0.4)")
    parser.add_argument("--bias", type=float, default=0.1, help="Bias for uncertainty set (default: 0.1)")
    parser.add_argument("--num_mdps", type=int, default=2, help="Number of MDPs in the uncertainty set (default: 2)")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--E", type=int, default=5, help="Periodic number for MDTL aggregation (default: 2)")
    parser.add_argument("--load_env_path", type=str, default=None, help="Path to the file containing pre-saved environment and uncertainty set (default: './env_data.pkl')")
    # parser.add_argument("--save_env_path", type=str, default="env_data.pkl", help="Path to save the current environment and uncertainty set (default: './env_data.pkl')")
    # Parse arguments
    args = parser.parse_args()
    print(args)
    main(args)

# python main.py --state_count 3 --action_count 2 --total_step 5000 --learning_rate 0.01 --discount_rate 0.95 --aggregation_mode avg --eva_max_iterations 5000 --R 0.4 --bias 0.1 --num_mdps 2 --random_seed 42 --E 5 --load_env_path ./predefined_env.pkl --save_env_path ./output_env.pkl

# python main.py --state_count 20～100  --action_count 30～80 --total_step 5000 --learning_rate 0.01 --discount_rate 0.95 --aggregation_mode avg --eva_max_iterations 5000 --R 0.4 --bias 0.1 --num_mdps 2 --random_seed 42 --E 5 --load_env_path ./predefined_env.pkl --save_env_path ./output_env.pkl