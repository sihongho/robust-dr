import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def main(args):
    # Load parameters from command-line arguments.
    settings = args.R_test
    total_step = args.total_step
    method = args.method

    results = []
    for i in range(len(settings)):
        path = f"./experiments/{settings[i]}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        v_robust_raw = data["all_V_nominal_robust"]
        v_robust_mean = []
        for i in range(total_step + 1):
            v_robust_mean.append(np.mean(v_robust_raw[i]))
        results.append(v_robust_mean)

    x = np.arange(total_step + 1)

    plt.figure(figsize=(14, 6))
    plt.plot(x, results[0], label="R_test=0.1")
    plt.plot(x, results[1], label="R_test=0.07")
    plt.plot(x, results[2], label="R_test=0.05")
    plt.plot(x, results[3], label="R_test=0.03")
    plt.plot(x, results[4], label="R_test=0.01")

    plt.xlabel("Step")
    plt.ylabel("V_nominal_robust")
    if method == "avg":
        plt.title("Robust Average Multi-learn")
    elif method == "max":
        plt.title("Robust Max Multi-learn")
    
    plt.legend()

    plt.savefig(f"plot_{method}.png")
    plt.show()

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Process different R_test results.")

    # Define individual arguments for each JSON file
    parser.add_argument("--R_test", type=str, nargs='+')
    parser.add_argument("--method", type=str, choices=['avg', 'max'])
    parser.add_argument("--total_step", type=int, default=300)

    args = parser.parse_args()
    print(args)
    main(args)