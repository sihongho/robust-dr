import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def main(args):
    # Load parameters from command-line arguments.
    R_test_01 = args.R_test_01
    R_test_03 = args.R_test_03
    R_test_05 = args.R_test_05
    R_test_07 = args.R_test_07
    R_test_10 = args.R_test_10
    total_step = args.total_step
    method = args.method

    results_01 = []
    for i in range(len(R_test_01)):
        path = f"./experiments/{R_test_01[i]}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        v_robust_raw = data["all_V_nominal_robust"]
        v_robust_mean = []
        for i in range(total_step + 1):
            v_robust_mean.append(np.mean(v_robust_raw[i]))
        results_01.append(v_robust_mean)

    mean_01 = np.mean(results_01, axis=0)
    std_01 = np.std(results_01, axis=0)

    results_03 = []
    for i in range(len(R_test_03)):
        path = f"./experiments/{R_test_03[i]}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        v_robust_raw = data["all_V_nominal_robust"]
        v_robust_mean = []
        for i in range(total_step + 1):
            v_robust_mean.append(np.mean(v_robust_raw[i]))
        results_03.append(v_robust_mean)

    mean_03 = np.mean(results_03, axis=0)
    std_03 = np.std(results_03, axis=0)

    results_05 = []
    for i in range(len(R_test_05)):
        path = f"./experiments/{R_test_05[i]}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        v_robust_raw = data["all_V_nominal_robust"]
        v_robust_mean = []
        for i in range(total_step + 1):
            v_robust_mean.append(np.mean(v_robust_raw[i]))
        results_05.append(v_robust_mean)

    mean_05 = np.mean(results_05, axis=0)
    std_05 = np.std(results_05, axis=0)

    results_07 = []
    for i in range(len(R_test_07)):
        path = f"./experiments/{R_test_07[i]}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        v_robust_raw = data["all_V_nominal_robust"]
        v_robust_mean = []
        for i in range(total_step + 1):
            v_robust_mean.append(np.mean(v_robust_raw[i]))
        results_07.append(v_robust_mean)

    mean_07 = np.mean(results_07, axis=0)
    std_07 = np.std(results_07, axis=0)

    results_10 = []
    for i in range(len(R_test_10)):
        path = f"./experiments/{R_test_10[i]}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        v_robust_raw = data["all_V_nominal_robust"]
        v_robust_mean = []
        for i in range(total_step + 1):
            v_robust_mean.append(np.mean(v_robust_raw[i]))
        results_10.append(v_robust_mean)

    mean_10 = np.mean(results_10, axis=0)
    std_10 = np.std(results_10, axis=0)
    print(std_10)



    x = np.arange(total_step + 1)

    plt.figure(figsize=(14, 6))
    plt.plot(x, mean_10, label="R_test=0.1", color='r')
    plt.plot(x, mean_07, label="R_test=0.07", color='b')
    plt.plot(x, mean_05, label="R_test=0.05", color='g')
    plt.plot(x, mean_03, label="R_test=0.03", color='orange')
    plt.plot(x, mean_01, label="R_test=0.01", color='purple')

    plt.fill_between(x, mean_10 - 10 * std_10, mean_10 + 10 * std_10, alpha=0.2, color='r')
    plt.fill_between(x, mean_07 - 10 * std_07, mean_07 + 10 * std_07, alpha=0.2, color='b')
    plt.fill_between(x, mean_05 - 10 * std_05, mean_05 + 10 * std_05, alpha=0.2, color='g')
    plt.fill_between(x, mean_03 - 10 * std_03, mean_03 + 10 * std_03, alpha=0.2, color='orange')
    plt.fill_between(x, mean_01 - 10 * std_01, mean_01 + 10 * std_01, alpha=0.2, color='purple')

    plt.xlabel("Time Step", fontsize=20)
    plt.ylabel("Robust Value Function", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if method == "avg":
        plt.title("MDTL-Avg", fontsize=24)
    elif method == "max":
        plt.title("MDTL-Max", fontsize=24)
    
    plt.legend(fontsize=16)

    plt.savefig(f"plot_{method}.png")
    plt.show()

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Process different R_test results.")

    # Define individual arguments for each JSON file
    parser.add_argument("--R_test_01", type=str, nargs='+')
    parser.add_argument("--R_test_03", type=str, nargs='+')
    parser.add_argument("--R_test_05", type=str, nargs='+')
    parser.add_argument("--R_test_07", type=str, nargs='+')
    parser.add_argument("--R_test_10", type=str, nargs='+')
    parser.add_argument("--method", type=str, choices=['avg', 'max'])
    parser.add_argument("--total_step", type=int, default=300)

    args = parser.parse_args()
    print(args)
    main(args)