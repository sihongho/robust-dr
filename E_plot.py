import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def main(args):
    m1_e1 = args.m1_e1
    m1_e2 = args.m1_e2
    m1_e3 = args.m1_e3
    m1_e4 = args.m1_e4
    # m1_e5 = args.m1_e5
    m2_e1 = args.m2_e1
    m2_e2 = args.m2_e2
    m2_e3 = args.m2_e3
    m2_e4 = args.m2_e4
    # m2_e5 = args.m2_e5
    total_step = args.total_step

    results_m1_e1 = []
    for i in range(len(m1_e1)):
        path = f"./experiments/{m1_e1[i]}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        v_robust_raw = data["all_V_nominal_robust"]
        v_robust_mean = []
        for i in range(total_step + 1):
            v_robust_mean.append(np.mean(v_robust_raw[i]))
        results_m1_e1.append(v_robust_mean)
    mean_m1_e1 = np.mean(results_m1_e1, axis=0)
    std_m1_e1 = np.std(results_m1_e1, axis=0)

    results_m1_e2 = []
    for i in range(len(m1_e2)):
        path = f"./experiments/{m1_e2[i]}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        v_robust_raw = data["all_V_nominal_robust"]
        v_robust_mean = []
        for i in range(total_step + 1):
            v_robust_mean.append(np.mean(v_robust_raw[i]))
        results_m1_e2.append(v_robust_mean)
    mean_m1_e2 = np.mean(results_m1_e2, axis=0)
    std_m1_e2 = np.std(results_m1_e2, axis=0)

    results_m1_e3 = []
    for i in range(len(m1_e3)):
        path = f"./experiments/{m1_e3[i]}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        v_robust_raw = data["all_V_nominal_robust"]
        v_robust_mean = []
        for i in range(total_step + 1):
            v_robust_mean.append(np.mean(v_robust_raw[i]))
        results_m1_e3.append(v_robust_mean)
    mean_m1_e3 = np.mean(results_m1_e3, axis=0)
    std_m1_e3 = np.std(results_m1_e3, axis=0)

    results_m1_e4 = []
    for i in range(len(m1_e4)):
        path = f"./experiments/{m1_e4[i]}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        v_robust_raw = data["all_V_nominal_robust"]
        v_robust_mean = []
        for i in range(total_step + 1):
            v_robust_mean.append(np.mean(v_robust_raw[i]))
        results_m1_e4.append(v_robust_mean)
    mean_m1_e4 = np.mean(results_m1_e4, axis=0)
    std_m1_e4 = np.std(results_m1_e4, axis=0)

    # results_m1_e5 = []
    # for i in range(len(m1_e5)):
    #     path = f"./experiments/{m1_e5[i]}/results.json"
    #     # Load the results from JSON files.
    #     with open(path, "r") as f:
    #         data = json.load(f)
    #     # Read all V results.
    #     v_robust_raw = data["all_V_nominal_robust"]
    #     v_robust_mean = []
    #     for i in range(total_step + 1):
    #         v_robust_mean.append(np.mean(v_robust_raw[i]))
    #     results_m1_e5.append(v_robust_mean)
    # mean_m1_e5 = np.mean(results_m1_e5, axis=0)
    # std_m1_e5 = np.std(results_m1_e5, axis=0)

    results_m2_e1 = []
    for i in range(len(m2_e1)):
        path = f"./experiments/{m2_e1[i]}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        v_robust_raw = data["all_V_nominal_robust"]
        v_robust_mean = []
        for i in range(total_step + 1):
            v_robust_mean.append(np.mean(v_robust_raw[i]))
        results_m2_e1.append(v_robust_mean)
    mean_m2_e1 = np.mean(results_m2_e1, axis=0)
    std_m2_e1 = np.std(results_m2_e1, axis=0)

    results_m2_e2 = []
    for i in range(len(m2_e2)):
        path = f"./experiments/{m2_e2[i]}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        v_robust_raw = data["all_V_nominal_robust"]
        v_robust_mean = []
        for i in range(total_step + 1):
            v_robust_mean.append(np.mean(v_robust_raw[i]))
        results_m2_e2.append(v_robust_mean)
    mean_m2_e2 = np.mean(results_m2_e2, axis=0)
    std_m2_e2 = np.std(results_m2_e2, axis=0)

    results_m2_e3 = []
    for i in range(len(m2_e3)):
        path = f"./experiments/{m2_e3[i]}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        v_robust_raw = data["all_V_nominal_robust"]
        v_robust_mean = []
        for i in range(total_step + 1):
            v_robust_mean.append(np.mean(v_robust_raw[i]))
        results_m2_e3.append(v_robust_mean)
    mean_m2_e3 = np.mean(results_m2_e3, axis=0)
    std_m2_e3 = np.std(results_m2_e3, axis=0)

    results_m2_e4 = []
    for i in range(len(m2_e4)):
        path = f"./experiments/{m2_e4[i]}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        v_robust_raw = data["all_V_nominal_robust"]
        v_robust_mean = []
        for i in range(total_step + 1):
            v_robust_mean.append(np.mean(v_robust_raw[i]))
        results_m2_e4.append(v_robust_mean)
    mean_m2_e4 = np.mean(results_m2_e4, axis=0)
    std_m2_e4 = np.std(results_m2_e4, axis=0)

    # results_m2_e5 = []
    # for i in range(len(m2_e5)):
    #     path = f"./experiments/{m2_e5[i]}/results.json"
    #     # Load the results from JSON files.
    #     with open(path, "r") as f:
    #         data = json.load(f)
    #     # Read all V results.
    #     v_robust_raw = data["all_V_nominal_robust"]
    #     v_robust_mean = []
    #     for i in range(total_step + 1):
    #         v_robust_mean.append(np.mean(v_robust_raw[i]))
    #     results_m2_e5.append(v_robust_mean)
    # mean_m2_e5 = np.mean(results_m2_e5, axis=0)
    # std_m2_e5 = np.std(results_m2_e5, axis=0)

    print(mean_m1_e2 == mean_m1_e2)

    x = np.arange(total_step + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(x, mean_m1_e1, label="E=1", color='r')
    axes[0].plot(x, mean_m1_e2, label="E=10", color='b')
    axes[0].plot(x, mean_m1_e3, label="E=20", color='g')
    axes[0].plot(x, mean_m1_e4, label="E=50", color='orange')
    # axes[0].plot(x, mean_m1_e5, label="E=50", color='purple')

    axes[0].fill_between(x, mean_m1_e1 - 10 * std_m1_e1, mean_m1_e1 + 10 * std_m1_e1, alpha=0.2, color='r')
    axes[0].fill_between(x, mean_m1_e2 - 10 * std_m1_e2, mean_m1_e2 + 10 * std_m1_e2, alpha=0.2, color='b')
    axes[0].fill_between(x, mean_m1_e3 - 10 * std_m1_e3, mean_m1_e3 + 10 * std_m1_e3, alpha=0.2, color='g')
    axes[0].fill_between(x, mean_m1_e4 - 10 * std_m1_e4, mean_m1_e4 + 10 * std_m1_e4, alpha=0.2, color='orange')
    # axes[0].fill_between(x, mean_m1_e5 - 10 * std_m1_e5, mean_m1_e5 + 10 * std_m1_e5, alpha=0.2, color='purple')

    axes[0].set_xlabel("Time Step", fontsize=20)
    axes[0].set_ylabel("Robust Value Function", fontsize=20)
    axes[0].set_title("MDTL-Avg", fontsize=24)
    axes[0].tick_params(axis='x', labelsize=14)
    axes[0].tick_params(axis='y', labelsize=14)
    axes[0].legend(fontsize=16)

    axes[1].plot(x, mean_m2_e1, label="E=1", color='r')
    axes[1].plot(x, mean_m2_e2, label="E=10", color='b')
    axes[1].plot(x, mean_m2_e3, label="E=20", color='g')
    axes[1].plot(x, mean_m2_e4, label="E=50", color='orange')
    # axes[1].plot(x, mean_m2_e5, label="E=50", color='purple')

    axes[1].fill_between(x, mean_m2_e1 - 10 * std_m2_e1, mean_m2_e1 + 10 * std_m2_e1, alpha=0.2, color='r')
    axes[1].fill_between(x, mean_m2_e2 - 10 * std_m2_e2, mean_m2_e2 + 10 * std_m2_e2, alpha=0.2, color='b')
    axes[1].fill_between(x, mean_m2_e3 - 10 * std_m2_e3, mean_m2_e3 + 10 * std_m2_e3, alpha=0.2, color='g')
    axes[1].fill_between(x, mean_m2_e4 - 10 * std_m2_e4, mean_m2_e4 + 10 * std_m2_e4, alpha=0.2, color='orange')
    # axes[1].fill_between(x, mean_m2_e5 - 10 * std_m2_e5, mean_m2_e5 + 10 * std_m2_e5, alpha=0.2, color='purple')

    axes[1].set_xlabel("Time Step", fontsize=20)
    axes[1].set_ylabel("Robust Value Function", fontsize=20)
    axes[1].set_title("MDTL-Max", fontsize=24)
    axes[1].tick_params(axis='x', labelsize=14)
    axes[1].tick_params(axis='y', labelsize=14)
    axes[1].legend(fontsize=16)

    plt.tight_layout()
    plt.savefig("plot_E.png")
    plt.show()

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Process different E results.")

    # Define individual arguments for each JSON file
    parser.add_argument("--m1_e1", type=str, nargs='+')
    parser.add_argument("--m1_e2", type=str, nargs='+')
    parser.add_argument("--m1_e3", type=str, nargs='+')
    parser.add_argument("--m1_e4", type=str, nargs='+')
    # parser.add_argument("--m1_e5", type=str, nargs='+')
    parser.add_argument("--m2_e1", type=str, nargs='+')
    parser.add_argument("--m2_e2", type=str, nargs='+')
    parser.add_argument("--m2_e3", type=str, nargs='+')
    parser.add_argument("--m2_e4", type=str, nargs='+')
    # parser.add_argument("--m2_e5", type=str, nargs='+')
    parser.add_argument("--total_step", type=int, default=150)

    args = parser.parse_args()
    print(args)
    main(args)


