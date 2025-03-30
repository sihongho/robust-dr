import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def main(args):
    # Load parameters from command-line arguments.
    ro_ave_mul = args.ro_ave_mul
    ro_max_mul = args.ro_max_mul
    non_ave_mul = args.non_ave_mul
    non_max_mul = args.non_max_mul
    total_step = args.total_step

    v1_nonrobust = []
    v1_robust = []
    for time in ro_ave_mul:
        path_ro_ave_mul = f"./experiments/{time}/results.json"
        # Load the results from JSON files.
        with open(path_ro_ave_mul, "r") as f:
            data_ro_ave_mul = json.load(f)
        # Read all V results.
        v1_nonrobust_raw = data_ro_ave_mul["all_V_nominal_nonrobust"]
        v1_robust_raw = data_ro_ave_mul["all_V_nominal_robust"]
        tmp_nonrobust_mean = []
        for i in range(total_step + 1):
            tmp_nonrobust_mean.append(np.mean(v1_nonrobust_raw[i]))
        v1_nonrobust.append(tmp_nonrobust_mean)
        tmp_robust_mean = []
        for i in range(total_step + 1):
            tmp_robust_mean.append(np.mean(v1_robust_raw[i]))
        v1_robust.append(tmp_robust_mean)

    v1_nonrobust_mean = np.mean(v1_nonrobust, axis=0)
    v1_nonrobust_std_dev = np.std(v1_nonrobust, axis=0)
    v1_robust_mean = np.mean(v1_robust, axis=0)
    v1_robust_std_dev = np.std(v1_robust, axis=0)

    v2_nonrobust = []
    v2_robust = []
    for time in ro_max_mul:
        path_ro_max_mul = f"./experiments/{time}/results.json"
        # Load the results from JSON files.
        with open(path_ro_max_mul, "r") as f:
            data_ro_max_mul = json.load(f)
        # Read all V results.
        v2_nonrobust_raw = data_ro_max_mul["all_V_nominal_nonrobust"]
        v2_robust_raw = data_ro_max_mul["all_V_nominal_robust"]
        tmp_nonrobust_mean = []
        for i in range(total_step + 1):
            tmp_nonrobust_mean.append(np.mean(v2_nonrobust_raw[i]))
        v2_nonrobust.append(tmp_nonrobust_mean)
        tmp_robust_mean = []
        for i in range(total_step + 1):
            tmp_robust_mean.append(np.mean(v2_robust_raw[i]))
        v2_robust.append(tmp_robust_mean)

    v2_nonrobust_mean = np.mean(v2_nonrobust, axis=0)
    v2_nonrobust_std_dev = np.std(v2_nonrobust, axis=0)
    v2_robust_mean = np.mean(v2_robust, axis=0)
    v2_robust_std_dev = np.std(v2_robust, axis=0)

    v3_nonrobust = []
    v3_robust = []
    for time in non_ave_mul:
        path_non_ave_mul = f"./experiments/{time}/results.json"
        # Load the results from JSON files.
        with open(path_non_ave_mul, "r") as f:
            data_non_ave_mul = json.load(f)
        # Read all V results.
        v3_nonrobust_raw = data_non_ave_mul["all_V_nominal_nonrobust"]
        v3_robust_raw = data_non_ave_mul["all_V_nominal_robust"]
        tmp_nonrobust_mean = []
        for i in range(total_step + 1):
            tmp_nonrobust_mean.append(np.mean(v3_nonrobust_raw[i]))
        v3_nonrobust.append(tmp_nonrobust_mean)
        tmp_robust_mean = []
        for i in range(total_step + 1):
            tmp_robust_mean.append(np.mean(v3_robust_raw[i]))
        v3_robust.append(tmp_robust_mean)

    v3_nonrobust_mean = np.mean(v3_nonrobust, axis=0)
    v3_nonrobust_std_dev = np.std(v3_nonrobust, axis=0)
    v3_robust_mean = np.mean(v3_robust, axis=0)
    v3_robust_std_dev = np.std(v3_robust, axis=0)
    
    v4_nonrobust = []
    v4_robust = []
    for time in non_max_mul:
        path_non_max_mul = f"./experiments/{time}/results.json"
        # Load the results from JSON files.
        with open(path_non_max_mul, "r") as f:
            data_non_max_mul = json.load(f)
        # Read all V results.
        v4_nonrobust_raw = data_non_max_mul["all_V_nominal_nonrobust"]
        v4_robust_raw = data_non_max_mul["all_V_nominal_robust"]
        tmp_nonrobust_mean = []
        for i in range(total_step + 1):
            tmp_nonrobust_mean.append(np.mean(v4_nonrobust_raw[i]))
        v4_nonrobust.append(tmp_nonrobust_mean)
        tmp_robust_mean = []
        for i in range(total_step + 1):
            tmp_robust_mean.append(np.mean(v4_robust_raw[i]))
        v4_robust.append(tmp_robust_mean)

    v4_nonrobust_mean = np.mean(v4_nonrobust, axis=0)
    v4_nonrobust_std_dev = np.std(v4_nonrobust, axis=0)
    v4_robust_mean = np.mean(v4_robust, axis=0)
    v4_robust_std_dev = np.std(v4_robust, axis=0)

    x = np.arange(total_step + 1)
    # plt.figure(figsize=(10, 6))

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # Plot nominal non-robust value function.
    axes[0].plot(x, v1_nonrobust_mean, label="MDTL-Avg", linestyle='-', color='b')
    axes[0].plot(x, v2_nonrobust_mean, label="MDTL-Max", linestyle=(0, (5, 7)), color='r')
    axes[0].plot(x, v3_nonrobust_mean, label="Non-robust DR", linestyle='-', color='g')
    axes[0].plot(x, v4_nonrobust_mean, label="Non-robust MDTL-Max", linestyle=(0, (5, 7)), color='orange')
    axes[0].fill_between(x, v1_nonrobust_mean - 10 * v1_nonrobust_std_dev, v1_nonrobust_mean + 10 * v1_nonrobust_std_dev, alpha=0.2, color='b')
    axes[0].fill_between(x, v2_nonrobust_mean - 10 * v2_nonrobust_std_dev, v2_nonrobust_mean + 10 * v2_nonrobust_std_dev, alpha=0.2, color='r')
    axes[0].fill_between(x, v3_nonrobust_mean - 10 * v3_nonrobust_std_dev, v3_nonrobust_mean + 10 * v3_nonrobust_std_dev, alpha=0.2, color='g')
    axes[0].fill_between(x, v4_nonrobust_mean - 10 * v4_nonrobust_std_dev, v4_nonrobust_mean + 10 * v4_nonrobust_std_dev, alpha=0.2, color='orange')
    axes[0].set_xlabel("Time Step", fontsize=20)
    axes[0].set_ylabel("Value Function", fontsize=20)
    axes[0].set_title("Target Domain Performance", fontsize=24)
    axes[0].tick_params(axis='x', labelsize=14)
    axes[0].tick_params(axis='y', labelsize=14)
    axes[0].legend(fontsize=18)

    # plt.plot(x, v1_nonrobust_mean, label="MDTL_Avg", linestyle='-', color='b')
    # plt.plot(x, v2_nonrobust_mean, label="MDTL_Max", linestyle=(0, (5, 7)), color='r')
    # plt.plot(x, v3_nonrobust_mean, label="Non-robust DR", linestyle='-', color='g')
    # plt.plot(x, v4_nonrobust_mean, label="Non-robust MDTL_Max", linestyle=(0, (5, 7)), color='orange')
    # plt.fill_between(x, v1_nonrobust_mean - 10 * v1_nonrobust_std_dev, v1_nonrobust_mean + 10 * v1_nonrobust_std_dev, alpha=0.2, color='b')
    # plt.fill_between(x, v2_nonrobust_mean - 10 * v2_nonrobust_std_dev, v2_nonrobust_mean + 10 * v2_nonrobust_std_dev, alpha=0.2, color='r')
    # plt.fill_between(x, v3_nonrobust_mean - 10 * v3_nonrobust_std_dev, v3_nonrobust_mean + 10 * v3_nonrobust_std_dev, alpha=0.2, color='g')
    # plt.fill_between(x, v4_nonrobust_mean - 10 * v4_nonrobust_std_dev, v4_nonrobust_mean + 10 * v4_nonrobust_std_dev, alpha=0.2, color='orange')
    # plt.xlabel("Time Step", fontsize=20)
    # plt.ylabel("Value Function", fontsize=20)
    # plt.title("Target Domain Performance", fontsize=24)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.legend(fontsize=18)
    # plt.savefig("plot1.png")
    # plt.show()

    # Plot nominal robust value function.
    axes[1].plot(x, v1_robust_mean, label="MDTL-Avg", linestyle='-', color='b')
    axes[1].plot(x, v2_robust_mean, label="MDTL-Max", linestyle=(0, (5, 7)), color='r')
    axes[1].plot(x, v3_robust_mean, label="Non-robust DR", linestyle='-', color='g')
    axes[1].plot(x, v4_robust_mean, label="Non-robust MDTL-Max", linestyle=(0, (5, 7)), color='orange')
    axes[1].fill_between(x, v1_robust_mean - 10 * v1_robust_std_dev, v1_robust_mean + 10 * v1_robust_std_dev, alpha=0.2, color='b')
    axes[1].fill_between(x, v2_robust_mean - 10 * v2_robust_std_dev, v2_robust_mean + 10 * v2_robust_std_dev, alpha=0.2, color='r')
    axes[1].fill_between(x, v3_robust_mean - 10 * v3_robust_std_dev, v3_robust_mean + 10 * v3_robust_std_dev, alpha=0.2, color='g')
    axes[1].fill_between(x, v4_robust_mean - 10 * v4_robust_std_dev, v4_robust_mean + 10 * v4_robust_std_dev, alpha=0.2, color='orange')
    axes[1].set_xlabel("Time Step", fontsize=20)
    axes[1].set_ylabel("Robust Value Function", fontsize=20)
    axes[1].set_title("Target Domain Performance with Model Uncertainty", fontsize=24)
    axes[1].tick_params(axis='x', labelsize=14)
    axes[1].tick_params(axis='y', labelsize=14)
    axes[1].legend(fontsize=18)
    # plt.figure(figsize=(10, 6))
    # plt.plot(x, v1_robust_mean, label="MDTL_Avg", linestyle='-', color='b')
    # plt.plot(x, v2_robust_mean, label="MDTL_Max", linestyle=(0, (5, 7)), color='r')
    # plt.plot(x, v3_robust_mean, label="Non-robust DR", linestyle='-', color='g')
    # plt.plot(x, v4_robust_mean, label="Non-robust MDTL_Max", linestyle=(0, (5, 7)), color='orange')
    # plt.fill_between(x, v1_robust_mean - 10 * v1_robust_std_dev, v1_robust_mean + 10 * v1_robust_std_dev, alpha=0.2, color='b')
    # plt.fill_between(x, v2_robust_mean - 10 * v2_robust_std_dev, v2_robust_mean + 10 * v2_robust_std_dev, alpha=0.2, color='r')
    # plt.fill_between(x, v3_robust_mean - 10 * v3_robust_std_dev, v3_robust_mean + 10 * v3_robust_std_dev, alpha=0.2, color='g')
    # plt.fill_between(x, v4_robust_mean - 10 * v4_robust_std_dev, v4_robust_mean + 10 * v4_robust_std_dev, alpha=0.2, color='orange')
    # plt.xlabel("Time Step", fontsize=20)
    # plt.ylabel("Robust Value Function", fontsize=20)
    # plt.title("Target Domain Performance with Model Uncertainty", fontsize=24)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.legend(fontsize=18)
    # plt.savefig("plot2.png")
    # plt.show()

    plt.tight_layout()
    plt.savefig("plot_adver.png")
    plt.show()

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Process all V results.")

    # Define individual arguments for each JSON file
    parser.add_argument("--ro_ave_mul", type=str, nargs='+', default="", required=True, help="Folder containing robust average multi-learn result")
    parser.add_argument("--ro_max_mul", type=str, nargs='+', default="", required=True, help="Folder containing robust max multi-learn result")
    parser.add_argument("--non_ave_mul", type=str, nargs='+', default="", required=True, help="Folder containing non-robust average multi-learn result")
    parser.add_argument("--non_max_mul", type=str, nargs='+', default="", required=True, help="Folder containing non-robust max multi-learn result")
    
    parser.add_argument("--total_step", type=int, default=1000, help="length of all_V")
    # Parse arguments
    args = parser.parse_args()
    print(args)
    main(args)