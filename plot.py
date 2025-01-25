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
    ro_sin_nom = args.ro_sin_nom
    non_sin_nom = args.non_sin_nom
    ro_sin_ave = args.ro_sin_ave
    non_sin_ave = args.non_sin_ave
    total_step = args.total_step
    x = np.arange(1, total_step + 1)
    plt.figure(figsize=(14, 6))

    if ro_ave_mul != "":
        path_ro_ave_mul = f"./experiments/{ro_ave_mul}/results.json"
        # Load the results from JSON files.
        with open(path_ro_ave_mul, "r") as f1:
            data_ro_ave_mul = json.load(f1)
        # Read all Q results.
        q1_raw = data_ro_ave_mul["all_Q"]
        q1_mean = []
        for q in q1_raw:
            q1_mean.append(np.mean(q))
        plt.plot(x, q1_mean, label="Robust Average Multi-learn")
    
    if ro_max_mul != "":
        path_ro_max_mul = f"./experiments/{ro_max_mul}/results.json"
        # Load the results from JSON files.
        with open(path_ro_max_mul, "r") as f2:
            data_ro_max_mul = json.load(f2)
        # Read all Q results.
        q2_raw = data_ro_max_mul["all_Q"]
        q2_mean = []
        for q in q2_raw:
            q2_mean.append(np.mean(q))
        plt.plot(x, q2_mean, label="Robust Max Multi-learn")

    if non_ave_mul != "":
        path_non_ave_mul = f"./experiments/{non_ave_mul}/results.json"
        # Load the results from JSON files.
        with open(path_non_ave_mul, "r") as f3:
            data_non_ave_mul = json.load(f3)
        # Read all Q results.
        q3_raw = data_non_ave_mul["all_Q"]
        q3_mean = []
        for q in q3_raw:
            q3_mean.append(np.mean(q))
        plt.plot(x, q3_mean, label="Non-robust Average Multi-learn")

    if non_max_mul != "":
        path_non_max_mul = f"./experiments/{non_max_mul}/results.json"
        # Load the results from JSON files.
        with open(path_non_max_mul, "r") as f4:
            data_non_max_mul = json.load(f4)
        # Read all Q results.
        q4_raw = data_non_max_mul["all_Q"]
        q4_mean = []
        for q in q4_raw:
            q4_mean.append(np.mean(q))
        plt.plot(x, q4_mean, label="Non-robust Max Multi-learn")

    if ro_sin_nom != "":
        path_ro_sin_nom = f"./experiments/{ro_sin_nom}/results.json"
        # Load the results from JSON files.
        with open(path_ro_sin_nom, "r") as f5:
            data_ro_sin_nom = json.load(f5)
        # Read all Q results.
        q5_raw = data_ro_sin_nom["all_Q"]
        q5_mean = []
        for q in q5_raw:
            q5_mean.append(np.mean(q))
        # plt.plot(x, q5_mean, label="Robust Single-learn Nominal")

    if non_sin_nom != "":
        path_non_sin_nom = f"./experiments/{non_sin_nom}/results.json"
        # Load the results from JSON files.
        with open(path_non_sin_nom, "r") as f6:
            data_non_sin_nom = json.load(f6)
        # Read all Q results.
        q6_raw = data_non_sin_nom["all_Q"]
        q6_mean = []
        for q in q6_raw:
            q6_mean.append(np.mean(q))
        # plt.plot(x, q6_mean, label="Non-robust Single-learn Nominal")

    if ro_sin_ave != "":
        path_ro_sin_ave = f"./experiments/{ro_sin_ave}/results.json"
        # Load the results from JSON files.
        with open(path_ro_sin_ave, "r") as f7:
            data_ro_sin_ave = json.load(f7)
        # Read all Q results.
        q7_raw = data_ro_sin_ave["all_Q"]
        q7_mean = []
        for q in q7_raw:
            q7_mean.append(np.mean(q))
        # plt.plot(x, q7_mean, label="Robust Single-learn Average")
    
    if non_sin_ave != "":
        path_non_sin_ave = f"./experiments/{non_sin_ave}/results.json"
        # Load the results from JSON files.
        with open(path_non_sin_ave, "r") as f8:
            data_non_sin_ave = json.load(f8)
        # Read all Q results.
        q8_raw = data_non_sin_ave["all_Q"]
        q8_mean = []
        for q in q8_raw:
            q8_mean.append(np.mean(q))
        # plt.plot(x, q8_mean, label="Non-robust Single-learn Average")
    
    # Labels and Title
    plt.xlabel("Epoch")
    plt.ylabel("Q")
    plt.title("All_Q")

    # Add Legend
    plt.legend()

    # Display the plot
    plt.show()

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Process all Q results.")

    # Define individual arguments for each JSON file
    parser.add_argument("--ro_ave_mul", type=str, default="", help="Folder containing robust average multi-learn result")
    parser.add_argument("--ro_max_mul", type=str, default="", help="Folder containing robust max multi-learn result")
    parser.add_argument("--non_ave_mul", type=str, default="", help="Folder containing non-robust average multi-learn result")
    parser.add_argument("--non_max_mul", type=str, default="", help="Folder containing non-robust max multi-learn result")
    parser.add_argument("--ro_sin_nom", type=str, default="", help="Folder containing robust single-learn nominal result")
    parser.add_argument("--non_sin_nom", type=str, default="", help="Folder containing non-robust single-learn nominal result")
    parser.add_argument("--ro_sin_ave", type=str, default="", help="Folder containing robust single-learn average result")
    parser.add_argument("--non_sin_ave", type=str, default="", help="Folder containing non-robust single-learn average result")
    parser.add_argument("--total_step", type=int, default=5000, help="length of all_Q")
    # Parse arguments
    args = parser.parse_args()
    print(args)
    main(args)
