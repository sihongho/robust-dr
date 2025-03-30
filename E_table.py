import json
import pandas as pd
import numpy as np
import argparse

def main(args):
    # Load parameters from command-line arguments.
    ro_ave_mul = args.ro_ave_mul
    ro_max_mul = args.ro_max_mul

    # Initialize table data.
    table_data = {
        "Algorithm": ["Robust Average Multi-learn", "Robust Max Multi-learn"],
        "E=1": [],
        "E=2": [],
        "E=3": [],
        "E=4": [],
        "E=5": []
    }

    tmp_results_1 = []
    for date in ro_ave_mul:
        path = f"./experiments/{date}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        V_nominal_robust = data["V_nominal_robust"]
        tmp_results_1.append(np.mean(V_nominal_robust))

    table_data["E=1"].append(tmp_results_1[0])
    table_data["E=2"].append(tmp_results_1[1])
    table_data["E=3"].append(tmp_results_1[2])
    table_data["E=4"].append(tmp_results_1[3])
    table_data["E=5"].append(tmp_results_1[4])

    tmp_results_2 = []
    for date in ro_max_mul:
        path = f"./experiments/{date}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        V_nominal_robust = data["V_nominal_robust"]
        tmp_results_2.append(np.mean(V_nominal_robust))

    table_data["E=1"].append(tmp_results_2[0])
    table_data["E=2"].append(tmp_results_2[1])
    table_data["E=3"].append(tmp_results_2[2])
    table_data["E=4"].append(tmp_results_2[3])
    table_data["E=5"].append(tmp_results_2[4])

    df = pd.DataFrame(table_data)
    print(df)

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Process different R_test results.")

    # Define individual arguments for each JSON file
    parser.add_argument("--ro_ave_mul", type=str, default="", nargs='+', help="Folder containing robust average multi-learn result")
    parser.add_argument("--ro_max_mul", type=str, default="", nargs='+', help="Folder containing robust max multi-learn result")

    # Parse arguments
    args = parser.parse_args()
    print(args)
    main(args)