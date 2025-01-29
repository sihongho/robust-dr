import json
import pandas as pd
import numpy as np
import argparse

def main(args):
    # Load parameters from command-line arguments.
    ro_ave_mul = args.ro_ave_mul
    ro_max_mul = args.ro_max_mul
    non_ave_mul = args.non_ave_mul
    non_max_mul = args.non_max_mul
    non_sin_nom = args.non_sin_nom

    # Initialize table data.
    table_data = {
        "Algorithm": ["Robust Average Multi-learn", "Robust Max Multi-learn", "Non-robust Average Multi-learn", "Non-robust Max Multi-learn", "Non-robust Single-learn Nominal"],
        "R_test=0.01": [],
        "R_test=0.03": [],
        "R_test=0.05": [],
        "R_test=0.07": [],
        "R_test=0.1": []
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

    table_data["R_test=0.01"].append(tmp_results_1[0])
    table_data["R_test=0.03"].append(tmp_results_1[1])
    table_data["R_test=0.05"].append(tmp_results_1[2])
    table_data["R_test=0.07"].append(tmp_results_1[3])
    table_data["R_test=0.1"].append(tmp_results_1[4])

    tmp_results_2 = []
    for date in ro_max_mul:
        path = f"./experiments/{date}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        V_nominal_robust = data["V_nominal_robust"]
        tmp_results_2.append(np.mean(V_nominal_robust))

    table_data["R_test=0.01"].append(tmp_results_2[0])
    table_data["R_test=0.03"].append(tmp_results_2[1])
    table_data["R_test=0.05"].append(tmp_results_2[2])
    table_data["R_test=0.07"].append(tmp_results_2[3])
    table_data["R_test=0.1"].append(tmp_results_2[4])

    tmp_results_3 = []
    for date in non_ave_mul:
        path = f"./experiments/{date}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        V_nominal_robust = data["V_nominal_robust"]
        tmp_results_3.append(np.mean(V_nominal_robust))

    table_data["R_test=0.01"].append(tmp_results_3[0])
    table_data["R_test=0.03"].append(tmp_results_3[1])
    table_data["R_test=0.05"].append(tmp_results_3[2])
    table_data["R_test=0.07"].append(tmp_results_3[3])
    table_data["R_test=0.1"].append(tmp_results_3[4])

    tmp_results_4 = []
    for date in non_max_mul:
        path = f"./experiments/{date}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        V_nominal_robust = data["V_nominal_robust"]
        tmp_results_4.append(np.mean(V_nominal_robust))

    table_data["R_test=0.01"].append(tmp_results_4[0])
    table_data["R_test=0.03"].append(tmp_results_4[1])
    table_data["R_test=0.05"].append(tmp_results_4[2])
    table_data["R_test=0.07"].append(tmp_results_4[3])
    table_data["R_test=0.1"].append(tmp_results_4[4])

    tmp_results_5 = []
    for date in non_sin_nom:
        path = f"./experiments/{date}/results.json"
        # Load the results from JSON files.
        with open(path, "r") as f:
            data = json.load(f)
        # Read all V results.
        V_nominal_robust = data["V_nominal_robust"]
        tmp_results_5.append(np.mean(V_nominal_robust))

    table_data["R_test=0.01"].append(tmp_results_5[0])
    table_data["R_test=0.03"].append(tmp_results_5[1])
    table_data["R_test=0.05"].append(tmp_results_5[2])
    table_data["R_test=0.07"].append(tmp_results_5[3])
    table_data["R_test=0.1"].append(tmp_results_5[4])

    df = pd.DataFrame(table_data)
    print(df)



    

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Process different R_test results.")

    # Define individual arguments for each JSON file
    parser.add_argument("--ro_ave_mul", type=str, default="", nargs='+', help="Folder containing robust average multi-learn result")
    parser.add_argument("--ro_max_mul", type=str, default="", nargs='+', help="Folder containing robust max multi-learn result")
    parser.add_argument("--non_ave_mul", type=str, default="", nargs='+', help="Folder containing non-robust average multi-learn result")
    parser.add_argument("--non_max_mul", type=str, default="", nargs='+', help="Folder containing non-robust max multi-learn result")
    parser.add_argument("--non_sin_nom", type=str, default="", nargs='+', help="Folder containing non-robust single-learn nominal result")

    # Parse arguments
    args = parser.parse_args()
    print(args)
    main(args)