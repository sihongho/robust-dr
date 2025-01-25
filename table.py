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
    ro_sin_nom = args.ro_sin_nom
    non_sin_nom = args.non_sin_nom
    ro_sin_ave = args.ro_sin_ave
    non_sin_ave = args.non_sin_ave

    # Initialize table data.
    table_data = {
        "Algorithm": [],
        "Nominal Non-robust": [],
        "Nominal Robust": [],
        "Average Non-robust": [], 
        "Average Robust": []
    }

    if ro_ave_mul != "":
        path_ro_ave_mul = f"./experiments/{ro_ave_mul}/results.json"
        # Load the results from JSON files.
        with open(path_ro_ave_mul, "r") as f1:
            data_ro_ave_mul = json.load(f1)
        # Read nominal non-robust value function.
        v1_nom_non = data_ro_ave_mul["V_nominal_nonrobust"]
        # Read nominal robust value function.
        v1_nom_ro = data_ro_ave_mul["V_nominal_robust"]
        # Read average non-robust value function.
        v1_ave_non = data_ro_ave_mul["V_avg_nonrobust"]
        # Read average robust value function.
        v1_ave_ro = data_ro_ave_mul["V_avg_robust"]
        # Add robust average multi-learn result to table.
        table_data["Algorithm"].append("Robust Average Multi-learn")
        table_data["Nominal Non-robust"].append(np.mean(v1_nom_non))
        table_data["Nominal Robust"].append(np.mean(v1_nom_ro))
        table_data["Average Non-robust"].append(np.mean(v1_ave_non))
        table_data["Average Robust"].append(np.mean(v1_ave_ro))

    if ro_max_mul != "":
        path_ro_max_mul = f"./experiments/{ro_max_mul}/results.json"
        # Load the results from JSON files.
        with open(path_ro_max_mul, "r") as f2:
            data_ro_max_mul = json.load(f2)
        # Read nominal non-robust value function.
        v2_nom_non = data_ro_max_mul["V_nominal_nonrobust"]
        # Read nominal robust value function.
        v2_nom_ro = data_ro_max_mul["V_nominal_robust"]
        # Read average non-robust value function.
        v2_ave_non = data_ro_max_mul["V_avg_nonrobust"]
        # Read average robust value function.
        v2_ave_ro = data_ro_max_mul["V_avg_robust"]
        # Add robust max multi-learn result to table.
        table_data["Algorithm"].append("Robust Max Multi-learn")
        table_data["Nominal Non-robust"].append(np.mean(v2_nom_non))
        table_data["Nominal Robust"].append(np.mean(v2_nom_ro))
        table_data["Average Non-robust"].append(np.mean(v2_ave_non))
        table_data["Average Robust"].append(np.mean(v2_ave_ro))

    if non_ave_mul != "":
        path_non_ave_mul = f"./experiments/{non_ave_mul}/results.json"
        # Load the results from JSON files.
        with open(path_non_ave_mul, "r") as f3:
            data_non_ave_mul = json.load(f3)
        # Read nominal non-robust value function.
        v3_nom_non = data_non_ave_mul["V_nominal_nonrobust"]
        # Read nominal robust value function.
        v3_nom_ro = data_non_ave_mul["V_nominal_robust"]
        # Read average non-robust value function.
        v3_ave_non = data_non_ave_mul["V_avg_nonrobust"]
        # Read average robust value function.
        v3_ave_ro = data_non_ave_mul["V_avg_robust"]
        # Add non-robust average multi-learn result to table.
        table_data["Algorithm"].append("Non-robust Avearge Multi-learn")
        table_data["Nominal Non-robust"].append(np.mean(v3_nom_non))
        table_data["Nominal Robust"].append(np.mean(v3_nom_ro))
        table_data["Average Non-robust"].append(np.mean(v3_ave_non))
        table_data["Average Robust"].append(np.mean(v3_ave_ro))

    if non_max_mul != "":
        path_non_max_mul = f"./experiments/{non_max_mul}/results.json"
        # Load the results from JSON files.
        with open(path_non_max_mul, "r") as f4:
            data_non_max_mul = json.load(f4)
        # Read nominal non-robust value function.
        v4_nom_non = data_non_max_mul["V_nominal_nonrobust"]
        # Read nominal robust value function.
        v4_nom_ro = data_non_max_mul["V_nominal_robust"]
        # Read average non-robust value function.
        v4_ave_non = data_non_max_mul["V_avg_nonrobust"]
        # Read average robust value function.
        v4_ave_ro = data_non_max_mul["V_avg_robust"]
        # Add non-robust max multi-learn result to table.
        table_data["Algorithm"].append("Non-robust Max Multi-learn")
        table_data["Nominal Non-robust"].append(np.mean(v4_nom_non))
        table_data["Nominal Robust"].append(np.mean(v4_nom_ro))
        table_data["Average Non-robust"].append(np.mean(v4_ave_non))
        table_data["Average Robust"].append(np.mean(v4_ave_ro))

    if ro_sin_nom != "":
        path_ro_sin_nom = f"./experiments/{ro_sin_nom}/results.json"
        # Load the results from JSON files.
        with open(path_ro_sin_nom, "r") as f5:
            data_ro_sin_nom = json.load(f5)
        # Read nominal non-robust value function.
        v5_nom_non = data_ro_sin_nom["V_nominal_nonrobust"]
        # Read nominal robust value function.
        v5_nom_ro = data_ro_sin_nom["V_nominal_robust"]
        # Read average non-robust value function.
        v5_ave_non = data_ro_sin_nom["V_avg_nonrobust"]
        # Read average robust value function.
        v5_ave_ro = data_ro_sin_nom["V_avg_robust"]
        # Add robust single-learn nominal result to table.
        table_data["Algorithm"].append("Robust Single-learn Nominal")
        table_data["Nominal Non-robust"].append(np.mean(v5_nom_non))
        table_data["Nominal Robust"].append(np.mean(v5_nom_ro))
        table_data["Average Non-robust"].append(np.mean(v5_ave_non))
        table_data["Average Robust"].append(np.mean(v5_ave_ro))

    if non_sin_nom != "":
        path_non_sin_nom = f"./experiments/{non_sin_nom}/results.json"
        # Load the results from JSON files.
        with open(path_non_sin_nom, "r") as f6:
            data_non_sin_nom = json.load(f6)
        # Read nominal non-robust value function.
        v6_nom_non = data_non_sin_nom["V_nominal_nonrobust"]
        # Read nominal robust value function.
        v6_nom_ro = data_non_sin_nom["V_nominal_robust"]
        # Read average non-robust value function.
        v6_ave_non = data_non_sin_nom["V_avg_nonrobust"]
        # Read average robust value function.
        v6_ave_ro = data_non_sin_nom["V_avg_robust"]
        # Add non-robust single-learn nominal result to table.
        table_data["Algorithm"].append("Non-robust Single-learn Nominal")
        table_data["Nominal Non-robust"].append(np.mean(v6_nom_non))
        table_data["Nominal Robust"].append(np.mean(v6_nom_ro))
        table_data["Average Non-robust"].append(np.mean(v6_ave_non))
        table_data["Average Robust"].append(np.mean(v6_ave_ro))

    if ro_sin_ave != "":
        path_ro_sin_ave = f"./experiments/{ro_sin_ave}/results.json"
        # Load the results from JSON files.
        with open(path_ro_sin_ave, "r") as f7:
            data_ro_sin_ave = json.load(f7)
        # Read nominal non-robust value function.
        v7_nom_non = data_ro_sin_ave["V_nominal_nonrobust"]
        # Read nominal robust value function.
        v7_nom_ro = data_ro_sin_ave["V_nominal_robust"]
        # Read average non-robust value function.
        v7_ave_non = data_ro_sin_ave["V_avg_nonrobust"]
        # Read average robust value function.
        v7_ave_ro = data_ro_sin_ave["V_avg_robust"]
        # Add robust single-learn average result to table.
        table_data["Algorithm"].append("Robust Single-learn Average")
        table_data["Nominal Non-robust"].append(np.mean(v7_nom_non))
        table_data["Nominal Robust"].append(np.mean(v7_nom_ro))
        table_data["Average Non-robust"].append(np.mean(v7_ave_non))
        table_data["Average Robust"].append(np.mean(v7_ave_ro))

    if non_sin_ave != "":
        path_non_sin_ave = f"./experiments/{non_sin_ave}/results.json"
        # Load the results from JSON files.
        with open(path_non_sin_ave, "r") as f8:
            data_non_sin_ave = json.load(f8)
        # Read nominal non-robust value function.
        v8_nom_non = data_non_sin_ave["V_nominal_nonrobust"]
        # Read nominal robust value function.
        v8_nom_ro = data_non_sin_ave["V_nominal_robust"]
        # Read average non-robust value function.
        v8_ave_non = data_non_sin_ave["V_avg_nonrobust"]
        # Read average robust value function.
        v8_ave_ro = data_non_sin_ave["V_avg_robust"]
        # Add non-robust single-learn average result to table.
        table_data["Algorithm"].append("Non-robust Single-learn Average")
        table_data["Nominal Non-robust"].append(np.mean(v8_nom_non))
        table_data["Nominal Robust"].append(np.mean(v8_nom_ro))
        table_data["Average Non-robust"].append(np.mean(v8_ave_non))
        table_data["Average Robust"].append(np.mean(v8_ave_ro))

    if table_data:
        df = pd.DataFrame(table_data)
        print(df)
    else:
        print("No valid JSON files provided. Table not created.")

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Process policy evaluation results.")

    # Define individual arguments for each JSON file
    parser.add_argument("--ro_ave_mul", type=str, default="", help="Folder containing robust average multi-learn result")
    parser.add_argument("--ro_max_mul", type=str, default="", help="Folder containing robust max multi-learn result")
    parser.add_argument("--non_ave_mul", type=str, default="", help="Folder containing non-robust average multi-learn result")
    parser.add_argument("--non_max_mul", type=str, default="", help="Folder containing non-robust max multi-learn result")
    parser.add_argument("--ro_sin_nom", type=str, default="", help="Folder containing robust single-learn nominal result")
    parser.add_argument("--non_sin_nom", type=str, default="", help="Folder containing non-robust single-learn nominal result")
    parser.add_argument("--ro_sin_ave", type=str, default="", help="Folder containing robust single-learn average result")
    parser.add_argument("--non_sin_ave", type=str, default="", help="Folder containing non-robust single-learn average result")

    # Parse arguments
    args = parser.parse_args()
    print(args)
    main(args)
