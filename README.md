# Energy Management and Optimization
This project focuses on managing and optimizing energy generation and storage using renewable resources such as wind and solar power. The project includes functions for energy management and optimization using a genetic algorithm, as well as decoding and interpreting the results.

# Table of Contents
*Installation*
Usage
Energy Management
Optimization with Genetic Algorithm
Decoding the Results
References
Installation
Clone the repository or download the code files.
Open the code in Google Colab or any other Python environment that supports Jupyter notebooks.
Ensure the following Python packages are installed:
sh
Copy code
pip install numpy pandas geneticalgorithm tabulate
Usage
Upload the Excel File: The script expects an Excel file containing energy data. Upload the file when prompted in Google Colab.
Run the Code: Execute the cells in the notebook to perform energy management calculations, optimize the system parameters, and decode the results.
Energy Management
The energy management function processes the input data and computes various metrics to handle energy generation, storage, and evacuation efficiently.

Code Overview
python
Copy code
import numpy as np
import pandas as pd
from google.colab import files

# Upload the Excel file
uploaded = files.upload()

# Get the filename
filename = list(uploaded.keys())[0]

# Read the Excel file into a DataFrame
df = pd.read_excel(filename)

year = np.array(df.iloc[1:, 0])
month = np.array(df.iloc[1:, 1])
day = np.array(df.iloc[1:, 2])
time_values = np.array([time.strftime('%H:%M') for time in df.iloc[1:, 3]])
wind_profile = np.array(df.iloc[1:, 4], dtype=float)
solar_profile = np.array(df.iloc[1:, 5], dtype=float)
Energy Management Function
python
Copy code
def energy_management(time_values, generation, a, planted_capacity, battery_soc, battery_efficiency):
    # Implementation of energy management logic
    # ...
    return {
        "Demand Values": demand_values,
        "Generation to Evacuate": generation_to_evacuate,
        "Deficit Values": deficit_values,
        "Excess Energy Values": excess_energy_values,
        "Cumulative Charge List": cumulative_charge_list,
        "Hourly Storage Results": hourly_storage_results,
        "Withdrawal from Storage List": withdrawal_from_storage_list,
        "Storage Loss List": storage_loss_list,
        "Excess After Battery": excess_after_battery,
        "Evacuated Post Charging List": evacuated_post_charging_list,
        "Final Loss List": final_loss_list,
        "DFR without Market Purchase": DFR_without_MarketPurchase_list,
        "Loss 1 List": loss_1_list,
        "Loss 2 List": loss_2_list 
    }
Optimization with Genetic Algorithm
The optimization function uses a genetic algorithm to find the optimal values for wind capacity, solar capacity, and battery state of charge.

Code Overview
python
Copy code
from geneticalgorithm import geneticalgorithm as ga

def objective_function_2(x):
    # Objective function for optimization
    # ...
    return -composite_objective

def optimize_energy_model_2():
    varbound = np.array([[0, 500], [0, 500], [0.2, 0.3], [0, 5], [0, 5]])
    algorithm_param = {
        'max_num_iteration': 14,
        'population_size': 15,
        'mutation_probability': 0.75,
        'parents_portion': 0.5,
        'crossover_probability': 0.8,
        'mutation_strength': 0.03,
        'elit_ratio': 0.1,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': 150
    }

    model = ga(function=objective_function_2, dimension=5, variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param)
    model.run()
    return model.output_dict['variable']

# Run optimization
best_values = optimize_energy_model_2()
print("Optimal values:", best_values)
Decoding the Results
After obtaining the optimal values from the genetic algorithm, the results are decoded and analyzed. This involves calculating various metrics such as correlation, penalties, and overall costs.

Code Overview
python
Copy code
def compute_correlation(array1, array2):
    correlation = np.corrcoef(array1, array2)[0, 1]
    return correlation

def calculate_penalty(dfr_values):
    total_penalty = 0
    for dfr in dfr_values:
        if (dfr < 0.9):
            penalty_factor = 1 * (0.9 - dfr)
            total_penalty += penalty_factor
    return total_penalty

def print_optimal_results(best_values):
    wind_capacity, solar_capacity, battery_soc, q_index, r_index = best_values
    hourly_wind_generation = (wind_profile * wind_capacity) / 50
    hourly_solar_generation = (solar_profile * solar_capacity) / 50
    combined_generation = np.add(hourly_wind_generation, hourly_solar_generation)

    q_rounded = int(q_index + 0.5) if q_index % 1 >= 0.5 else int(q_index)
    r_rounded = int(r_index + 0.5) if r_index % 1 >= 0.5 else int(r_index)

    q_times = [f'{i:02d}:00' for i in range(5 + q_rounded, min(12, 5 + q_rounded + 2))]
    r_times = [f'{i:02d}:00' for i in range(17 + r_rounded, min(24, 17 + r_rounded + 2))]
    time_intervals = q_times + r_times

    results = energy_management(time_values, combined_generation, time_intervals, planted_capacity, battery_soc, battery_efficiency)

    dfr_values = results['DFR without Market Purchase']
    nonzero_dfr_values = [dfr for dfr in dfr_values if dfr != 0]
    penalty = calculate_penalty(nonzero_dfr_values)

    if len(nonzero_dfr_values) > 0:
        dfr = np.mean(nonzero_dfr_values)
    else:
        dfr = 0

    cost_wind = wind_capacity * 5
    cost_solar = solar_capacity * 3
    battery_cost = 300 * battery_soc * 4
    total_cost = cost_wind + cost_solar + battery_cost

    optimal_results = {
        "Optimal Time Intervals": time_intervals,
        "Optimal DFR": dfr,
        "Wind Capacity": wind_capacity,
        "Solar Capacity": solar_capacity,
        "Battery SOC": battery_soc,
        "Penalty": penalty,
        "Total Cost": total_cost
    }

    return optimal_results

from tabulate import tabulate

best_values_1 = [500, 335, 0.25, 2, 1]
best_values_2 = best_values_a
best_values_3 = best_values_b

result_1 = print_optimal_results(best_values_1)
result_2 = print_optimal_results(best_values_2)
result_3 = print_optimal_results(best_values_3)

results = [result_1, result_2, result_3]

headers = list(result_1.keys())
rows = [list(result.values()) for result in results]

print(tabulate(rows, headers=headers, tablefmt="grid"))
print('Penalty reduced:', result_1["Penalty"] - result_2["Penalty"])
print('Cost difference:', abs(result_1["Total Cost"] - result_2["Total Cost"]))
Decoding Process
Correlation Calculation: Computes the correlation between different energy values.
Penalty Calculation: Calculates penalties based on deviation from desired DFR values.
Result Interpretation: Interprets the optimal results and calculates relevant metrics such as total cost and penalty.
References
Genetic Algorithm implementation: geneticalgorithm
Tabulate library: tabulate






