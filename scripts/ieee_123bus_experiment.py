import os
import shutil
import numpy as np
import random
import matplotlib.pyplot as plt

from differential_privacy_ami.data_processing import load_data, process_data
from differential_privacy_ami.power_flow import RadialNetwork

EXPERIMENT_NAME = 'ieee_123bus'
T_SET = 100 # Limit Data Timesteps
EPSILON_VALUES = np.linspace(50, 1000, 10)
EPSILON = 1000

NETWORK_NAME = 'redd_house_neighborhood'
B = 5e3 # HVAC Rated Power

def get_paths():
    base = os.path.join(os.path.dirname(__file__), '..')
    return {
        "data": os.path.join(base, 'data'),
        "experiments": os.path.join(base, 'experiments'),
        "scripts": os.path.join(base, 'scripts'),
        "ieee_123bus": os.path.join(base, 'data', 'ieee_123bus'),
        "redd": os.path.join(base, 'data', 'redd'),
    }

def load_redd_houses():
    paths = get_paths()
    files = [
        os.path.join(paths["redd"], f)
        for f in os.listdir(paths["redd"])
        if f.endswith(".mat") and "HF" not in f
    ]
    houses = []
    for k in range(6):
        data = process_data(load_data(files[k]))
        P = data["Y"]
        if T_SET:
            P = P[:T_SET]
        houses.append(P) # List of 1D Numpy Arrays
    return houses

def assign_houses_to_load(houses, desired_load_power):
    num_houses = 0
    n = len(houses[0])
    load_profile = np.zeros(n)

    while np.max(load_profile) < desired_load_power:
        i = random.randrange(len(houses))
        temp_profile = load_profile + houses[i]
        temp = np.max(temp_profile)
        if temp > desired_load_power:
            num_houses += 1
            factor = desired_load_power / np.max(temp)
            scaled_load_profile = houses[i] * factor
            load_profile = load_profile + scaled_load_profile
            break
        else:
            num_houses += 1
            load_profile += houses[i]

    return num_houses, load_profile

def make_private_load_profile(load_profile, num_houses, B, epsilon):
    b = 2 * B / epsilon
    noise = np.random.laplace(0, b, size=(num_houses, len(load_profile)))
    noisy_load_profile = load_profile + noise.sum(axis=0)
    return noisy_load_profile

def make_network():

    paths = get_paths()

    # Make Experiment Folder
    experiment_folderpath = os.path.join(paths["experiments"], EXPERIMENT_NAME)
    if os.path.exists(experiment_folderpath): shutil.rmtree(experiment_folderpath)
    os.makedirs(experiment_folderpath)

    # Copy Original DSS Folder
    source_folderpath = paths["ieee_123bus"]
    destination_folderpath = os.path.join(experiment_folderpath, 'ieee_123bus_original')
    shutil.copytree(source_folderpath, destination_folderpath, dirs_exist_ok=True)

    # Load Network from Original IEEE123Bus System
    original_dss_filepath = os.path.join(destination_folderpath, 'Master.dss')
    network = RadialNetwork(NETWORK_NAME, dss_filepath=original_dss_filepath, B=B)

    # Create REDD Neighborhood Network
    houses = load_redd_houses()

    P_loads_original = network.P # Original IEEE123Bus Loads
    Q_loads_original = network.Q # Original IEEE123Bus Loads
    P_loads = {} # REDD Neighborhood Loads
    Q_loads = {} # REDD Neighborhood Loads
    P_tilde = {} # Private Neighborhood Loads
    Q_tilde = {} # Private Neighborhood Loads
    n_houses_list = []

    for node, profile in P_loads_original.items():

        P_max = np.max(profile) # Desired Real Power Load from Original IEEE123Bus System
        Q_max = np.max(Q_loads_original[node]) # Corresponding Reactive Power Load
        ratio = 0.0 if np.isclose(P_max, 0.0) else Q_max / P_max

        # Regular Neighborhood
        n_houses, load_profile = assign_houses_to_load(houses, P_max) # Create Load by Aggregating REDD Houses
        n_houses_list.append(n_houses) # Remember how many houses assigned to each load (Important for privacy).
        P_loads[node] = load_profile
        Q_loads[node] = load_profile * ratio

        # Private Neighborhood
        P_tilde[node] = make_private_load_profile(load_profile, n_houses, B, EPSILON) # Apply Differential Privacy
        Q_tilde[node] = Q_loads[node] # Noise applied only to Active Power

    network.P = P_loads
    network.Q = Q_loads
    network.P_tilde = P_tilde
    network.Q_tilde = Q_tilde

    # Save Network - Export To DSS Files
    dss_filepath = os.path.join(experiment_folderpath, NETWORK_NAME + '.dss')
    network.dss_filepath = dss_filepath
    network.export_to_dss(tilde=False) # Writes Entire System in one DSS File
    dss_filepath = os.path.join(experiment_folderpath, NETWORK_NAME + '_private.dss')
    network.dss_filepath = dss_filepath
    network.export_to_dss(tilde=True) # Writes Entire System in one DSS File

def load_network():
    paths = get_paths()
    dss_filepath = os.path.join(paths["experiments"], EXPERIMENT_NAME, NETWORK_NAME + '_private.dss')
    network = RadialNetwork(NETWORK_NAME, dss_filepath=dss_filepath)
    P_tilde = network.P
    Q_tilde = network.Q
    dss_filepath = os.path.join(paths["experiments"], EXPERIMENT_NAME, NETWORK_NAME + '.dss')
    network = RadialNetwork(NETWORK_NAME, dss_filepath=dss_filepath)
    network.P_tilde = P_tilde
    network.Q_tilde = Q_tilde
    return network

def solve_network(network, show=False):

    # Get Results
    p_acc_line_th, v_acc_node_th, p_acc_th, v_acc_th = network.theoretical_accuracy(B, EPSILON)  # Theoretical accuracy - does not require solving power flow.

    # Solve using Lin-Dist-Flow
    network.lin_dist_flow(tilde=False)  # Solve Using True P injection
    network.lin_dist_flow(tilde=True)  # Solve Using Noisy P injection
    p_acc_line_ldf, v_acc_node_ldf, p_acc_ldf, v_acc_ldf = network.empirical_accuracy()  # Get Accuracy using difference between results.

    # Solve Using OpenDSS (Nonlinear Dist-Flow)
    network.solve_dss(tilde=False)  # Solve Using True P injection
    network.solve_dss(tilde=True)  # Solve Using Noisy P injection
    p_acc_line_dss, v_acc_node_dss, p_acc_dss, v_acc_dss = network.empirical_accuracy()  # Get Accuracy using difference between results.

    if show:
        network.power_flow_results(show=True, tilde=False)
        network.power_flow_results(show=True, tilde=True)

    return{
        'p_acc_line_th': p_acc_line_th,
        'v_acc_node_th': v_acc_node_th,
        'p_acc_th': p_acc_th,
        'v_acc_th': v_acc_th,
        'p_acc_line_ldf': p_acc_line_ldf,
        'v_acc_node_ldf': v_acc_node_ldf,
        'p_acc_ldf': p_acc_ldf,
        'v_acc_ldf': v_acc_ldf,
        'p_acc_line_dss': p_acc_line_dss,
        'v_acc_node_dss': v_acc_node_dss,
        'p_acc_dss': p_acc_dss,
        'v_acc_dss': v_acc_dss
    }

# make_network() # Only need to run this function once.
network = load_network()
results = solve_network(network, show=True)

print('')

