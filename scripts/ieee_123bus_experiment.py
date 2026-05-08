import os
import shutil
import numpy as np
import random
import matplotlib.pyplot as plt

from differential_privacy_ami.data_processing import load_data, process_data
from differential_privacy_ami.dp_power_flow import PrivateRadialNetwork

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
    network = PrivateRadialNetwork(NETWORK_NAME, dss_filepath=original_dss_filepath, B=B)

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
    dss_filepath = os.path.join(paths["experiments"], EXPERIMENT_NAME, NETWORK_NAME + '.dss')
    private_dss_filepath = os.path.join(paths["experiments"], EXPERIMENT_NAME, NETWORK_NAME + '_private.dss')
    network = PrivateRadialNetwork(NETWORK_NAME, dss_filepath=dss_filepath, private_dss_filepath=private_dss_filepath, B=B)
    return network

def solve_network(network, show=False):

    # Solve using Lin-Dist-Flow
    network.lin_dist_flow(tilde=False)  # Solve Using True P injection
    network.lin_dist_flow(tilde=True)  # Solve Using Noisy P injection
    e_p_line_ldf, e_v_line_ldf, p_acc_line_ldf, v_acc_node_ldf, p_acc_ldf, v_acc_ldf = network.empirical_accuracy()  # Get Accuracy using difference between results.

    # Solve Using OpenDSS (Nonlinear Dist-Flow)
    network.solve_dss(tilde=False)  # Solve Using True P injection
    network.solve_dss(tilde=True)  # Solve Using Noisy P injection
    e_p_line_dss, e_v_line_dss, p_acc_line_dss, v_acc_node_dss, p_acc_dss, v_acc_dss = network.empirical_accuracy()  # Get Accuracy using difference between results.

    # Theoretical Accuracy, Normalized by True Dist-Flow Results
    e_p_line_th, e_v_node_th, p_acc_line_th, v_acc_node_th, p_acc_th, v_acc_th = network.theoretical_accuracy(B, EPSILON)  # Theoretical accuracy

    if show:
        network.power_flow_results(show=True, tilde=False)
        network.power_flow_results(show=True, tilde=True)

    return{
        'e_p_line_th': e_p_line_th,
        'e_v_node_th': e_v_node_th,
        'p_acc_line_th': p_acc_line_th,
        'v_acc_node_th': v_acc_node_th,
        'p_acc_th': p_acc_th,
        'v_acc_th': v_acc_th,
        'e_p_line_ldf': e_p_line_ldf,
        'e_v_node_ldf': e_v_line_ldf,
        'p_acc_line_ldf': p_acc_line_ldf,
        'v_acc_node_ldf': v_acc_node_ldf,
        'p_acc_ldf': p_acc_ldf,
        'v_acc_ldf': v_acc_ldf,
        'e_p_line_dss': e_p_line_dss,
        'e_v_node_dss': e_v_line_dss,
        'p_acc_line_dss': p_acc_line_dss,
        'v_acc_node_dss': v_acc_node_dss,
        'p_acc_dss': p_acc_dss,
        'v_acc_dss': v_acc_dss
    }

def plot_absolute_error(network, results, plot_th=False, plot_ldf=False, plot_dss=False):

    paths = get_paths()
    save_folderpath = os.path.join(paths["experiments"], EXPERIMENT_NAME, 'plots')
    if not os.path.exists(save_folderpath): os.makedirs(save_folderpath)

    def get_line_data(key):
        data = results[key]
        dist = np.array([network.distance_to_root(j) for (i, j),data in data.items()])
        vals = np.array(list(data.values()))
        return dist, vals

    def get_node_data(key):
        data = results[key]
        dist = np.array([network.distance_to_root(i) for i in data.keys()])
        vals = np.array(list(data.values()))
        return dist, vals

    def plot_fit(x, y, label):
        if len(x) < 2: return
        coeffs = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = np.polyval(coeffs, x_fit)
        plt.plot(x_fit, y_fit, linestyle='--', label=f'{label} fit')

    # Plot Branch Power Accuracy
    if plot_th:
        dist_line, e_p_line_th = get_line_data('e_p_line_th')
        mask = e_p_line_th != 0
        plt.scatter(dist_line[mask], e_p_line_th[mask], label='TH')
        plot_fit(dist_line[mask], e_p_line_th[mask], 'TH')

    if plot_ldf:
        dist_line, e_p_line_ldf = get_line_data('e_p_line_ldf')
        plt.scatter(dist_line, e_p_line_ldf, label='LDF')
        plot_fit(dist_line, e_p_line_ldf, 'LDF')

    if plot_dss:
        dist_line, e_p_line_dss = get_line_data('e_p_line_dss')
        plt.scatter(dist_line, e_p_line_dss, label='DSS')
        plot_fit(dist_line, e_p_line_dss, 'DSS')

    plt.xlabel('Distance to Root')
    plt.ylabel('Branch Power Error')
    plt.legend()
    plt.title('Branch Power Error vs Distance')
    name = ""
    if plot_th: name += 'TH_'
    if plot_ldf: name += 'LDF_'
    if plot_dss: name += 'DSS'
    plt.savefig(os.path.join(save_folderpath, f'BPE_{name}.png'))

    # Plot Node Voltage Accuracy
    plt.figure()
    if plot_th:
        dist_node, e_v_node_th = get_node_data('e_v_node_th')
        plt.scatter(dist_node, e_v_node_th, label='TH')
        plot_fit(dist_node, e_v_node_th, 'TH')

    if plot_ldf:
        dist_node, e_v_node = get_node_data('e_v_node_ldf')
        plt.scatter(dist_node, e_v_node, label='LDF')
        plot_fit(dist_node, e_v_node, 'LDF')

    if plot_dss:
        dist_node, e_v_node = get_node_data('e_v_node_dss')
        plt.scatter(dist_node, e_v_node, label='DSS')
        plot_fit(dist_node, e_v_node, 'DSS')

    plt.xlabel("Distance to Root")
    plt.ylabel("Node Voltage Error")
    plt.legend()
    plt.title("Node Voltage Error vs Distance")
    name = ""
    if plot_th: name += 'TH_'
    if plot_ldf: name += 'LDF_'
    if plot_dss: name += 'DSS'
    plt.savefig(os.path.join(save_folderpath, f'NVE_{name}.png'))

    plt.show()

def plot_normalized_accuracy(network, results, plot_th=False, plot_ldf=False, plot_dss=False):

    paths = get_paths()
    save_folderpath = os.path.join(paths["experiments"], EXPERIMENT_NAME, 'plots')
    if not os.path.exists(save_folderpath): os.makedirs(save_folderpath)

    def get_line_data(key):
        data = results[key]
        dist = np.array([network.distance_to_root(j) for (i, j),data in data.items()])
        vals = np.array(list(data.values()))
        return dist, vals

    def get_node_data(key):
        data = results[key]
        dist = np.array([network.distance_to_root(i) for i in data.keys()])
        vals = np.array(list(data.values()))
        return dist, vals

    def plot_fit(x, y, label):
        if len(x) < 2: return
        coeffs = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = np.polyval(coeffs, x_fit)
        plt.plot(x_fit, y_fit, linestyle='--', label=f'{label} fit')

    # Plot Branch Power Accuracy

    if plot_th:
        dist_line, p_acc_th = get_line_data('p_acc_line_th')
        mask = p_acc_th != 0
        plt.scatter(dist_line[mask], p_acc_th[mask], label='TH')
        plot_fit(dist_line[mask], p_acc_th[mask], 'TH')

    if plot_ldf:
        dist_line, p_acc_ldf = get_line_data('p_acc_line_ldf')
        plt.scatter(dist_line, p_acc_ldf, label='LDF')
        plot_fit(dist_line, p_acc_ldf, 'LDF')

    if plot_dss:
        dist_line, p_acc_dss = get_line_data('p_acc_line_dss')
        plt.scatter(dist_line, p_acc_dss, label='DSS')
        plot_fit(dist_line, p_acc_dss, 'DSS')

    plt.xlabel('Distance to Root')
    plt.ylabel('Branch Power Accuracy')
    plt.legend()
    plt.title('Branch Power Accuracy vs Distance')
    name = ""
    if plot_th: name += 'TH_'
    if plot_ldf: name += 'LDF_'
    if plot_dss: name += 'DSS'
    plt.savefig(os.path.join(save_folderpath, f'BPA_{name}.png'))

    # Plot Node Voltage Accuracy
    plt.figure()

    if plot_th:
        dist_node, v_acc_th = get_node_data('v_acc_node_th')
        plt.scatter(dist_node, v_acc_th, label='TH')
        plot_fit(dist_node, v_acc_th, 'TH')

    if plot_ldf:
        dist_node, v_acc_ldf = get_node_data('v_acc_node_ldf')
        plt.scatter(dist_node, v_acc_ldf, label='LDF')
        plot_fit(dist_node, v_acc_ldf, 'LDF')

    if plot_dss:
        dist_node, v_acc_dss = get_node_data('v_acc_node_dss')
        plt.scatter(dist_node, v_acc_dss, label='DSS')
        plot_fit(dist_node, v_acc_dss, 'DSS')

    plt.xlabel("Distance to Root")
    plt.ylabel("Node Voltage Accuracy")
    plt.legend()
    plt.title("Node Voltage Accuracy vs Distance")
    name = ""
    if plot_th: name += 'TH_'
    if plot_ldf: name += 'LDF_'
    if plot_dss: name += 'DSS'
    plt.savefig(os.path.join(save_folderpath, f'NVA_{name}.png'))

    plt.show()

make_network() # Only need to run this function once.
network = load_network()
results = solve_network(network)

# Individual Results Absolute Error
# plot_absolute_error(network, results, plot_th=False, plot_ldf=False, plot_dss=True)
# plot_absolute_error(network, results, plot_th=False, plot_ldf=True, plot_dss=False)
# plot_absolute_error(network, results, plot_th=True, plot_ldf=False, plot_dss=False)

# Individual Results Normalized Accuracy
#plot_normalized_accuracy(network, results, plot_th=False, plot_ldf=False, plot_dss=True)
#plot_normalized_accuracy(network, results, plot_th=False, plot_ldf=True, plot_dss=False)
#plot_normalized_accuracy(network, results, plot_th=True, plot_ldf=False, plot_dss=False)

# Comparative Results Absolute Error
# Good - Absolute Error is Lower than Theoretical Error
# Questionable - LDF Error is much higher than DSS Error
# plot_absolute_error(network, results, plot_th=True, plot_ldf=True) # Validate Formulation
# plot_absolute_error(network, results, plot_th=True, plot_dss=True) # Compare Against Real Nonlinear Results
# plot_absolute_error(network, results, plot_ldf=True, plot_dss=True) # Compare Linear versus Nonlinear Results

# Comparative Results Normalized Accuracy
# BPA Good - Empirical Accuracy higher than Theoretical Accuracy
# NVA Not Good - Empirical Accuracy lower than Theoretical Accuracy
# Good - LDF BPA and DSS BPA exactly overlap.
# Good - LDF NVA and DSS NVA Approximately overlap
# plot_normalized_accuracy(network, results, plot_th=True, plot_ldf=True) # Validate Formulation
# plot_normalized_accuracy(network, results, plot_th=True, plot_dss=True) # Compare Against Real Nonlinear Results
# plot_normalized_accuracy(network, results, plot_ldf=True, plot_dss=True) # Compare Linear versus Nonlinear Results

print('')

