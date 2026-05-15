import networkx.algorithms.minors.contraction
import numpy as np
import random
import os

from .data_processing import load_data, process_data
from .power_flow import RadialNetwork

class PrivateRadialNetwork(RadialNetwork):

    def __init__(self, name, save_folderpath, dss_filepath, epsilon=None, appliance_power_bound=None):

        self.dss_filepath = dss_filepath
        if epsilon: self.epsilon = epsilon
        if appliance_power_bound: self.B = appliance_power_bound
        super().__init__(name, dss_filepath)

        # Initialize Base Class
        self.network_folderpath = os.path.join(save_folderpath)
        if not os.path.exists(self.network_folderpath): os.makedirs(self.network_folderpath)
        self.dss_filepath = os.path.join(save_folderpath, name + '.dss')
        self.export_to_dss()

    def read_privacy_params(self):
        epsilon = None
        appliance_power_bound = None
        with open(self.dss_filepath, 'r') as f:
            for line in f:
                if line.startswith('!'):
                    content = line[1:].strip()
                    value = float(content.split('=',1)[1].strip())
                    if content.startswith('EPSILON='):
                        epsilon = value
                    if content.startswith('APPLIANCE_POWER_BOUND='):
                        appliance_power_bound = value
        return epsilon, appliance_power_bound

    def write_privacy_params(self):
        new_epsilon_line = f"! EPSILON={self.epsilon}"
        new_B_line = f"! = APPLIANCE_POWER_BOUND={self.B}"
        with open(self.dss_filepath, 'r') as f:
            lines = f.readlines()
        updated_lines = []
        found_epsilon = False
        found_B = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("! EPSILON="):
                updated_lines.append(new_epsilon_line + "\n")
                found_epsilon = True
            elif stripped.startswith("! APPLIANCE_POWER_BOUND="):
                updated_lines.append(new_B_line + "\n")
                found_power = True
            else: updated_lines.append(line)

        header_lines = []
        if not found_epsilon:
            header_lines.append(new_epsilon_line + "\n")
        if not found_B:
            header_lines.append(new_B_line + "\n")

        with open(self.dss_filepath, 'w') as f:
            f.writelines(header_lines + updated_lines)

    def build_from_dss(self):
        super().build_from_dss()

    def export_to_dss(self):
        super().export_to_dss()

    # Set of all nodes along the path from root to node i.
    def C(self, i):
        return super().C(i)

    # Set of all nodes downstream of node i.
    def D(self, i):
        return super().D(i)

    # Set of all lines along the path from root to node i.
    def L(self, i):
        return super().L(i)

    # Set of all lines connected to node i.
    def M(self, i):
        return super().M(i)

    def lin_dist_flow(self):
        super().lin_dist_flow()

    def solve_dss(self):
        super().solve_dss()

    def power_flow_results(self, t=0, return_results=False, show=False, csv_folderpath=None):
        if return_results: return super().power_flow_results(t, return_results, show, csv_folderpath)
        else: super().power_flow_results(t, return_results, show, csv_folderpath)

    def compute_error(self, node_results_1, node_results_2, line_results_1, line_results_2):
        return super().compute_error(node_results_1, node_results_2, line_results_1, line_results_2)

    def make_private_load_profile(self, load_profile, num_houses):
        b = 2 * self.B / self.epsilon
        noise = np.random.laplace(0, b, size=(num_houses, len(load_profile)))
        noisy_load_profile = load_profile + noise.sum(axis=0)
        return noisy_load_profile

    def load_redd_houses(self, redd_folderpath, T_limit=None):
        filepaths = [
            os.path.join(redd_folderpath, f)
            for f in os.listdir(redd_folderpath)
            if f.endswith(".mat") and "HF" not in f
        ]
        houses = []
        for k in range(6):
            data = process_data(load_data(filepaths[k]))
            P = data['Y']
            if T_limit: P = P[:T_limit]
            houses.append(P)
        return houses

    def assign_houses_to_load(self, houses, desired_load_power):
        num_houses = 0
        n = min(len(h) for h in houses)
        load_profile = np.zeros(n)

        while np.max(load_profile) < desired_load_power:
            i = random.randrange(len(houses))
            temp_profile = load_profile + houses[i][:n]
            temp = np.max(temp_profile)
            if temp > desired_load_power:
                num_houses += 1
                factor = desired_load_power / np.max(temp)
                scaled_load_profile = houses[i][:n] * factor
                load_profile = load_profile + scaled_load_profile
                break
            else:
                num_houses += 1
                load_profile += houses[i][:n]

        return num_houses, load_profile

    def make_private_neighborhood(self, houses):

        P_loads_original = self.P
        Q_loads_original = self.Q
        P_loads = {}
        Q_loads = {}
        P_loads_tilde = {}
        Q_loads_tilde = {}

        n_houses_list = []

        for node, profile in P_loads_original.items():
            P_max = np.max(profile)
            Q_max = np.max(Q_loads_original[node])
            ratio = 0.0 if np.isclose(P_max, 0.0) else Q_max / P_max
            # Regular Neighborhood
            n_houses, load_profile = self.assign_houses_to_load(houses, P_max)
            n_houses_list.append(n_houses)
            P_loads[node] = load_profile
            Q_loads[node] = load_profile * ratio
            # Private Neighborhood
            P_loads_tilde[node] = self.make_private_load_profile(load_profile, n_houses)
            Q_loads_tilde[node] = P_loads_tilde[node] * ratio

        self.P = P_loads
        self.Q = Q_loads
        self.P_tilde = P_loads_tilde
        self.Q_tilde = Q_loads_tilde
