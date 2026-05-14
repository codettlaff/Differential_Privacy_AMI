import numpy as np
import os
from power_flow import RadialNetwork

class PrivateRadialNetwork(RadialNetwork):

    def __init__(self, name, save_folderpath, private_dss_filepath=None, original_dss_filepath=None, epsilon=None, appliance_power_bound=None):

        # Initialize From Private DSS File
        if private_dss_filepath:
            self.dss_filepath = private_dss_filepath
            super().__init__(name, private_dss_filepath)
            self.epsilon, self.B = self.read_privacy_params()
        else:
            self.dss_filepath = original_dss_filepath
            self.epsilon, self.B = epsilon, appliance_power_bound
            super().__init__(name, original_dss_filepath)
            self.write_privacy_params()

        # Initialize Base Class
        self.network_folderpath = os.path.join(save_folderpath)
        if not os.path.exists(self.network_folderpath): os.makedirs(self.network_folderpath)
        self.private_dss_filepath = os.path.join(save_folderpath, name + '.dss')

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
        super().C(i)

    # Set of all nodes downstream of node i.
    def D(self, i):
        super().D(i)

    # Set of all lines along the path from root to node i.
    def L(self, i):
        super().L(i)

    # Set of all lines connected to node i.
    def M(self, i):
        super().M(i)

    def lin_dist_flow(self):
        super().lin_dist_flow()

    def solve_dss(self):
        super().solve_dss()

    def power_flow_results(self, t=0, return_results=False, show=False, csv_folderpath=None):
        super().power_flow_results(t, return_results, show, csv_folderpath)

    def make_private_load_profile(self, load_profile, num_houses):
        b = 2 * self.B / self.epsilon
        noise = np.random.laplace(0, b, size=(num_houses, len(load_profile)))
        noisy_load_profile = load_profile + noise.sum(axis=0)
        return noisy_load_profile