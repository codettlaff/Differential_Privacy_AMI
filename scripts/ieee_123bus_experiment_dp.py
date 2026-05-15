import os
import copy

from differential_privacy_ami.dp_network import PrivateRadialNetwork

EPSILON = 1e3
NETWORK_NAME = 'redd_house_neighborhood'
APPLIANCE_POWER_BOUND = 5e3

ieee_123_filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'ieee_123bus', 'Master.dss')
save_folderpath = os.path.join(os.path.dirname(__file__), '..', 'experiments', NETWORK_NAME)
redd_folderpath = os.path.join(os.path.dirname(__file__), '..', 'data', 'redd')

def make_redd_neighboorhood_network(ieee_123_filepath, save_folderpath, redd_folderpath):

    ieee_123bus_network = PrivateRadialNetwork('ieee_123bus', dss_filepath=ieee_123_filepath, save_folderpath=save_folderpath, epsilon=EPSILON, appliance_power_bound=APPLIANCE_POWER_BOUND)

    houses = ieee_123bus_network.load_redd_houses(redd_folderpath)
    P_loads, Q_loads, P_loads_tilde, Q_loads_tilde = ieee_123bus_network.make_private_neighborhood_loads(houses)
    T_new = len(P_loads)

    redd_neighborhood = copy.deepcopy(ieee_123bus_network)
    redd_neighborhood.name = NETWORK_NAME
    redd_neighborhood.T = T_new
    redd_neighborhood.dss_filepath = os.path.join(save_folderpath, redd_neighborhood.name + '.dss')
    redd_neighborhood.P = P_loads
    redd_neighborhood.Q = Q_loads
    redd_neighborhood.export_to_dss()

    redd_neighborhood_private = copy.deepcopy(ieee_123bus_network)
    redd_neighborhood_private.name = NETWORK_NAME + '_private'
    redd_neighborhood_private.T = T_new
    redd_neighborhood_private.dss_filepath = os.path.join(save_folderpath, redd_neighborhood_private.name + '.dss')
    redd_neighborhood_private.P = P_loads_tilde
    redd_neighborhood_private.Q = Q_loads_tilde
    redd_neighborhood_private.export_to_dss()

# make_redd_neighboorhood_network(ieee_123_filepath, save_folderpath, redd_folderpath)

redd_neighborhood_dss_filepath = os.path.join(save_folderpath, NETWORK_NAME + '.dss')
redd_neighborhood_private_dss_filepath = os.path.join(save_folderpath, NETWORK_NAME + '_private.dss')

redd_neighborhood = PrivateRadialNetwork(NETWORK_NAME, dss_filepath=redd_neighborhood_dss_filepath, save_folderpath=save_folderpath, epsilon=EPSILON, appliance_power_bound=APPLIANCE_POWER_BOUND)
redd_neighborhood_private = PrivateRadialNetwork(NETWORK_NAME, dss_filepath=redd_neighborhood_private_dss_filepath, save_folderpath=save_folderpath, epsilon=EPSILON, appliance_power_bound=APPLIANCE_POWER_BOUND)

redd_neighborhood.solve_dss()
node_results, line_results = redd_neighborhood.power_flow_results(show=True, return_results=True)
redd_neighborhood_private.solve_dss()
node_results_private, line_results_private = redd_neighborhood_private.power_flow_results(show=True, return_results=True)

error = redd_neighborhood_private.compute_error(node_results, node_results_private, line_results, line_results_private)

print('')