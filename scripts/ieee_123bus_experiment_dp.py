import os

from differential_privacy_ami.dp_network import PrivateRadialNetwork

EPSILON = 1e3
NETWORK_NAME = 'redd_house_neighborhood'
APPLIANCE_POWER_BOUND = 5e3

ieee_123_filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'ieee_123bus', 'Master.dss')
save_folderpath = os.path.join(os.path.dirname(__file__), '..', 'experiments', NETWORK_NAME)
redd_folderpath = os.path.join(os.path.dirname(__file__), '..', 'data', 'redd')

network = PrivateRadialNetwork(NETWORK_NAME, dss_filepath=ieee_123_filepath, save_folderpath=save_folderpath, epsilon=EPSILON, appliance_power_bound=APPLIANCE_POWER_BOUND)

houses = network.load_redd_houses(redd_folderpath)
network.make_private_neighborhood(houses)
network.export_to_dss()
network.write_privacy_params()
network.lin_dist_flow()
ldf_node_results, ldf_line_results = network.power_flow_results(show=True, return_results=True)
network.solve_dss()
df_node_results, df_line_results = network.power_flow_results(show=True, return_results=True)
error = network.compute_error(ldf_node_results, df_node_results, ldf_line_results, df_line_results)

print('')