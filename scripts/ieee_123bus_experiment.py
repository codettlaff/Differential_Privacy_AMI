import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from differential_privacy_ami.power_flow import RadialNetwork

EXPERIMENT_NAME = 'ieee_123bus'
NETWORK_NAME = 'ieee_123bus'

def get_paths():
    base = os.path.join(os.path.dirname(__file__), '..')
    return {
        "data": os.path.join(base, 'data'),
        "experiments": os.path.join(base, 'experiments'),
        "scripts": os.path.join(base, 'scripts'),
        "ieee_123bus": os.path.join(base, 'data', 'ieee_123bus')
    }

def make_network():
    paths = get_paths()
    experiment_folderpath = os.path.join(paths['experiments'], EXPERIMENT_NAME)
    if os.path.exists(experiment_folderpath): shutil.rmtree(experiment_folderpath)
    os.makedirs(experiment_folderpath)
    # Copy Original DSS Folder
    source_folderpath = paths['ieee_123bus']
    destination_folderpath = os.path.join(experiment_folderpath, 'ieee_123bus_original')
    shutil.copytree(source_folderpath, destination_folderpath, dirs_exist_ok=True)
    # Load Network from Original IEEE123Bus System
    original_dss_filepath = os.path.join(destination_folderpath, 'Master.dss')
    network = RadialNetwork(NETWORK_NAME, dss_filepath=original_dss_filepath)
    dss_filepath = os.path.join(experiment_folderpath, NETWORK_NAME + '.dss')
    network.dss_filepath = dss_filepath
    network.export_to_dss()

def load_network():
    paths = get_paths()
    dss_filepath = os.path.join(paths["experiments"], EXPERIMENT_NAME, NETWORK_NAME + '.dss')
    network = RadialNetwork(NETWORK_NAME, dss_filepath=dss_filepath)
    return network

make_network()
network = load_network()

print('')

