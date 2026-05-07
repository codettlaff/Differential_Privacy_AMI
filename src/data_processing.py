import numpy as np
import scipy.io
from sklearn.model_selection import KFold

def load_data(filepath):

    raw = scipy.io.loadmat(filepath)

    if 'labelInp' not in raw or 'labelOut' not in raw:
        raise ValueError('Missing Input or Output Labels')

    app_power_labels = [l.strip() for l in raw['labelInp'][2:]]
    agg_power_labels = [l.strip() for l in raw['labelOut'][2:]]
    agg_power_units = [l.strip() for l in raw['unitInp'][2:]]
    app_power_units = [l.strip() for l in raw['unitOut'][2:]]
    datetimes = raw['input'][:,0]
    sampling_period = np.mean(np.diff(datetimes))
    agg_power = raw['input'][:, 2:]
    app_powers = raw['output'][:, 2:]

    return{
        'X': app_powers,
        'Y': agg_power,
        'sampling_period': sampling_period,
        'Y_labels': app_power_labels,
        'X_labels': agg_power_labels,
        'Y_units': app_power_units,
        'X_units': agg_power_units,
    }

def device_type(profile, tol=1e-3, max_states=5):

    profile = np.asarray(profile).flatten()

    # Round values slightly to collapse small noise
    rounded = np.round(profile / tol) * tol
    unique_vals = np.unique(rounded)

    # Check one-state (ON/OFF)
    if np.all(np.isin(unique_vals, [0, 1])):
        return "one-state"

    # Few discrete values → multi-state
    if len(unique_vals) <= max_states:
        return "multi-state"

    # Otherwise continuous
    return "continuous"

def ghost_data(p_agg, p_appliances):

    # Sum appliance power at each timestep
    appliance_sum = np.sum(p_appliances, axis=1)

    # Ghost power = aggregated - sum of appliances
    ghost_power = p_agg - appliance_sum

    # Add ghost column to appliance matrix
    p_appliances_with_ghost = np.column_stack((ghost_power, p_appliances))

    # Total energy calculations
    total_energy = np.sum(p_agg)
    ghost_energy = np.sum(np.abs(ghost_power))

    ghost_percent = 100 * ghost_energy / total_energy

    return p_appliances_with_ghost, ghost_percent

# REDD Data
def process_data(data):

    data['Y'] = data['Y'][:, 0]
    data['Y_labels'] = 'P_agg'

    device_types = []
    for i in range(data['X'].shape[1]):
        profile = data['X'][:, i]
        device_types.append(device_type(profile))
    device_types = list(set(device_types))
    data['device_types'] = device_types

    # add 'GHOST' as oth entry to data['Y']['out_labels']
    data['X_labels'].insert(0, 'GHOST')
    data['X'], data['ghost_percent'] = ghost_data(data['Y'], data['X'])

    return data

def trim_data(data, n_samples):

    X = data['X']
    Y = data['Y']

    if n_samples > X.shape[0]: return data

    trimmed_data = {}
    trimmed_data['X'] = X[:n_samples]
    trimmed_data['Y'] = Y[:n_samples]
    trimmed_data['X_labels'] = data['X_labels']
    trimmed_data['Y_labels'] = data['Y_labels']

    return trimmed_data

def split_data(data, method='1-fold', rT=0.7, rV=0.15, kfold=5, fold=1, shuffle=False, random_state=42):
    """
    :param data: Dict, output of load_dataset()
    :param method: '1-fold' or 'k-fold'
    :param rT: Train ratio
    :param rV: Validation ratio
    :param kfold: Number of folds
    :param fold: Which fold to use
    :param shuffle: Whether to shuffle
    :return: data_split
    """

    X = data['X']
    Y = data['Y']
    N = X.shape[0]

    data_split = {}

    # 1-Fold Split
    if method == '1-fold':

        if shuffle:
            idx = np.random.permutation(N)
            X = X[idx]
            Y = Y[idx]

        n_train = int(rT * N)
        n_val = int(rV * N)

        X_train = X[:n_train]
        Y_train = Y[:n_train]

        X_val = X[n_train:n_train+n_val]
        Y_val = Y[n_train:n_train+n_val]

        X_test = X[n_train+n_val:]
        Y_test = Y[n_train+n_val:]

    elif method == 'k-fold':

        if shuffle: kf = KFold(n_splits=kfold, shuffle=shuffle, random_state=random_state)
        else: kf = KFold(n_splits=kfold, shuffle=shuffle)

        fold_idx = 1
        for train_idx, test_idx in kf.split(X):

            if fold_idx == fold:
                X_train_full = X[train_idx]
                Y_train_full = Y[train_idx]
                X_test = X[test_idx]
                Y_test = Y[test_idx]
                break

            fold_idx += 1

        n_train_full = X_train_full.shape[0]
        n_val = int(rV * n_train_full)

        X_val = X_train_full[:n_val]
        Y_val = Y_train_full[:n_val]

        X_train = X_train_full[n_val:]
        Y_train = Y_train_full[n_val:]

    else:
        raise ValueError('Invalid method')

    data_split['Train'] = {'X': X_train, 'Y': Y_train}
    data_split['Val'] = {'X': X_val, 'Y': Y_val}
    data_split['Test'] = {'X': X_test, 'Y': Y_test}
    data_split['sampling_period'] = 'sampling_period'
    data_split['X_labels'] = data['X_labels']
    data_split['Y_labels'] = data['Y_labels']
    data_split['X_units'] = data['X_units']
    data_split['Y_units'] = data['Y_units']

    return data_split