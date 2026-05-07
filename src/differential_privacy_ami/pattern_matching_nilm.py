# Casey Dettlaff
# Reference: BaseNILM toolkit for energy disaggregation, Dr. Pascal A. Schirmer

from numpy import savez_compressed
import numpy as np
from tqdm import tqdm
from scipy import stats

def features(data, feature_selection):

    if 'all' in feature_selection: F = 14
    else: F = len(feature_selection)
    N,T = data.shape
    out = np.zeros((N, F))
    idx = 0

    if 'Mean' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.mean(data, axis=1)
        idx = idx + 1
    if 'Std' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.std(data, axis=1)
        idx = idx + 1
    if 'RMS' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.sqrt(np.mean(data ** 2, axis=1))
        idx = idx + 1
    if 'Peak2Rms' in feature_selection or 'all' in feature_selection:
        temp = np.max(data, axis=1)
        temp2 = np.sqrt(np.mean(data ** 2, axis=1))
        out[:, idx] = np.divide(temp, temp2)
        idx = idx + 1
    if 'Median' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.median(data, axis=1)
        idx = idx + 1
    if 'Min' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.min(data, axis=1)
        idx = idx + 1
    if 'Max' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.max(data, axis=1)
        idx = idx + 1
    if 'Per25' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.percentile(data, 25, axis=1)
        idx = idx + 1
    if 'Per75' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.percentile(data, 75, axis=1)
        idx = idx + 1
    if 'Energy' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.mean(data, axis=1)
        idx = idx + 1
    if 'Var' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.var(data, axis=1)
        idx = idx + 1
    if 'Range' in feature_selection or 'all' in feature_selection:
        out[:, idx] = np.ptp(data, axis=1)
        idx = idx + 1
    if '3rdMoment' in feature_selection or 'all' in feature_selection:
        out[:, idx] = stats.skew(data, axis=1)
        idx = idx + 1
    if '4thMoment' in feature_selection or 'all' in feature_selection:
        out[:, idx] = stats.kurtosis(data, axis=1)
        idx = idx + 1

    # Post-processing
    out = np.nan_to_num(out)
    out[out == np.inf] = 0

    return out

def trainMdlPM(data_train, mdl_filepath, save_mdl=False, return_mdl=False):

    N = data_train['Y'].shape[0]
    window_length = data_train['Y'].shape[1]
    numApp = data_train['X'].shape[2]

    mdl = np.zeros((N, window_length, numApp+1))

    mdl[:, :, 0] = data_train['Y']
    mdl[:, :, 1:] = data_train['X']

    if save_mdl: savez_compressed(mdl_filepath, mdl)
    if return_mdl: return mdl

def window_data(X, Y, window_length, stride=1):

    # N = number of windows
    # Before windowing, Y shape = (T,).
    # Before windowing, X shape = (T, numApp).
    # After windowing, Y shape = (N, window_length).
    # After windowing, X shape = (N, window_length, numApp).

    T = Y.shape[0]
    numApp = X.shape[1] # Number of Appliances

    N = (T - window_length) // stride + 1 # Number of Windows

    Y_win = np.zeros((N, window_length), dtype=np.float32)
    X_win = np.zeros((N, window_length, numApp), dtype=np.float32)

    idx = 0
    for start in range(0, T - window_length + 1, stride):
        end = start + window_length
        Y_win[idx] = Y[start:end]
        X_win[idx] = X[start:end]
        idx += 1

    return X_win, Y_win

def unwindow_data(X_win, window_length, stride):
    """
    Inverse of window_data for Y-type input
    (N, window_length, numApp) → (T_original, numApp)
    """

    N, T, numApp = X_win.shape

    if T != window_length:
        raise ValueError("window_length mismatch.")

    # Recover original length T
    T_original = (N - 1) * stride + window_length

    X_recon = np.zeros((T_original, numApp))
    counts = np.zeros(T_original)

    for i in range(N):
        start = i * stride
        end = start + window_length

        X_recon[start:end] += X_win[i]
        counts[start:end] += 1

    counts[counts == 0] = 1
    X_recon /= counts[:, None]

    return X_recon

def dtw_distance(x,y):

    x = np.asarray(x)
    y = np.asarray(y)

    T_x = len(x)
    T_y = len(y)

    dist_func = lambda a, b: abs(a - b)

    # Cost Matrix
    D = np.zeros((T_x + 1, T_y + 1))
    D[0, :] = np.inf
    D[:, 0] = np.inf
    D[0, 0] = 0

    # Fill Matrix
    for i in range(1, T_x + 1):
        for j in range(1, T_y + 1):
            cost = dist_func(x[i - 1], y[j - 1])

            D[i, j] = cost + min(
                D[i - 1, j],  # insertion
                D[i, j - 1],  # deletion
                D[i - 1, j - 1]  # match
            )

    distance = D[T_x, T_y]

    # Backtrack to recover path
    i, j = T_x, T_y
    path = []

    while i > 0 and j > 0:
        path.append((i - 1, j - 1))

        steps = [
            D[i - 1, j],
            D[i, j - 1],
            D[i - 1, j - 1]
        ]

        argmin = np.argmin(steps)

        if argmin == 0:
            i -= 1
        elif argmin == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

    path.reverse()

    return distance, path

# Build Matrix D(i,j) which stores cumulative cost to align x[0:i] and y[0:j]
# Warping path is perfect diagonal - no warping.
# DTW is not giving any benefit over simple L1 distance.

def correlation(x, y):

    eps = 1e-12  # Avoid divide-by-zero errors.

    x = np.asarray(x)
    y = np.asarray(y)

    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    x_norm = np.linalg.norm(x_centered) + eps
    y_norm = np.linalg.norm(y_centered) + eps
    corr = np.dot(x_centered, y_centered) / (x_norm * y_norm)

    return corr

def testMdlPM(X_test, Y_test, mdl, method='correlation_maximization', feature_selection=None, C=0.01):

    # Read template database shape
    N_mdl, T_mdl, numApp_mdl = mdl.shape  # N,T = num_samples, num_timesteps
    numApp_mdl = numApp_mdl - 1

    C = int(np.floor(C * N_mdl))  # number of candidate model patterns to compare using DTW

    # Test Data Shape
    N_Xtest, T_Xtest, numApp_Xtest = X_test.shape
    N_Ytest, T_Ytest = Y_test.shape

    X_pred = np.zeros((N_Ytest, T_Ytest, numApp_Xtest))

    # Check mdl and test data are formatted the same
    if not (T_mdl == T_Xtest == T_Ytest): raise ValueError(f'Timestep mismatch: mdl={T_mdl}, X_test={T_Xtest}, Y_test={T_Ytest}')
    if numApp_mdl != numApp_Xtest : raise ValueError(f'numApp mismatch: mdl={numApp_mdl}, Y_test={numApp_Xtest}')

    # Feature Extraction
    if feature_selection:
        features_mdl = features(mdl[:, :, 0], feature_selection)
        features_Y = features(Y_test, feature_selection)

    sel_list = [] # For debugging

    # Find Top Candidates
    for i in tqdm(range(N_Xtest)):

        if feature_selection:
            feature_diff = abs(features_Y[i, :] - features_mdl)
            feature_diff = np.sum(feature_diff, axis=1)  # Reduce over feature dimension
            C_eff = min(C, N_mdl)
            idx = np.argpartition(feature_diff, C_eff)[:C_eff]  # Indices of C smallest feature distances
        else: # Disable feature filtering for debugging
            C_eff = N_mdl
            idx = np.arange(N_mdl)

        tempMdl = mdl[idx, :, :]  # Top C candidates

        # For top candidates, Compute Correlation / Distance
        dist = np.zeros(C_eff)

        y = Y_test[i, :]
        y_norm = y / np.sum(y)

        for ii in range(C_eff):

            template = tempMdl[ii, :, 0]
            template_norm = template / np.sum(template)

            if method.split('_')[0] == 'correlation': dist_ii = correlation(y_norm, template_norm)
            elif method.split('_')[0] == 'dtw': dist_ii, path = dtw_distance(y_norm, template_norm)
            else: raise ValueError(f'Unknown Method {method}.')

            dist[ii] = dist_ii

        # Remove all 'None' entries
        dist = dist[~np.isnan(dist)]
        if method.split('_')[1] == 'maximization': sel = np.argmax(np.abs(dist))
        elif method.split('_')[1] == 'minimization': sel = np.argmin(np.abs(dist))
        else: raise ValueError(f'Unknown Method {method}.')

        sel_list.append(sel) # For debugging

        best_template_agg = tempMdl[sel, :, 0]
        best_template_apps = tempMdl[sel, :, 1:]
        # scale = np.sum(x) / np.sum(best_template_agg) # If best_template is all 0, this will cause error
        # scale = np.sum(x) / np.sum(np.sum(best_template_apps, axis=1))
        X_pred[i, :, :] = best_template_apps

    return X_pred

def evaluate_prediction(Y_pred, Y_test):

    if Y_pred.shape != Y_test.shape:
        raise ValueError(f"Shape mismatch: Y_pred {Y_pred.shape}, Y_test {Y_test.shape}")

    if Y_test.ndim != 2:
        raise ValueError("Expected unwindowed data with shape (T, numApp).")

    # Overall metrics
    mae = np.mean(np.abs(Y_pred - Y_test))
    rmse = np.sqrt(np.mean((Y_pred - Y_test) ** 2))

    denom = np.sum(Y_test ** 2)
    nde = np.sum((Y_pred - Y_test) ** 2) / denom if denom != 0 else 0.0

    # Per-appliance metrics
    mae_per_app = np.mean(np.abs(Y_pred - Y_test), axis=0)
    rmse_per_app = np.sqrt(np.mean((Y_pred - Y_test) ** 2, axis=0))

    metrics = {
        "MAE_overall": mae,
        "RMSE_overall": rmse,
        "NDE_overall": nde,
        "MAE_per_appliance": mae_per_app,
        "RMSE_per_appliance": rmse_per_app,
    }

    return metrics

def energy_accuracy(Y_pred, Y_true):
    Y_pred = np.asarray(Y_pred)
    Y_true = np.asarray(Y_true)

    numerator = np.sum(np.abs(Y_true - Y_pred))
    denominator = 2 * np.sum(Y_true)

    acc = 1 - numerator / denominator
    return acc

def get_results(Y_pred, Y_true, Y_labels):

    results = {}

    # Overall metrics
    mae = np.mean(np.abs(Y_pred - Y_true))
    rmse = np.sqrt(np.mean((Y_pred - Y_true) ** 2))

    denom = np.sum(Y_true ** 2)
    nde = np.sum((Y_pred - Y_true) ** 2) / denom if denom != 0 else 0.0

    # Per-appliance metrics
    mae_per_app = np.mean(np.abs(Y_pred - Y_true), axis=0)
    rmse_per_app = np.sqrt(np.mean((Y_pred - Y_true) ** 2, axis=0))

    results['agg_MAE'] = mae
    results['agg_RMSE'] = rmse
    results['agg_NDE'] = nde
    results['agg_EACC'] = energy_accuracy(Y_pred, Y_true)

    K = len(Y_labels)
    for k in range(K):
        results[f'{Y_labels[k]}_MAE'] = mae_per_app[k]
        results[f'{Y_labels[k]}_RMSE'] = rmse_per_app[k]

    return results