import numpy as np


def mase(y_true, y_pred, y_train, season=24):
    # scale-free error relative to seasonal naive on training set
    naive_err = np.abs(y_train[season:] - y_train[:-season])
    scale = np.mean(naive_err)
    if scale < 1e-9:
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / scale


def smape(y_true, y_pred):
    # symmetric MAPE, 0-200 range
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 1e-9
    if not mask.any():
        return np.nan
    return 100.0 * np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask])


def naive_seasonal_forecast(y_context, horizon, season=24):
    # repeat the last full season cycle — this IS the MASE=1 baseline
    if len(y_context) < season:
        return np.full(horizon, np.nanmean(y_context))
    tail = y_context[-season:]
    reps = (horizon // season) + 1
    return np.tile(tail, reps)[:horizon]
