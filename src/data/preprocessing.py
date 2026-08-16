import numpy as np
import pandas as pd

N_TIME_FEATURES = 6  # hour_sin/cos, dow_sin/cos, month_sin/cos


def make_time_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    h = np.asarray(timestamps.hour, dtype=np.float32)
    d = np.asarray(timestamps.dayofweek, dtype=np.float32)
    m = np.asarray(timestamps.month, dtype=np.float32)
    return np.stack([
        np.sin(2 * np.pi * h / 24),
        np.cos(2 * np.pi * h / 24),
        np.sin(2 * np.pi * d / 7),
        np.cos(2 * np.pi * d / 7),
        np.sin(2 * np.pi * (m - 1) / 12),
        np.cos(2 * np.pi * (m - 1) / 12),
    ], axis=1).astype(np.float32)  # (T, 6)


def _ffill(arr: np.ndarray) -> np.ndarray:
    s = pd.Series(arr)
    return s.ffill().bfill().to_numpy(dtype=np.float32)


def extract_windows(
    values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    context_len: int,
    horizon: int,
    step: int,
) -> list[dict]:
    windows = []
    total = context_len + horizon

    for start in range(0, len(values) - total + 1, step):
        ctx = values[start : start + context_len]
        tgt = values[start + context_len : start + total]

        if np.isnan(ctx).mean() > 0.1 or np.isnan(tgt).any():
            continue

        windows.append({
            "x_values": _ffill(ctx),
            "x_features": make_time_features(timestamps[start : start + context_len]),
            "y_values": tgt.astype(np.float32),
        })

    return windows
