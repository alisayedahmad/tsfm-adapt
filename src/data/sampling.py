import numpy as np
import pandas as pd


def select_buildings(df, n=20, min_completeness=0.80, min_std=0.1, seed=42):
    # filter on data quality, then random sample
    completeness = df.notna().mean()
    stds = df.std()

    candidates = [
        c for c in df.columns
        if completeness[c] >= min_completeness and stds[c] > min_std
    ]

    if len(candidates) < n:
        print(f"[sampling] only {len(candidates)} buildings pass filters (wanted {n})")
        return candidates

    rng = np.random.default_rng(seed)
    selected = list(rng.choice(candidates, size=n, replace=False))
    print(f"[sampling] {len(selected)}/{len(df.columns)} buildings selected "
          f"(min_completeness={min_completeness}, min_std={min_std})")
    return selected


def wide_to_long(df, building_ids):
    # convert wide BDG2 format to statsforecast long format (unique_id, ds, y)
    frames = []
    for bid in building_ids:
        s = df[bid].copy()
        # forward-fill gaps to keep regular freq — statsforecast needs this
        # (bfill mops up any leading NaNs ffill can't reach — AutoETS chokes
        # on any remaining NaN in seasonal_decompose)
        s = s.ffill().bfill()
        frame = pd.DataFrame({
            "unique_id": bid,
            "ds": s.index,
            "y": s.values,
        })
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
