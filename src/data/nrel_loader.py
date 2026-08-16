import gzip
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SOLAR_URL = (
    "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data"
    "/master/solar-energy/solar_AL.txt.gz"
)

# 137 PV plants in Alabama, 10-min resolution, year 2006
RAW_NAME = "solar_AL.txt.gz"
START = "2006-01-01 00:00:00"


def download(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / RAW_NAME
    if path.exists():
        return path
    print(f"downloading {SOLAR_URL}")
    urllib.request.urlretrieve(SOLAR_URL, path)
    return path


def load_hourly(dest_dir: Path) -> pd.DataFrame:
    """
    Returns wide DataFrame, hourly, DatetimeIndex, 137 plant columns.
    Original data is 10-min; hourly is the mean over each hour.
    """
    path = download(dest_dir)

    with gzip.open(path, "rt") as f:
        arr = np.array([line.strip().split(",") for line in f], dtype=np.float32)

    idx = pd.date_range(START, periods=len(arr), freq="10min")
    df = pd.DataFrame(arr, index=idx, columns=[f"plant_{i:03d}" for i in range(arr.shape[1])])
    return df.resample("h").mean()


def select_plants(df: pd.DataFrame, n: int = 20, seed: int = 42) -> list[str]:
    # keep plants with real generation, drop near-dead sensors
    daily_max = df.resample("D").max()
    active = (daily_max > 0.1).mean()
    usable = active[active > 0.9].index.tolist()

    rng = np.random.default_rng(seed)
    return sorted(rng.choice(usable, size=min(n, len(usable)), replace=False).tolist())


if __name__ == "__main__":
    out = Path(__file__).parent.parent.parent / "data" / "raw"
    df = load_hourly(out)
    print(df.shape, df.index[0], "->", df.index[-1])
    print("plants selected:", select_plants(df)[:5], "...")
