import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, SeasonalNaive


def run_statistical_baselines(df_long, horizon=24, step_size=168, n_windows=8):
    # rolling eval via statsforecast cross_validation
    # n_jobs=1 — safe on Windows, avoids multiprocessing issues
    models = [
        SeasonalNaive(season_length=24),
        AutoETS(season_length=24),
        AutoTheta(season_length=24),
    ]

    sf = StatsForecast(models=models, freq="h", n_jobs=1)

    print("[baselines] running cross_validation "
          f"(h={horizon}, step={step_size}, windows={n_windows})")
    print("[baselines] this takes 10-30 min with n_jobs=1, "
          "bump to -1 on linux if you want speed")

    cv = sf.cross_validation(
        df=df_long,
        h=horizon,
        step_size=step_size,
        n_windows=n_windows,
    )

    # statsforecast names columns after model classes
    print(f"[baselines] done — {len(cv)} forecast rows, "
          f"cutoffs: {cv['cutoff'].nunique()}")
    return cv
