"""
Phase 5 — is the no-gap result specific to Chronos, or do all TSFMs hold up?

Each TSFM lives in its own venv (uni2ts and timesfm pin conflicting deps), so
this script runs ONE model per invocation and writes its own result file.
aggregate_phase5.py then merges whatever is present.

  .venv          -> python experiments/run_phase5.py --model chronos --dataset bdg2
  .venv-tsfm     -> python experiments/run_phase5.py --model timesfm --dataset bdg2
  .venv-moirai   -> python experiments/run_phase5.py --model moirai  --dataset bdg2

Baselines are computed on every run (they are cheap and CPU only) so each
result file is self-contained and readable on its own.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.point_metrics import mase, smape

RESULTS_DIR = ROOT / "results" / "phase5"

CONTEXT_LEN = 168
HORIZON     = 24
N_WINDOWS   = 8
STEP_SIZE   = 168
SEASON      = 24


def load_dataset(name):
    if name == "bdg2":
        df = pd.read_csv(
            ROOT / "data" / "raw" / "electricity_cleaned.csv",
            index_col=0, parse_dates=True,
        )
        with open(ROOT / "results" / "phase1" / "phase1_config.json") as f:
            cols = json.load(f)["buildings"]
        return df, cols

    if name == "solar":
        from src.data.nrel_loader import load_hourly, select_plants
        df = load_hourly(ROOT / "data" / "raw")
        return df, select_plants(df, n=20, seed=42)

    raise ValueError(f"unknown dataset {name}")


def build_model(name):
    if name == "chronos":
        import torch
        from src.models.tsfm_wrappers.chronos_wrapper import ChronosForecaster
        inner = ChronosForecaster(device="cuda" if torch.cuda.is_available() else "cpu")
        return "Chronos", lambda ctx, h: np.asarray(inner.predict(ctx, horizon=h), dtype=np.float64)

    if name == "timesfm":
        from src.models.tsfm_wrappers.timesfm_wrapper import TimesFMForecaster
        m = TimesFMForecaster()
        return "TimesFM", lambda ctx, h: m.predict(ctx, h)

    if name == "moirai":
        from src.models.tsfm_wrappers.moirai_wrapper import MoiraiForecaster
        m = MoiraiForecaster()
        return "Moirai", lambda ctx, h: m.predict(ctx, h)

    raise ValueError(f"unknown model {name}")


def build_cutoffs(index, n_windows, horizon, step):
    last = index[-1] - pd.Timedelta(hours=int(horizon))
    out = []
    for i in range(n_windows):
        c = last - pd.Timedelta(hours=int(step * (n_windows - 1 - i)))
        if index.get_loc(c) >= CONTEXT_LEN:
            out.append(c)
    return out


def daily_naive(ctx, horizon, season_len=24):
    reps = int(np.ceil(horizon / season_len))
    return np.tile(ctx[-season_len:], reps)[:horizon]


def daily_mean7(ctx, horizon, season_len=24, n_days=7):
    usable = min(n_days, len(ctx) // season_len)
    profile = ctx[-usable * season_len:].reshape(usable, season_len).mean(axis=0)
    reps = int(np.ceil(horizon / season_len))
    return np.tile(profile, reps)[:horizon]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["chronos", "timesfm", "moirai"])
    ap.add_argument("--dataset", required=True, choices=["bdg2", "solar"])
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    horizon = args.horizon
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Phase 5 — {args.model} on {args.dataset}, {horizon}h ahead")
    print("=" * 60)

    df, columns = load_dataset(args.dataset)
    n_windows = 2 if args.quick else N_WINDOWS
    if args.quick:
        columns = columns[:3]
        print(">>> quick mode <<<")

    cutoffs = build_cutoffs(df.index, n_windows, horizon, STEP_SIZE)
    print(f"series: {len(columns)}  cutoffs: {len(cutoffs)}")

    model_name, predict = build_model(args.model)
    print(f"model: {model_name}")

    records = []
    t0 = time.time()

    for si, col in enumerate(columns, 1):
        series = df[col].ffill().bfill()
        values = series.to_numpy(dtype=np.float64)

        for cutoff in cutoffs:
            cidx = series.index.get_loc(cutoff)
            ctx = values[cidx - CONTEXT_LEN + 1 : cidx + 1].astype(np.float32)
            tgt = values[cidx + 1 : cidx + 1 + horizon].astype(np.float32)

            if np.isnan(tgt).any() or np.isnan(ctx).any():
                continue

            preds = {
                "DailyNaive": daily_naive(ctx, horizon),
                "DailyMean7": daily_mean7(ctx, horizon),
                model_name: predict(ctx, horizon),
            }

            for method, pred in preds.items():
                records.append({
                    "series": col, "cutoff": cutoff, "method": method,
                    "dataset": args.dataset, "horizon": horizon,
                    "mase": mase(tgt, pred, values, season=SEASON),
                    "smape": smape(tgt, pred),
                })

        print(f"  [{si}/{len(columns)}] {col}")

    elapsed = time.time() - t0
    results = pd.DataFrame(records)

    tag = f"{args.dataset}_{args.model}_h{horizon}"
    results.to_parquet(RESULTS_DIR / f"phase5_{tag}.parquet")

    summary = (
        results.groupby("method")
        .agg(MASE_median=("mase", "median"), MASE_mean=("mase", "mean"),
             sMAPE_mean=("smape", "mean"), n_evals=("mase", "count"))
        .round(3)
        .sort_values("MASE_median")
    )

    print()
    print(summary.to_string())

    best_naive = min(
        summary.loc["DailyNaive", "MASE_median"],
        summary.loc["DailyMean7", "MASE_median"],
    )
    tsfm_mase = summary.loc[model_name, "MASE_median"]
    gap = (tsfm_mase - best_naive) / best_naive * 100
    print(f"\n{model_name} vs best naive: {tsfm_mase:.3f} vs {best_naive:.3f} ({gap:+.1f}%)")
    print(f"wall time: {elapsed:.0f}s")
    print(f"saved: phase5_{tag}.parquet")


if __name__ == "__main__":
    main()
