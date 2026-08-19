"""
Diagnostic — dump raw predictions for a few windows so the three models can be
compared side by side against ground truth.

Phase 5 says TimesFM and Moirai lose badly to a naive baseline while Chronos
holds. That pattern is equally consistent with a real domain gap and with a
broken wrapper, so this dumps the actual numbers before anyone writes a
conclusion.

Run once per venv, same as run_phase5:
    python experiments/diagnose_predictions.py --model chronos --dataset bdg2
    python experiments/diagnose_predictions.py --model timesfm --dataset bdg2
    python experiments/diagnose_predictions.py --model moirai  --dataset bdg2

Then: python experiments/plot_diagnosis.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "diagnosis"

CONTEXT_LEN = 168
HORIZON = 24
N_SERIES = 4      # a handful is enough to see a systematic problem
N_CUTOFFS = 2


def load_dataset(name):
    if name == "bdg2":
        df = pd.read_csv(
            ROOT / "data" / "raw" / "electricity_cleaned.csv",
            index_col=0, parse_dates=True,
        )
        with open(ROOT / "results" / "phase1" / "phase1_config.json") as f:
            return df, json.load(f)["buildings"]
    if name == "solar":
        from src.data.nrel_loader import load_hourly, select_plants
        df = load_hourly(ROOT / "data" / "raw")
        return df, select_plants(df, n=20, seed=42)
    raise ValueError(name)


def build_model(name):
    if name == "chronos":
        import torch
        from src.models.tsfm_wrappers.chronos_wrapper import ChronosForecaster
        inner = ChronosForecaster(device="cuda" if torch.cuda.is_available() else "cpu")
        return "Chronos", lambda c, h: np.asarray(inner.predict(c, horizon=h), dtype=np.float64)
    if name == "timesfm":
        from src.models.tsfm_wrappers.timesfm_wrapper import TimesFMForecaster
        m = TimesFMForecaster()
        return "TimesFM", lambda c, h: m.predict(c, h)
    if name == "moirai":
        from src.models.tsfm_wrappers.moirai_wrapper import MoiraiForecaster
        m = MoiraiForecaster()
        return "Moirai", lambda c, h: m.predict(c, h)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["chronos", "timesfm", "moirai"])
    ap.add_argument("--dataset", default="bdg2", choices=["bdg2", "solar"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, columns = load_dataset(args.dataset)
    columns = columns[:N_SERIES]

    last = df.index[-1] - pd.Timedelta(hours=int(HORIZON))
    cutoffs = [last - pd.Timedelta(hours=int(168 * i)) for i in range(N_CUTOFFS)][::-1]

    model_name, predict = build_model(args.model)
    print(f"{model_name} on {args.dataset}")

    rows = []
    for col in columns:
        series = df[col].ffill().bfill()
        values = series.to_numpy(dtype=np.float64)

        for cutoff in cutoffs:
            cidx = series.index.get_loc(cutoff)
            ctx = values[cidx - CONTEXT_LEN + 1 : cidx + 1].astype(np.float32)
            tgt = values[cidx + 1 : cidx + 1 + HORIZON].astype(np.float32)
            if np.isnan(tgt).any() or np.isnan(ctx).any():
                continue

            pred = predict(ctx, HORIZON)

            # correlation between prediction and truth catches phase shifts:
            # a model with the right shape but wrong offset still correlates high
            corr = float(np.corrcoef(pred, tgt)[0, 1]) if np.std(pred) > 1e-9 else np.nan
            # best lag tells us if the prediction is simply shifted in time
            lags = range(-6, 7)
            best_lag, best_corr = 0, -2.0
            for lag in lags:
                if lag < 0:
                    a, b = pred[-lag:], tgt[:lag]
                elif lag > 0:
                    a, b = pred[:-lag], tgt[lag:]
                else:
                    a, b = pred, tgt
                if len(a) > 3 and np.std(a) > 1e-9 and np.std(b) > 1e-9:
                    c = float(np.corrcoef(a, b)[0, 1])
                    if c > best_corr:
                        best_corr, best_lag = c, lag

            for h in range(HORIZON):
                rows.append({
                    "model": model_name, "dataset": args.dataset,
                    "series": col, "cutoff": str(cutoff), "h": h,
                    "pred": float(pred[h]), "actual": float(tgt[h]),
                    "ctx_mean": float(ctx.mean()), "ctx_std": float(ctx.std()),
                    "corr": corr, "best_lag": best_lag, "best_corr": best_corr,
                })

            print(f"  {col[:28]:28s} corr={corr:+.3f}  best_lag={best_lag:+d} "
                  f"({best_corr:+.3f})  pred[{pred.min():.1f},{pred.max():.1f}] "
                  f"true[{tgt.min():.1f},{tgt.max():.1f}]")

    out = pd.DataFrame(rows)
    path = OUT_DIR / f"diag_{args.dataset}_{args.model}.parquet"
    out.to_parquet(path)

    print()
    print(f"mean corr: {out.groupby(['series', 'cutoff'])['corr'].first().mean():+.3f}")
    print(f"lag distribution: {out.groupby(['series', 'cutoff'])['best_lag'].first().value_counts().to_dict()}")
    print(f"saved {path.name}")


if __name__ == "__main__":
    main()
