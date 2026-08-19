"""
Merge every phase5_*.parquet into one comparison table.
Runs in any venv, only needs pandas.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "phase5"


def main():
    files = sorted(RESULTS_DIR.glob("phase5_*.parquet"))
    if not files:
        print(f"nothing in {RESULTS_DIR}")
        sys.exit(1)

    print(f"merging {len(files)} result files")
    frames = []
    for f in files:
        d = pd.read_parquet(f)
        d = d.reset_index(drop=True)
        d["cutoff"] = pd.to_datetime(d["cutoff"]).astype(str)
        frames.append(d)
    df = pd.concat(frames, axis=0, ignore_index=True)
    # baselines get recomputed on every run, keep one copy per (dataset, horizon, series, cutoff)
    df = df.drop_duplicates(subset=["dataset", "horizon", "series", "cutoff", "method"])

    for (dataset, horizon), grp in df.groupby(["dataset", "horizon"]):
        summary = (
            grp.groupby("method")
            .agg(MASE_median=("mase", "median"), MASE_mean=("mase", "mean"),
                 sMAPE_mean=("smape", "mean"), n_evals=("mase", "count"))
            .round(3)
            .sort_values("MASE_median")
        )

        print()
        print("=" * 60)
        print(f"{dataset}, {horizon}h ahead")
        print("=" * 60)
        print(summary.to_string())

        naives = [m for m in summary.index if "Naive" in m or "Mean" in m]
        tsfms = [m for m in summary.index if m not in naives]
        if naives and tsfms:
            best_naive = summary.loc[naives, "MASE_median"].min()
            print()
            for m in tsfms:
                gap = (summary.loc[m, "MASE_median"] - best_naive) / best_naive * 100
                verdict = "gap" if gap > 15 else ("wins" if gap < -15 else "tied")
                print(f"  {m:10s} {gap:+6.1f}% vs best naive   {verdict}")

    out = RESULTS_DIR / "phase5_combined.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
