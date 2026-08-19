"""
Plot the diagnosis dumps: every model against ground truth on the same windows.
Run in any venv that has matplotlib.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent.parent
DIAG_DIR = ROOT / "results" / "diagnosis"


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "bdg2"
    files = sorted(DIAG_DIR.glob(f"diag_{dataset}_*.parquet"))
    if not files:
        print(f"no diag files for {dataset} in {DIAG_DIR}")
        sys.exit(1)

    frames = [pd.read_parquet(f).reset_index(drop=True) for f in files]
    df = pd.concat(frames, ignore_index=True)
    models = sorted(df["model"].unique())
    print(f"models: {models}")

    keys = df[["series", "cutoff"]].drop_duplicates().values.tolist()[:6]

    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 2.6 * len(keys)), squeeze=False)
    colors = {"Chronos": "tab:blue", "TimesFM": "tab:orange", "Moirai": "tab:green"}

    for ax, (series, cutoff) in zip(axes[:, 0], keys):
        sub = df[(df["series"] == series) & (df["cutoff"] == cutoff)]

        truth = sub[sub["model"] == models[0]].sort_values("h")
        ax.plot(truth["h"], truth["actual"], "k-", lw=2.2, label="actual", zorder=5)

        for m in models:
            d = sub[sub["model"] == m].sort_values("h")
            if d.empty:
                continue
            c = d["corr"].iloc[0]
            lag = d["best_lag"].iloc[0]
            ax.plot(d["h"], d["pred"], "--", lw=1.4, color=colors.get(m),
                    label=f"{m} (r={c:+.2f}, lag={lag:+d})")

        ax.set_title(f"{series}  {cutoff}", fontsize=9)
        ax.set_xlabel("hours ahead")
        ax.legend(fontsize=7, ncol=4)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = DIAG_DIR / f"diagnosis_{dataset}.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    print()
    stats = (
        df.groupby(["model", "series", "cutoff"])
        .agg(corr=("corr", "first"), best_lag=("best_lag", "first"),
             best_corr=("best_corr", "first"))
        .reset_index()
        .groupby("model")
        .agg(mean_corr=("corr", "mean"), mean_best_corr=("best_corr", "mean"),
             median_lag=("best_lag", "median"))
        .round(3)
    )
    print(stats.to_string())
    print()
    print("reading it: mean_corr near 1 means the shape is right. If mean_best_corr")
    print("is much higher than mean_corr, the prediction is time shifted and")
    print("median_lag says by how many hours. Low on both means the model is lost.")


if __name__ == "__main__":
    main()
