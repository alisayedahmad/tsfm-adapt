# zero-shot chronos sur un batiment bdg2, comparaison a une baseline naive

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from chronos import ChronosPipeline

CSV_PATH = Path("data/raw/electricity_cleaned.csv")
CONTEXT_LENGTH = 24 * 14  # deux semaines de contexte
PREDICTION_LENGTH = 24


def pick_building(df: pd.DataFrame) -> str:
    completeness = df.drop(columns="timestamp").notna().mean()
    candidates = completeness[completeness > 0.99].index
    if len(candidates) == 0:
        raise ValueError("aucun batiment avec assez peu de NaN pour ce test")
    return candidates[0]


def load_series():
    df = pd.read_csv(CSV_PATH)
    building = pick_building(df)
    series = df[building].dropna()

    if len(series) < CONTEXT_LENGTH + PREDICTION_LENGTH:
        raise ValueError(f"{building} n'a pas assez de points pour ce test")

    context = series.iloc[-(CONTEXT_LENGTH + PREDICTION_LENGTH):-PREDICTION_LENGTH]
    target = series.iloc[-PREDICTION_LENGTH:]

    print("batiment utilise:", building)
    print("longueur contexte:", len(context))
    print("longueur cible:", len(target))

    return context.values, target.values


def naive_baseline(context: np.ndarray) -> np.ndarray:
    last_day = context[-24:]
    return last_day


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def run():
    context, target = load_series()

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
    )

    context_tensor = torch.tensor(context, dtype=torch.float32)
    forecast = pipeline.predict(
        context_tensor,
        prediction_length=PREDICTION_LENGTH,
        num_samples=20,
    )

    median_forecast = np.median(forecast[0].numpy(), axis=0)

    naive = naive_baseline(context)

    print("mae chronos zero-shot:", round(mae(median_forecast, target), 3))
    print("mae baseline naive (jour precedent):", round(mae(naive, target), 3))


if __name__ == "__main__":
    run()