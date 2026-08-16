from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.nrel_loader import load_hourly, select_plants

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


@pytest.fixture(scope="module")
def solar_df():
    return load_hourly(DATA_DIR)


def test_shape_and_range(solar_df):
    assert solar_df.shape == (8760, 137)
    assert solar_df.index[0] == pd.Timestamp("2006-01-01 00:00:00")
    assert solar_df.index[-1] == pd.Timestamp("2006-12-31 23:00:00")


def test_hourly_frequency(solar_df):
    deltas = solar_df.index.to_series().diff().dropna().unique()
    assert len(deltas) == 1
    assert deltas[0] == pd.Timedelta(hours=1)


def test_no_negative_generation(solar_df):
    assert (solar_df.to_numpy() >= 0).all()


def test_night_hours_are_zero(solar_df):
    # 2am should be zero everywhere, solar plants do not generate at night
    night = solar_df[solar_df.index.hour == 2]
    assert night.to_numpy().max() == 0


def test_select_plants_deterministic(solar_df):
    a = select_plants(solar_df, n=20, seed=42)
    b = select_plants(solar_df, n=20, seed=42)
    assert a == b
    assert len(a) == 20


def test_selected_plants_are_active(solar_df):
    for p in select_plants(solar_df, n=20, seed=42):
        assert solar_df[p].max() > 0.1
