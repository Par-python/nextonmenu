"""Deterministic synthetic trend fixtures (no network)."""
import numpy as np
import pandas as pd
import pytest


def _weekly_index(n_weeks):
    return pd.date_range("2010-01-03", periods=n_weeks, freq="W")


@pytest.fixture
def rising_then_peak_iot():
    """A curve: long flat era, sharp rise, peak, then plateau. ~520 weeks (10y)."""
    n = 520
    idx = _weekly_index(n)
    flat = np.full(150, 5.0)                      # pre-niche noise floor
    rise = np.linspace(5, 100, 260)               # the rising edge to peak
    plateau = np.full(110, 90.0)                  # post-peak plateau
    values = np.concatenate([flat, rise, plateau])
    rng = np.random.default_rng(0)
    values = np.clip(values + rng.normal(0, 1.5, n), 0, 100)
    return pd.DataFrame({"value": values}, index=idx)


@pytest.fixture
def flat_iot():
    """A curve that never rises — stays near the noise floor."""
    n = 520
    idx = _weekly_index(n)
    rng = np.random.default_rng(1)
    values = np.clip(np.full(n, 4.0) + rng.normal(0, 1.0, n), 0, 100)
    return pd.DataFrame({"value": values}, index=idx)


@pytest.fixture
def region_concentrated():
    """Interest concentrated in 2 of 10 regions => low geographic entropy."""
    return pd.DataFrame({"interest": [100, 80, 0, 0, 0, 0, 0, 0, 0, 0]})


@pytest.fixture
def region_uniform():
    """Interest spread evenly => high geographic entropy."""
    return pd.DataFrame({"interest": [50] * 10})
