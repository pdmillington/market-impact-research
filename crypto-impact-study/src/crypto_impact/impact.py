"""Impact sample preparation."""

from __future__ import annotations

import pandas as pd


def prepare_impact_observations(windows: pd.DataFrame) -> pd.DataFrame:
    """Return positive-quantity observations for impact estimation."""

    required = {"q_t", "signed_impact"}
    missing = required.difference(windows.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    observations = windows.copy()
    observations["abs_imbalance"] = observations["q_t"].abs()
    observations = observations.loc[observations["abs_imbalance"] > 0].copy()
    return observations
