"""Impact sample preparation."""

from __future__ import annotations

import pandas as pd


def prepare_impact_observations(windows: pd.DataFrame) -> pd.DataFrame:
    """Return valid fixed-window observations for impact estimation."""

    required = {"signed_imbalance", "abs_signed_imbalance", "gross_volume", "participation_ratio", "signed_impact"}
    legacy_required = {"q_t", "signed_impact"}
    if required.issubset(windows.columns):
        observations = windows.copy()
    elif legacy_required.issubset(windows.columns):
        observations = windows.copy()
        observations["signed_imbalance"] = observations["q_t"]
        observations["abs_signed_imbalance"] = observations["q_t"].abs()
        if "absolute_volume" in observations.columns:
            observations["gross_volume"] = observations["absolute_volume"]
            observations["participation_ratio"] = observations["abs_signed_imbalance"] / observations["gross_volume"]
    else:
        missing = required.difference(windows.columns)
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    observations["abs_imbalance"] = observations["abs_signed_imbalance"]
    observations = observations.loc[
        (observations["abs_signed_imbalance"] > 0)
        & (observations["gross_volume"] > 0)
        & observations["participation_ratio"].notna()
    ].copy()
    return observations
