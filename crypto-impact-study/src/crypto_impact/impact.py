"""Impact sample preparation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def add_trailing_impact_normalisation(
    observations: pd.DataFrame,
    *,
    window: str = "1D",
    min_periods: int = 100,
    timestamp_col: str | None = "end_time",
    imbalance_col: str = "signed_imbalance",
    volume_col: str = "gross_volume",
    return_col: str = "current_return_bps",
    duration_col: str = "duration_seconds",
    require_full_window: bool = False,
) -> pd.DataFrame:
    """Add backward-looking volume and volatility normalisations.

    ``normalised_imbalance`` is absolute signed imbalance divided by trailing
    market volume. ``normalised_signed_impact`` is the contemporaneous return,
    signed by the imbalance direction, divided by trailing realised volatility.
    The current bar is excluded from both trailing benchmarks.

    With ``require_full_window`` the benchmarks are defined only once a full
    ``window`` of history has elapsed since the first observation (a time-based
    warm-up). Use it with ``min_periods=1``: a bar-count minimum instead drops
    bars from quiet periods, when fewer bars complete within the window, and so
    selects the sample on trading activity.
    """

    required = {imbalance_col, volume_col, return_col, duration_col}
    if timestamp_col is not None:
        required.add(timestamp_col)
    missing = required.difference(observations.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if min_periods < 1:
        raise ValueError("min_periods must be at least 1.")
    window_seconds = pd.Timedelta(window).total_seconds()
    if window_seconds <= 0:
        raise ValueError("window must have positive duration.")

    data = observations.copy()
    if timestamp_col is None:
        timestamps = pd.to_datetime(data.index, utc=True)
    else:
        timestamps = pd.to_datetime(data[timestamp_col], utc=True)

    order = np.argsort(np.asarray(timestamps))
    data = data.iloc[order].copy()
    time_index = pd.DatetimeIndex(np.asarray(timestamps)[order])

    volume = pd.Series(
        pd.to_numeric(data[volume_col], errors="coerce").to_numpy(),
        index=time_index,
    )
    returns = pd.Series(
        pd.to_numeric(data[return_col], errors="coerce").to_numpy(),
        index=time_index,
    )

    trailing_volume = volume.rolling(
        window, closed="left", min_periods=min_periods
    ).sum()
    trailing_realised_volatility = np.sqrt(
        returns.pow(2).rolling(
            window, closed="left", min_periods=min_periods
        ).sum()
    )

    if require_full_window:
        warm_up = time_index < time_index.min() + pd.Timedelta(window)
        trailing_volume[warm_up] = np.nan
        trailing_realised_volatility[warm_up] = np.nan

    signed_impact = (
        np.sign(pd.to_numeric(data[imbalance_col], errors="coerce"))
        * pd.to_numeric(data[return_col], errors="coerce")
    )
    data["trailing_market_volume"] = trailing_volume.to_numpy()
    data["trailing_realised_volatility_bps"] = (
        trailing_realised_volatility.to_numpy()
    )
    data["signed_current_impact_bps"] = signed_impact
    data["normalised_imbalance"] = (
        pd.to_numeric(data[imbalance_col], errors="coerce").abs()
        / data["trailing_market_volume"]
    )
    data["normalised_signed_impact"] = (
        data["signed_current_impact_bps"]
        / data["trailing_realised_volatility_bps"]
    )
    data["flow_rate_btc_per_second"] = (
        pd.to_numeric(data[volume_col], errors="coerce")
        / pd.to_numeric(data[duration_col], errors="coerce")
    )
    data["trailing_volume_rate_btc_per_second"] = (
        data["trailing_market_volume"] / window_seconds
    )
    data["flow_rate_ratio"] = (
        data["flow_rate_btc_per_second"]
        / data["trailing_volume_rate_btc_per_second"]
    )
    return data.replace([np.inf, -np.inf], np.nan)


def add_normalised_impact_trajectories(
    observations: pd.DataFrame,
    *,
    horizons: Sequence[int] = (1, 2, 3, 5),
    current_impact_col: str = "signed_current_impact_bps",
    volatility_col: str = "trailing_realised_volatility_bps",
) -> pd.DataFrame:
    """Add normalized total-impact and post-bar response trajectories.

    Total impact at horizon ``h`` is contemporaneous signed impact plus the
    cumulative return from the current bar end to bar ``t+h``, signed using
    the current imbalance direction. Horizon zero is contemporaneous impact.
    """

    horizon_values = tuple(int(value) for value in horizons)
    if any(value < 1 for value in horizon_values):
        raise ValueError("horizons must contain positive integers.")
    required = {current_impact_col, volatility_col}
    required.update(
        f"signed_future_return_{horizon}_bps" for horizon in horizon_values
    )
    missing = required.difference(observations.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    data = observations.copy()
    volatility = pd.to_numeric(data[volatility_col], errors="coerce")
    current = pd.to_numeric(data[current_impact_col], errors="coerce")
    data["normalised_total_impact_0"] = current / volatility
    data["normalised_post_response_0"] = 0.0
    for horizon in horizon_values:
        future_col = f"signed_future_return_{horizon}_bps"
        future = pd.to_numeric(data[future_col], errors="coerce")
        data[f"normalised_post_response_{horizon}"] = future / volatility
        data[f"normalised_total_impact_{horizon}"] = (
            current + future
        ) / volatility
    return data.replace([np.inf, -np.inf], np.nan)


def summarise_normalised_impact_trajectory(
    observations: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    horizons: Sequence[int] = (0, 1, 2, 3, 5),
) -> pd.DataFrame:
    """Summarise normalized impact trajectories and fractions remaining."""

    groups = tuple(group_cols)
    if not groups:
        raise ValueError("At least one grouping column is required.")
    horizon_values = tuple(int(value) for value in horizons)
    required = set(groups)
    required.update(
        f"normalised_total_impact_{horizon}" for horizon in horizon_values
    )
    missing = required.difference(observations.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    group_key: str | list[str]
    group_key = groups[0] if len(groups) == 1 else list(groups)
    for keys, group in observations.groupby(group_key, observed=True):
        key_values = (keys,) if len(groups) == 1 else tuple(keys)
        labels = dict(zip(groups, key_values, strict=True))
        for horizon in horizon_values:
            values = group[f"normalised_total_impact_{horizon}"].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    **labels,
                    "horizon": horizon,
                    "count": len(values),
                    "mean_total_impact": values.mean(),
                    "median_total_impact": values.median(),
                    "impact_std": values.std(),
                    "standard_error": values.std() / np.sqrt(len(values)),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    baseline = result.loc[
        result["horizon"] == 0,
        [*groups, "mean_total_impact"],
    ].rename(columns={"mean_total_impact": "mean_initial_impact"})
    result = result.merge(baseline, on=list(groups), how="left")
    result["fraction_remaining"] = (
        result["mean_total_impact"] / result["mean_initial_impact"]
    )
    result["fraction_reversed"] = 1.0 - result["fraction_remaining"]
    return result


def prepare_impact_observations(windows: pd.DataFrame) -> pd.DataFrame:
    """Return valid fixed-window observations for impact estimation."""

    required = {"signed_imbalance", "abs_signed_imbalance", "gross_volume", "absolute_imbalance_ratio", "signed_impact"}
    legacy_required = {"q_t", "signed_impact"}
    if required.issubset(windows.columns):
        observations = windows.copy()
    elif legacy_required.issubset(windows.columns):
        observations = windows.copy()
        observations["signed_imbalance"] = observations["q_t"]
        observations["abs_signed_imbalance"] = observations["q_t"].abs()
        if "absolute_volume" in observations.columns:
            observations["gross_volume"] = observations["absolute_volume"]
            observations["absolute_imbalance_ratio"] = observations["abs_signed_imbalance"] / observations["gross_volume"]
    else:
        missing = required.difference(windows.columns)
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    observations["abs_imbalance"] = observations["abs_signed_imbalance"]
    observations = observations.loc[
        (observations["abs_signed_imbalance"] > 0)
        & (observations["gross_volume"] > 0)
        & observations["absolute_imbalance_ratio"].notna()
    ].copy()
    return observations
