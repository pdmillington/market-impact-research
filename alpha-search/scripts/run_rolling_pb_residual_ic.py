"""Rolling scale-free P&B residual correlations with future returns."""

from __future__ import annotations

import argparse
from dataclasses import MISSING, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
from scipy.stats import rankdata


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))

from alpha_search.impact_surface import SurfaceCells, surface_cells  # noqa: E402
from alpha_search.pb_scaling import fit_pb_scaling, predict_pb_scaling  # noqa: E402
from alpha_search.research import add_months  # noqa: E402


def utc_ms(value: str) -> int:
    return int(
        datetime.strptime(value, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1_000
    )


def month_starts(start: str, end: str) -> list[str]:
    values = []
    current = start[:7]
    final = end[:7]
    while current <= final:
        values.append(f"{current}-01")
        current = add_months(current, 1)
    return values


def cell_rows(cells: SurfaceCells, *, metadata: dict) -> list[dict]:
    return [
        {
            **metadata,
            "mean_relative_imbalance": float(cells.mean_imbalance[index]),
            "mean_relative_activity": float(cells.mean_activity[index]),
            "mean_normalised_impact": float(cells.mean_impact[index]),
            "observations": int(cells.observations[index]),
        }
        for index in range(len(cells.mean_impact))
    ]


def frame_as_cells(frame: pl.DataFrame) -> SurfaceCells:
    size = frame.height
    return SurfaceCells(
        imbalance_cuts=np.asarray([], dtype=float),
        activity_cuts=np.asarray([], dtype=float),
        mean_imbalance=frame["mean_relative_imbalance"].to_numpy(),
        mean_activity=frame["mean_relative_activity"].to_numpy(),
        mean_impact=frame["mean_normalised_impact"].to_numpy(),
        standard_error=np.ones(size, dtype=float),
        observations=frame["observations"].to_numpy(),
        imbalance_bin=np.zeros(size, dtype=np.int16),
        activity_bin=np.zeros(size, dtype=np.int16),
    )


def prepare_rolling_cells(config: dict, cache_root: Path) -> pl.DataFrame:
    dates = month_starts(config["calibration_start"], config["calibration_end"])
    n_z = int(config["surface"]["imbalance_bins"])
    n_a = int(config["surface"]["activity_bins"])
    rows: list[dict] = []
    for n_events in config["event_sizes"]:
        path = cache_root / f"event_{n_events}" / "part-000.parquet"
        print(f"Preparing rolling cells for event_{n_events}", flush=True)
        frame = (
            pl.scan_parquet(path)
            .select(
                "end_time_ms",
                "signed_imbalance",
                "relative_imbalance",
                "relative_activity",
                "normalised_signed_impact",
            )
            .collect()
            .rechunk()
        )
        time = frame["end_time_ms"].to_numpy()
        side = np.sign(frame["signed_imbalance"].to_numpy()).astype(np.int8)
        z = frame["relative_imbalance"].to_numpy()
        activity = frame["relative_activity"].to_numpy()
        impact = frame["normalised_signed_impact"].to_numpy()
        finite = (
            np.isfinite(z)
            & np.isfinite(activity)
            & np.isfinite(impact)
            & (z > 0)
            & (activity > 0)
            & (side != 0)
        )
        for calibration_date in dates:
            train_end = utc_ms(calibration_date)
            for window_months in config["training_windows_months"]:
                train_start = utc_ms(
                    f"{add_months(calibration_date[:7], -int(window_months))}-01"
                )
                selected_period = finite & (time >= train_start) & (time < train_end)
                for side_value, side_name in [(-1, "sell"), (1, "buy")]:
                    selected = selected_period & (side == side_value)
                    if selected.sum() < n_z * n_a * 3:
                        continue
                    cells = surface_cells(
                        z[selected],
                        activity[selected],
                        impact[selected],
                        imbalance_bins=n_z,
                        activity_bins=n_a,
                    )
                    rows.extend(
                        cell_rows(
                            cells,
                            metadata={
                                "calibration_date": calibration_date,
                                "training_window_months": int(window_months),
                                "n_events": int(n_events),
                                "side": side_name,
                            },
                        )
                    )
        del frame
    return pl.DataFrame(rows)


def fit_rolling_models(cells: pl.DataFrame, config: dict) -> pl.DataFrame:
    rows = []
    groups = (
        cells.select("calibration_date", "training_window_months")
        .unique()
        .sort(["calibration_date", "training_window_months"])
    )
    for group in groups.iter_rows(named=True):
        calibration_date = group["calibration_date"]
        window = group["training_window_months"]
        training = cells.filter(
            (pl.col("calibration_date") == calibration_date)
            & (pl.col("training_window_months") == window)
        )
        side = np.where(training["side"].to_numpy() == "buy", 1, -1)
        fit = fit_pb_scaling(
            training["n_events"].to_numpy(),
            side,
            training["mean_relative_imbalance"].to_numpy(),
            training["mean_relative_activity"].to_numpy(),
            training["mean_normalised_impact"].to_numpy(),
            family=config["model_family"],
            reference_events=int(config["reference_events"]),
            n_starts=int(config["multi_starts"]),
            random_state=12_071 + int(window),
        )
        rows.append(
            {
                "calibration_date": calibration_date,
                "training_window_months": window,
                "training_cells": training.height,
                **asdict(fit),
                "tail_exponent": fit.tail_exponent,
            }
        )
        print(f"Fitted {calibration_date}, {window}m", flush=True)
    return pl.DataFrame(rows)


def fit_from_row(row: dict):
    from alpha_search.pb_scaling import PBScalingFit

    fields = {}
    for key, definition in PBScalingFit.__dataclass_fields__.items():
        if key in row:
            fields[key] = row[key]
        elif definition.default is not MISSING:
            fields[key] = definition.default
    return PBScalingFit(**fields)


def future_targets(
    decision_time: np.ndarray,
    decision_price: np.ndarray,
    price_clock_time: np.ndarray,
    price_clock_price: np.ndarray,
    horizons_seconds: list[int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    targets = {}
    for horizon_seconds in horizons_seconds:
        desired = decision_time + int(horizon_seconds) * 1_000
        index = np.searchsorted(price_clock_time, desired, side="left")
        valid = index < len(price_clock_time)
        returns = np.full(len(decision_time), np.nan)
        elapsed_seconds = np.full(len(decision_time), np.nan)
        returns[valid] = 10_000.0 * np.log(
            price_clock_price[index[valid]] / decision_price[valid]
        )
        elapsed_seconds[valid] = (
            price_clock_time[index[valid]] - decision_time[valid]
        ) / 1_000.0
        targets[int(horizon_seconds)] = (returns, elapsed_seconds)
    return targets


def configured_horizons_seconds(config: dict) -> list[int]:
    """Return exact horizons while retaining compatibility with the v1 config."""

    if "future_horizons_seconds" in config:
        values = [int(value) for value in config["future_horizons_seconds"]]
    else:
        values = [int(value) * 60 for value in config["future_horizons_minutes"]]
    if not values or any(value <= 0 for value in values):
        raise ValueError("Future horizons must contain positive durations.")
    if len(set(values)) != len(values):
        raise ValueError("Future horizons must be unique.")
    return sorted(values)


def conditional_residual_z(
    calibration_residual: np.ndarray,
    calibration_pressure: np.ndarray,
    calibration_side: np.ndarray,
    test_residual: np.ndarray,
    test_pressure: np.ndarray,
    test_side: np.ndarray,
    *,
    pressure_bins: int,
    minimum_bin_observations: int = 100,
) -> np.ndarray:
    """Past-only robust z-score conditional on side and scaled-pressure bin."""

    output = np.full(len(test_residual), np.nan)
    for side_value in (-1, 1):
        train = (
            (calibration_side == side_value)
            & np.isfinite(calibration_residual)
            & np.isfinite(calibration_pressure)
        )
        test = (
            (test_side == side_value)
            & np.isfinite(test_residual)
            & np.isfinite(test_pressure)
        )
        if train.sum() < minimum_bin_observations or not test.any():
            continue
        train_residual = calibration_residual[train]
        train_pressure = calibration_pressure[train]
        cuts = np.unique(
            np.quantile(
                train_pressure,
                np.linspace(0.0, 1.0, pressure_bins + 1)[1:-1],
            )
        )
        train_labels = np.digitize(train_pressure, cuts, right=True)
        test_indices = np.flatnonzero(test)
        test_labels = np.digitize(test_pressure[test], cuts, right=True)
        global_centre = float(np.median(train_residual))
        global_mad = float(np.median(np.abs(train_residual - global_centre)))
        global_scale = max(1.4826 * global_mad, float(train_residual.std()), 1e-8)
        for label in range(len(cuts) + 1):
            train_bin = train_labels == label
            test_bin = test_labels == label
            if not test_bin.any():
                continue
            if train_bin.sum() >= minimum_bin_observations:
                values = train_residual[train_bin]
                centre = float(np.median(values))
                mad = float(np.median(np.abs(values - centre)))
                scale = max(1.4826 * mad, 1e-8)
            else:
                centre = global_centre
                scale = global_scale
            output[test_indices[test_bin]] = (
                test_residual[test_indices[test_bin]] - centre
            ) / scale
    return output


def correlation_row(
    residual: np.ndarray,
    target: np.ndarray,
    elapsed: np.ndarray,
) -> dict:
    finite = np.isfinite(residual) & np.isfinite(target) & np.isfinite(elapsed)
    x = residual[finite]
    y = target[finite]
    actual_elapsed = elapsed[finite]
    if len(x) < 20:
        return {"observations": len(x)}
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    lower, upper = np.quantile(x, [0.2, 0.8])
    bottom = y[x <= lower]
    top = y[x >= upper]
    winsor_lower, winsor_upper = np.quantile(y, [0.01, 0.99])
    winsorised = np.clip(y, winsor_lower, winsor_upper)
    winsorised_bottom = winsorised[x <= lower]
    winsorised_top = winsorised[x >= upper]
    return {
        "observations": len(x),
        "pearson_ic": pearson,
        "spearman_ic": spearman,
        "residual_std": float(x.std()),
        "future_return_std_bps": float(y.std()),
        "bottom_residual_quintile_mean_bps": float(bottom.mean()),
        "top_residual_quintile_mean_bps": float(top.mean()),
        "top_minus_bottom_bps": float(top.mean() - bottom.mean()),
        "winsorised_top_minus_bottom_bps": float(
            winsorised_top.mean() - winsorised_bottom.mean()
        ),
        "mean_residual": float(x.mean()),
        "mean_future_return_bps": float(y.mean()),
        "sum_residual": float(x.sum()),
        "sum_future_return": float(y.sum()),
        "sum_residual_squared": float(np.dot(x, x)),
        "sum_future_return_squared": float(np.dot(y, y)),
        "sum_cross_product": float(np.dot(x, y)),
        "median_elapsed_seconds": float(np.median(actual_elapsed)),
        "p90_elapsed_seconds": float(np.quantile(actual_elapsed, 0.9)),
    }


def residual_quantile_rows(
    residual: np.ndarray,
    target: np.ndarray,
    elapsed: np.ndarray,
    *,
    quantiles: int = 5,
) -> list[dict]:
    finite = np.isfinite(residual) & np.isfinite(target) & np.isfinite(elapsed)
    x = residual[finite]
    y = target[finite]
    actual_elapsed = elapsed[finite]
    if len(x) < quantiles * 10:
        return []
    cuts = np.unique(np.quantile(x, np.linspace(0, 1, quantiles + 1)[1:-1]))
    labels = np.digitize(x, cuts, right=True)
    winsor_lower, winsor_upper = np.quantile(y, [0.01, 0.99])
    winsorised = np.clip(y, winsor_lower, winsor_upper)
    rows = []
    for label in range(len(cuts) + 1):
        chosen = labels == label
        if not chosen.any():
            continue
        rows.append(
            {
                "residual_quantile": label + 1,
                "observations": int(chosen.sum()),
                "mean_scaled_residual": float(x[chosen].mean()),
                "median_scaled_residual": float(np.median(x[chosen])),
                "mean_future_return_bps": float(y[chosen].mean()),
                "median_future_return_bps": float(np.median(y[chosen])),
                "winsorised_mean_future_return_bps": float(
                    winsorised[chosen].mean()
                ),
                "median_elapsed_seconds": float(np.median(actual_elapsed[chosen])),
            }
        )
    return rows


def evaluate_residuals(
    fits: pl.DataFrame,
    config: dict,
    cache_root: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    price_path = cache_root / "event_200" / "part-000.parquet"
    price_clock = (
        pl.scan_parquet(price_path)
        .select("end_time_ms", "end_price")
        .collect()
        .rechunk()
    )
    price_clock_time = price_clock["end_time_ms"].to_numpy()
    price_clock_price = price_clock["end_price"].to_numpy()
    horizons_seconds = configured_horizons_seconds(config)
    standardisation_months = int(config.get("standardisation_lookback_months", 3))
    pressure_bins = int(config.get("standardisation_pressure_bins", 5))
    rows = []
    quantile_output = []
    for n_events in config["event_sizes"]:
        print(f"Evaluating residual IC for event_{n_events}", flush=True)
        path = cache_root / f"event_{n_events}" / "part-000.parquet"
        frame = (
            pl.scan_parquet(path)
            .select(
                "end_time_ms",
                "end_price",
                "signed_imbalance",
                "relative_imbalance",
                "relative_activity",
                "normalised_signed_impact",
            )
            .collect()
            .rechunk()
        )
        time = frame["end_time_ms"].to_numpy()
        price = frame["end_price"].to_numpy()
        side = np.sign(frame["signed_imbalance"].to_numpy()).astype(np.int8)
        z = frame["relative_imbalance"].to_numpy()
        activity = frame["relative_activity"].to_numpy()
        impact = frame["normalised_signed_impact"].to_numpy()
        all_targets = future_targets(
            time,
            price,
            price_clock_time,
            price_clock_price,
            horizons_seconds,
        )
        for fit_row in fits.iter_rows(named=True):
            calibration_date = fit_row["calibration_date"]
            test_start = utc_ms(calibration_date)
            test_end = utc_ms(f"{add_months(calibration_date[:7], 1)}-01")
            selected = (
                (time >= test_start)
                & (time < test_end)
                & np.isfinite(z)
                & np.isfinite(activity)
                & np.isfinite(impact)
                & (z > 0)
                & (activity > 0)
                & (side != 0)
            )
            if selected.sum() < 20:
                continue
            fit = fit_from_row(fit_row)
            local_side = side[selected]
            prediction = predict_pb_scaling(
                fit,
                np.full(selected.sum(), n_events),
                local_side,
                z[selected],
                activity[selected],
            )
            ratio = n_events / float(fit.reference_events)
            r_reference = np.where(
                local_side > 0, fit.r_buy_reference, fit.r_sell_reference
            )
            response_scale = r_reference * ratio**fit.r_exponent_psi
            scaled_residual = (impact[selected] - prediction) / response_scale
            q_scale = fit.q_reference * ratio**fit.q_exponent_xi
            scaled_pressure = (
                z[selected]
                * np.power(activity[selected], fit.activity_exponent_gamma)
                / q_scale
            )

            standardisation_start = utc_ms(
                f"{add_months(calibration_date[:7], -standardisation_months)}-01"
            )
            calibration_selected = (
                (time >= standardisation_start)
                & (time < test_start)
                & np.isfinite(z)
                & np.isfinite(activity)
                & np.isfinite(impact)
                & (z > 0)
                & (activity > 0)
                & (side != 0)
            )
            calibration_side = side[calibration_selected]
            calibration_prediction = predict_pb_scaling(
                fit,
                np.full(calibration_selected.sum(), n_events),
                calibration_side,
                z[calibration_selected],
                activity[calibration_selected],
            )
            calibration_reference = np.where(
                calibration_side > 0, fit.r_buy_reference, fit.r_sell_reference
            )
            calibration_response_scale = (
                calibration_reference * ratio**fit.r_exponent_psi
            )
            calibration_residual = (
                impact[calibration_selected] - calibration_prediction
            ) / calibration_response_scale
            calibration_pressure = (
                z[calibration_selected]
                * np.power(
                    activity[calibration_selected], fit.activity_exponent_gamma
                )
                / q_scale
            )
            conditional_z = conditional_residual_z(
                calibration_residual,
                calibration_pressure,
                calibration_side,
                scaled_residual,
                scaled_pressure,
                local_side,
                pressure_bins=pressure_bins,
            )
            residual_variants = {
                "scaled": scaled_residual,
                "conditional_z": conditional_z,
            }
            side_scopes = {
                "all": np.ones(len(local_side), dtype=bool),
                "buy": local_side > 0,
                "sell": local_side < 0,
            }
            for horizon_seconds in horizons_seconds:
                future_return, elapsed = all_targets[horizon_seconds]
                aligned_return = local_side * future_return[selected]
                for residual_variant, residual_values in residual_variants.items():
                    for side_name, side_scope in side_scopes.items():
                        metrics = correlation_row(
                            residual_values[side_scope],
                            aligned_return[side_scope],
                            elapsed[selected][side_scope],
                        )
                        metadata = {
                            "calibration_date": calibration_date,
                            "test_month": calibration_date[:7],
                            "training_window_months": fit_row[
                                "training_window_months"
                            ],
                            "n_events": int(n_events),
                            "side": side_name,
                            "residual_variant": residual_variant,
                            "horizon_seconds": int(horizon_seconds),
                            "horizon_minutes": float(horizon_seconds / 60.0),
                        }
                        rows.append({**metadata, **metrics})
                        for quantile_row in residual_quantile_rows(
                            residual_values[side_scope],
                            aligned_return[side_scope],
                            elapsed[selected][side_scope],
                        ):
                            quantile_output.append({**metadata, **quantile_row})
        del frame
    return pl.DataFrame(rows), pl.DataFrame(quantile_output)


def pooled_summary(monthly: pl.DataFrame) -> pl.DataFrame:
    group_columns = [
        "training_window_months",
        "n_events",
        "side",
        "residual_variant",
        "horizon_seconds",
        "horizon_minutes",
    ]
    rows = []
    for keys, frame in monthly.group_by(group_columns, maintain_order=True):
        window, n_events, side, residual_variant, horizon_seconds, horizon = keys
        count = int(frame["observations"].sum())
        sum_x = float(frame["sum_residual"].sum())
        sum_y = float(frame["sum_future_return"].sum())
        sum_x2 = float(frame["sum_residual_squared"].sum())
        sum_y2 = float(frame["sum_future_return_squared"].sum())
        sum_xy = float(frame["sum_cross_product"].sum())
        covariance = sum_xy - sum_x * sum_y / count
        variance_x = sum_x2 - sum_x**2 / count
        variance_y = sum_y2 - sum_y**2 / count
        pooled_ic = covariance / np.sqrt(variance_x * variance_y)
        monthly_spearman = frame["spearman_ic"].to_numpy()
        mean_monthly = float(np.mean(monthly_spearman))
        standard_error = float(np.std(monthly_spearman, ddof=1) / np.sqrt(len(frame)))
        rows.append(
            {
                "training_window_months": int(window),
                "n_events": int(n_events),
                "side": side,
                "residual_variant": residual_variant,
                "horizon_seconds": int(horizon_seconds),
                "horizon_minutes": float(horizon),
                "months": frame.height,
                "observations": count,
                "pooled_pearson_ic": float(pooled_ic),
                "mean_monthly_spearman_ic": mean_monthly,
                "median_monthly_spearman_ic": float(np.median(monthly_spearman)),
                "monthly_spearman_t_stat": mean_monthly / standard_error,
                "negative_spearman_month_fraction": float(
                    np.mean(monthly_spearman < 0)
                ),
                "mean_top_minus_bottom_bps": float(
                    frame["top_minus_bottom_bps"].mean()
                ),
                "mean_winsorised_top_minus_bottom_bps": float(
                    frame["winsorised_top_minus_bottom_bps"].mean()
                ),
                "median_elapsed_seconds": float(
                    frame["median_elapsed_seconds"].median()
                ),
                "p90_elapsed_seconds": float(
                    frame["p90_elapsed_seconds"].median()
                ),
            }
        )
    return pl.DataFrame(rows).sort(group_columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO
            / "alpha-search"
            / "configs"
            / "rolling_pb_residual_response_v2.json"
        ),
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reuse-cells", action="store_true")
    parser.add_argument("--reuse-cells-from", type=Path)
    parser.add_argument("--reuse-fits-from", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cache_root = args.cache_root or (
        REPO
        / "shared-data"
        / "features"
        / "alpha"
        / f"symbol={config['symbol']}"
        / config["cache_version"]
    )
    output = args.output_dir or (
        REPO / "alpha-search" / "reports" / config["experiment_id"]
    )
    output.mkdir(parents=True, exist_ok=True)
    cells_path = output / "rolling_training_cells.parquet"
    if args.reuse_cells_from:
        cells = pl.read_parquet(args.reuse_cells_from)
        cells.write_parquet(cells_path, compression="zstd")
        print(f"Reused {args.reuse_cells_from}", flush=True)
    elif args.reuse_cells and cells_path.exists():
        print(f"Reusing {cells_path}", flush=True)
        cells = pl.read_parquet(cells_path)
    else:
        cells = prepare_rolling_cells(config, cache_root)
        cells.write_parquet(cells_path, compression="zstd")
    if args.reuse_fits_from:
        fits = pl.read_csv(args.reuse_fits_from, infer_schema_length=None)
        print(f"Reused {args.reuse_fits_from}", flush=True)
    else:
        fits = fit_rolling_models(cells, config)
    monthly, quantiles = evaluate_residuals(fits, config, cache_root)
    summary = pooled_summary(monthly)
    fits.write_csv(output / "rolling_fits.csv")
    monthly.write_csv(output / "monthly_residual_ic.csv")
    quantiles.write_csv(output / "monthly_residual_quantiles.csv")
    summary.write_csv(output / "residual_ic_summary.csv")
    manifest = {
        "experiment_id": config["experiment_id"],
        "config": str(args.config.resolve()),
        "cache_root": str(cache_root.resolve()),
        "reuse_cells_from": (
            str(args.reuse_cells_from.resolve()) if args.reuse_cells_from else None
        ),
        "reuse_fits_from": (
            str(args.reuse_fits_from.resolve()) if args.reuse_fits_from else None
        ),
        "outputs": [
            "rolling_training_cells.parquet",
            "rolling_fits.csv",
            "monthly_residual_ic.csv",
            "monthly_residual_quantiles.csv",
            "residual_ic_summary.csv"
        ],
        "holdout_months_excluded": config["holdout_months_excluded"],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
