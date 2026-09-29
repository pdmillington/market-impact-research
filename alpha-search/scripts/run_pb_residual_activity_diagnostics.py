"""Diagnose remaining activity structure in scale-free P&B residuals."""

from __future__ import annotations

import argparse
from dataclasses import MISSING
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))

from alpha_search.pb_scaling import PBScalingFit, predict_pb_scaling  # noqa: E402


def utc_ms(value: str) -> int:
    return int(
        datetime.strptime(value, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1_000
    )


def fit_from_row(row: dict) -> PBScalingFit:
    fields = {}
    for key, definition in PBScalingFit.__dataclass_fields__.items():
        if key in row:
            fields[key] = row[key]
        elif definition.default is not MISSING:
            fields[key] = definition.default
    return PBScalingFit(**fields)


def finite_base(
    z: np.ndarray,
    activity: np.ndarray,
    impact: np.ndarray,
    side: np.ndarray,
) -> np.ndarray:
    return (
        np.isfinite(z)
        & np.isfinite(activity)
        & np.isfinite(impact)
        & (z > 0)
        & (activity > 0)
        & (side != 0)
    )


def model_coordinates(
    fit: PBScalingFit,
    n_events: int,
    side: np.ndarray,
    z: np.ndarray,
    activity: np.ndarray,
    impact: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ratio = n_events / float(fit.reference_events)
    prediction = predict_pb_scaling(
        fit,
        np.full(len(z), n_events),
        side,
        z,
        activity,
    )
    raw_residual = impact - prediction
    r_reference = np.where(
        side > 0, fit.r_buy_reference, fit.r_sell_reference
    )
    response_scale = r_reference * ratio**fit.r_exponent_psi
    scaled_residual = raw_residual / response_scale
    q_scale = fit.q_reference * ratio**fit.q_exponent_xi
    scaled_pressure = z * activity**fit.activity_exponent_gamma / q_scale
    normal_activity = fit.activity_reference * (
        ratio**fit.activity_scale_exponent_eta
    )
    log_relative_activity = np.log(activity / normal_activity)
    return raw_residual, scaled_residual, scaled_pressure, log_relative_activity


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or np.std(x[finite]) == 0 or np.std(y[finite]) == 0:
        return np.nan
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def slope(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or np.var(x[finite]) == 0:
        return np.nan
    centred_x = x[finite] - np.mean(x[finite])
    centred_y = y[finite] - np.mean(y[finite])
    return float(np.dot(centred_x, centred_y) / np.dot(centred_x, centred_x))


def pressure_controlled_relation(
    activity: np.ndarray,
    residual: np.ndarray,
    pressure_labels: np.ndarray,
) -> tuple[float, float]:
    adjusted_activity = np.full(len(activity), np.nan)
    adjusted_residual = np.full(len(residual), np.nan)
    for label in np.unique(pressure_labels):
        chosen = pressure_labels == label
        adjusted_activity[chosen] = activity[chosen] - np.mean(activity[chosen])
        adjusted_residual[chosen] = residual[chosen] - np.mean(residual[chosen])
    return (
        safe_correlation(adjusted_residual, adjusted_activity),
        slope(adjusted_activity, adjusted_residual),
    )


def descriptive_stats(values: np.ndarray, prefix: str) -> dict:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            f"mean_{prefix}": np.nan,
            f"median_{prefix}": np.nan,
            f"standard_error_{prefix}": np.nan,
        }
    return {
        f"mean_{prefix}": float(np.mean(finite)),
        f"median_{prefix}": float(np.median(finite)),
        f"standard_error_{prefix}": float(
            np.std(finite, ddof=1) / np.sqrt(len(finite))
        ) if len(finite) > 1 else np.nan,
    }


def cell_diagnostics(
    cells: pl.DataFrame,
    fits: pl.DataFrame,
    family: str,
) -> pl.DataFrame:
    rows = []
    scenarios = (
        cells.filter(pl.col("split") == "test")
        .select("sample", "fold", "n_events", "side")
        .unique()
        .sort(["sample", "fold", "n_events", "side"])
    )
    for scenario in scenarios.iter_rows(named=True):
        frame = cells.filter(
            (pl.col("sample") == scenario["sample"])
            & (pl.col("fold") == scenario["fold"])
            & (pl.col("split") == "test")
            & (pl.col("n_events") == scenario["n_events"])
            & (pl.col("side") == scenario["side"])
        )
        fit_row = fits.filter(
            (pl.col("sample") == scenario["sample"])
            & (pl.col("fold") == scenario["fold"])
            & (pl.col("family") == family)
        ).row(0, named=True)
        fit = fit_from_row(fit_row)
        side_value = 1 if scenario["side"] == "buy" else -1
        side = np.full(frame.height, side_value, dtype=np.int8)
        raw, scaled, pressure, log_activity = model_coordinates(
            fit,
            int(scenario["n_events"]),
            side,
            frame["mean_relative_imbalance"].to_numpy(),
            frame["mean_relative_activity"].to_numpy(),
            frame["mean_normalised_impact"].to_numpy(),
        )
        enriched = frame.with_columns(
            pl.lit(family).alias("family"),
            pl.Series("raw_residual", raw),
            pl.Series("scaled_residual", scaled),
            pl.Series("scaled_pressure", pressure),
            pl.Series("log_relative_activity", log_activity),
        )
        rows.extend(enriched.to_dicts())
    return pl.DataFrame(rows)


def bar_diagnostics(
    scaling_config: dict,
    diagnostic_config: dict,
    fits: pl.DataFrame,
    cache_root: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    curve_rows = []
    grid_rows = []
    monthly_rows = []
    family = diagnostic_config["model_family"]
    activity_bins = int(diagnostic_config["activity_bins"])
    pressure_bins = int(diagnostic_config["pressure_bins"])
    minimum = int(diagnostic_config["minimum_group_observations"])

    samples = {sample["name"]: sample for sample in scaling_config["samples"]}
    for n_events in scaling_config["event_sizes"]:
        print(f"Diagnosing event_{n_events}", flush=True)
        path = cache_root / f"event_{n_events}" / "part-000.parquet"
        frame = (
            pl.scan_parquet(path)
            .select(
                "end_time_ms",
                "signed_imbalance",
                "relative_imbalance",
                "relative_activity",
                "normalised_signed_impact",
            )
            .filter(pl.col("end_time_ms") < utc_ms(scaling_config["descriptive_fit_end"]))
            .collect()
            .rechunk()
        )
        time = frame["end_time_ms"].to_numpy()
        side = np.sign(frame["signed_imbalance"].to_numpy()).astype(np.int8)
        z = frame["relative_imbalance"].to_numpy()
        activity = frame["relative_activity"].to_numpy()
        impact = frame["normalised_signed_impact"].to_numpy()
        finite = finite_base(z, activity, impact, side)

        for sample_name, sample in samples.items():
            sample_start = utc_ms(sample["start"])
            for fold in scaling_config["folds"]:
                fold_label = str(fold["fold"])
                train_end = utc_ms(fold["train_end"])
                test_start = utc_ms(fold["test_start"])
                test_end = utc_ms(fold["test_end"])
                fit_row = fits.filter(
                    (pl.col("sample") == sample_name)
                    & (pl.col("fold") == fold_label)
                    & (pl.col("family") == family)
                ).row(0, named=True)
                fit = fit_from_row(fit_row)
                train = finite & (time >= sample_start) & (time < train_end)
                test = finite & (time >= test_start) & (time < test_end)
                if train.sum() < minimum or test.sum() < minimum:
                    continue
                _, _, train_pressure, train_log_activity = model_coordinates(
                    fit,
                    int(n_events),
                    side[train],
                    z[train],
                    activity[train],
                    impact[train],
                )
                raw, scaled, pressure, log_activity = model_coordinates(
                    fit,
                    int(n_events),
                    side[test],
                    z[test],
                    activity[test],
                    impact[test],
                )
                activity_cuts = np.unique(
                    np.quantile(
                        train_log_activity,
                        np.linspace(0.0, 1.0, activity_bins + 1)[1:-1],
                    )
                )
                pressure_cuts = np.unique(
                    np.quantile(
                        train_pressure,
                        np.linspace(0.0, 1.0, pressure_bins + 1)[1:-1],
                    )
                )
                activity_labels = np.digitize(
                    log_activity, activity_cuts, right=True
                )
                pressure_labels = np.digitize(
                    pressure, pressure_cuts, right=True
                )
                local_side = side[test]
                local_time = time[test]
                months = local_time.astype("datetime64[ms]").astype("datetime64[M]")
                side_scopes = {
                    "all": np.ones(len(local_side), dtype=bool),
                    "buy": local_side > 0,
                    "sell": local_side < 0,
                }
                base_metadata = {
                    "sample": sample_name,
                    "fold": fold_label,
                    "family": family,
                    "n_events": int(n_events),
                }
                for side_name, side_scope in side_scopes.items():
                    for activity_label in range(len(activity_cuts) + 1):
                        chosen = side_scope & (activity_labels == activity_label)
                        if chosen.sum() < minimum:
                            continue
                        curve_rows.append(
                            {
                                **base_metadata,
                                "side": side_name,
                                "activity_bin": activity_label + 1,
                                "observations": int(chosen.sum()),
                                "mean_log_relative_activity": float(
                                    np.mean(log_activity[chosen])
                                ),
                                "mean_scaled_pressure": float(
                                    np.mean(pressure[chosen])
                                ),
                                **descriptive_stats(raw[chosen], "raw_residual"),
                                **descriptive_stats(
                                    scaled[chosen], "scaled_residual"
                                ),
                            }
                        )
                    for pressure_label in range(len(pressure_cuts) + 1):
                        for activity_label in range(len(activity_cuts) + 1):
                            chosen = (
                                side_scope
                                & (pressure_labels == pressure_label)
                                & (activity_labels == activity_label)
                            )
                            if chosen.sum() < minimum:
                                continue
                            grid_rows.append(
                                {
                                    **base_metadata,
                                    "side": side_name,
                                    "pressure_bin": pressure_label + 1,
                                    "activity_bin": activity_label + 1,
                                    "observations": int(chosen.sum()),
                                    "mean_log_relative_activity": float(
                                        np.mean(log_activity[chosen])
                                    ),
                                    "mean_scaled_pressure": float(
                                        np.mean(pressure[chosen])
                                    ),
                                    **descriptive_stats(raw[chosen], "raw_residual"),
                                    **descriptive_stats(
                                        scaled[chosen], "scaled_residual"
                                    ),
                                }
                            )
                    for month in np.unique(months):
                        chosen = side_scope & (months == month)
                        if chosen.sum() < minimum:
                            continue
                        partial_raw_corr, partial_raw_slope = (
                            pressure_controlled_relation(
                                log_activity[chosen],
                                raw[chosen],
                                pressure_labels[chosen],
                            )
                        )
                        partial_scaled_corr, partial_scaled_slope = (
                            pressure_controlled_relation(
                                log_activity[chosen],
                                scaled[chosen],
                                pressure_labels[chosen],
                            )
                        )
                        monthly_rows.append(
                            {
                                **base_metadata,
                                "side": side_name,
                                "test_month": str(month),
                                "observations": int(chosen.sum()),
                                "raw_activity_correlation": safe_correlation(
                                    raw[chosen], log_activity[chosen]
                                ),
                                "raw_activity_slope": slope(
                                    log_activity[chosen], raw[chosen]
                                ),
                                "scaled_activity_correlation": safe_correlation(
                                    scaled[chosen], log_activity[chosen]
                                ),
                                "scaled_activity_slope": slope(
                                    log_activity[chosen], scaled[chosen]
                                ),
                                "pressure_controlled_raw_correlation": partial_raw_corr,
                                "pressure_controlled_raw_slope": partial_raw_slope,
                                "pressure_controlled_scaled_correlation": partial_scaled_corr,
                                "pressure_controlled_scaled_slope": partial_scaled_slope,
                            }
                        )
        del frame
    return (
        pl.DataFrame(curve_rows),
        pl.DataFrame(grid_rows),
        pl.DataFrame(monthly_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO
            / "alpha-search"
            / "configs"
            / "pb_residual_activity_diagnostics_v1.json"
        ),
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    scaling_config_path = REPO / config["scaling_config"]
    scaling_config = json.loads(scaling_config_path.read_text())
    scaling_results = (
        REPO / "alpha-search" / "reports" / config["scaling_experiment"]
    )
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

    cells = pl.read_csv(
        scaling_results / "scaling_cells.csv", infer_schema_length=None
    )
    fits = pl.read_csv(
        scaling_results / "scaling_fits.csv", infer_schema_length=None
    )
    cell_output = cell_diagnostics(cells, fits, config["model_family"])
    curves, grids, monthly = bar_diagnostics(
        scaling_config, config, fits, cache_root
    )
    cell_output.write_csv(output / "cell_residual_activity.csv")
    curves.write_csv(output / "bar_activity_curves.csv")
    grids.write_csv(output / "bar_pressure_activity_grid.csv")
    monthly.write_csv(output / "monthly_activity_relations.csv")
    manifest = {
        "experiment_id": config["experiment_id"],
        "config": str(args.config.resolve()),
        "scaling_results": str(scaling_results.resolve()),
        "cache_root": str(cache_root.resolve()),
        "outputs": [
            "cell_residual_activity.csv",
            "bar_activity_curves.csv",
            "bar_pressure_activity_grid.csv",
            "monthly_activity_relations.csv",
        ],
        "holdout_months_excluded": config["holdout_months_excluded"],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
