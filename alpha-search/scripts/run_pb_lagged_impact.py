"""Test a previous-bar dynamic extension of a frozen scale-free P&B surface."""

from __future__ import annotations

import argparse
from dataclasses import MISSING, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))

from alpha_search.lagged_impact import (  # noqa: E402
    MODEL_FEATURES,
    fit_equal_period_ols,
    predict_lagged_correction,
)
from alpha_search.pb_scaling import (  # noqa: E402
    PBScalingFit,
    predict_pb_scaling,
)


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


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or np.std(x[finite]) == 0 or np.std(y[finite]) == 0:
        return np.nan
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def pressure_controlled_correlation(
    activity: np.ndarray,
    residual: np.ndarray,
    pressure_labels: np.ndarray,
) -> float:
    adjusted_activity = np.full(len(activity), np.nan)
    adjusted_residual = np.full(len(residual), np.nan)
    for label in np.unique(pressure_labels):
        chosen = pressure_labels == label
        if chosen.sum() < 2:
            continue
        adjusted_activity[chosen] = activity[chosen] - np.mean(activity[chosen])
        adjusted_residual[chosen] = residual[chosen] - np.mean(residual[chosen])
    return safe_correlation(adjusted_residual, adjusted_activity)


def prediction_metrics(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    residual = observed - predicted
    return (
        float(np.sqrt(np.mean(np.square(residual)))),
        float(np.mean(np.abs(residual))),
    )


def fit_row(
    fit,
    *,
    metadata: dict,
    duration_reference_seconds: float,
) -> dict:
    coefficients = dict(zip(fit.feature_names, fit.coefficients))
    return {
        **metadata,
        "model": fit.model,
        "training_observations": fit.observations,
        "training_months": fit.periods,
        "duration_reference_seconds": duration_reference_seconds,
        "scaled_condition_number": fit.scaled_condition_number,
        **{
            name: coefficients.get(name, 0.0)
            for name in sorted({item for names in MODEL_FEATURES.values() for item in names})
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO
            / "alpha-search"
            / "configs"
            / "pb_lagged_impact_extension_v4.json"
        ),
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    base_config_path = REPO / config["base_scaling_config"]
    base_config = json.loads(base_config_path.read_text())
    base_results = (
        REPO
        / "alpha-search"
        / "reports"
        / config["base_scaling_experiment"]
    )
    base_fits = pl.read_csv(
        base_results / "scaling_fits.csv", infer_schema_length=None
    ).with_columns(pl.col("fold").cast(pl.Utf8))
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

    evaluation_rows: list[dict] = []
    monthly_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    curve_rows: list[dict] = []
    ledger_rows: list[dict] = []

    samples = {row["name"]: row for row in base_config["samples"]}
    dynamic_models = [
        model for model in config["models"] if model != "frozen_pb"
    ]

    for n_events in config["event_sizes"]:
        path = cache_root / f"event_{n_events}" / "part-000.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Loading event_{n_events}", flush=True)
        frame = (
            pl.scan_parquet(path)
            .select(
                "bar_number",
                "end_time_ms",
                "duration_seconds",
                "signed_imbalance",
                "relative_imbalance",
                "relative_activity",
                "normalised_signed_impact",
            )
            .filter(pl.col("end_time_ms") < utc_ms(base_config["descriptive_fit_end"]))
            .sort("bar_number")
            .collect()
            .rechunk()
        )
        time = frame["end_time_ms"].to_numpy()
        duration = frame["duration_seconds"].to_numpy()
        side = np.sign(frame["signed_imbalance"].to_numpy()).astype(np.int8)
        z = frame["relative_imbalance"].to_numpy()
        activity = frame["relative_activity"].to_numpy()
        observed = frame["normalised_signed_impact"].to_numpy()
        month = time.astype("datetime64[ms]").astype("datetime64[M]").astype(np.int64)

        previous_side = np.roll(side, 1)
        previous_time = np.roll(time, 1)
        previous_z = np.roll(z, 1)
        previous_activity = np.roll(activity, 1)
        previous_side[0] = 0
        previous_time[0] = 0
        previous_z[0] = np.nan
        previous_activity[0] = np.nan

        for sample_name, sample in samples.items():
            sample_start = utc_ms(sample["start"])
            for fold in base_config["folds"]:
                fold_label = str(fold["fold"])
                train_end = utc_ms(fold["train_end"])
                test_start = utc_ms(fold["test_start"])
                test_end = utc_ms(fold["test_end"])
                base_row = base_fits.filter(
                    (pl.col("sample") == sample_name)
                    & (pl.col("fold") == fold_label)
                    & (pl.col("family") == config["base_model_family"])
                ).row(0, named=True)
                frozen_fit = fit_from_row(base_row)
                event_count_array = np.full(len(frame), n_events, dtype=float)
                base_prediction = predict_pb_scaling(
                    frozen_fit, event_count_array, side, z, activity
                )
                previous_prediction = predict_pb_scaling(
                    frozen_fit,
                    event_count_array,
                    previous_side,
                    previous_z,
                    previous_activity,
                )
                lag_impulse = side * previous_side * previous_prediction
                event_ratio = n_events / float(frozen_fit.reference_events)
                normal_activity = frozen_fit.activity_reference * (
                    event_ratio**frozen_fit.activity_scale_exponent_eta
                )
                previous_log_activity = np.log(previous_activity / normal_activity)
                current_log_activity = np.log(activity / normal_activity)
                q_scale = frozen_fit.q_reference * (
                    event_ratio**frozen_fit.q_exponent_xi
                )
                current_pressure = (
                    z * activity**frozen_fit.activity_exponent_gamma / q_scale
                )
                base_residual = observed - base_prediction
                finite = (
                    np.isfinite(observed)
                    & np.isfinite(base_prediction)
                    & np.isfinite(lag_impulse)
                    & np.isfinite(previous_log_activity)
                    & np.isfinite(current_log_activity)
                    & np.isfinite(current_pressure)
                    & np.isfinite(duration)
                    & (duration > 0)
                    & (side != 0)
                    & (previous_side != 0)
                )
                train = (
                    finite
                    & (time >= sample_start)
                    & (previous_time >= sample_start)
                    & (time < train_end)
                )
                test = finite & (time >= test_start) & (time < test_end)
                if train.sum() < 1_000 or test.sum() < 1_000:
                    raise ValueError(
                        f"Insufficient observations for {sample_name}, fold {fold_label}, "
                        f"event_{n_events}."
                    )
                duration_reference = float(np.median(duration[train]))
                log_duration = np.log(duration / duration_reference)
                pressure_cuts = np.unique(
                    np.quantile(current_pressure[train], np.linspace(0, 1, 6)[1:-1])
                )
                test_pressure_labels = np.digitize(
                    current_pressure[test], pressure_cuts, right=True
                )
                metadata = {
                    "sample": sample_name,
                    "fold": fold_label,
                    "n_events": int(n_events),
                    "pb_family": config["base_model_family"],
                    "pb_source_experiment": config["base_scaling_experiment"],
                    "train_start": sample["start"],
                    "pb_calibration_end": fold["train_end"],
                    "lag_calibration_end": fold["train_end"],
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                }
                ledger_rows.append(
                    {
                        **metadata,
                        "pb_parameters_frozen_before_lag_fit": True,
                        "pb_fit_target": base_config["surface"]["fit_target"],
                        "lag_fit_target": "bar_residual_after_frozen_pb",
                        "lag_weighting": config["calibration_weighting"],
                        "training_observations": int(train.sum()),
                        "test_observations": int(test.sum()),
                        "duration_reference_seconds": duration_reference,
                    }
                )

                fitted = {}
                for model in dynamic_models:
                    model_fit = fit_equal_period_ols(
                        model,
                        current_side=side[train],
                        lag_impulse=lag_impulse[train],
                        previous_log_activity=previous_log_activity[train],
                        log_duration=log_duration[train],
                        residual=base_residual[train],
                        period=month[train],
                    )
                    fitted[model] = model_fit
                    coefficient_rows.append(
                        fit_row(
                            model_fit,
                            metadata=metadata,
                            duration_reference_seconds=duration_reference,
                        )
                    )

                predictions = {"frozen_pb": base_prediction[test]}
                for model, model_fit in fitted.items():
                    correction = predict_lagged_correction(
                        model_fit,
                        current_side=side[test],
                        lag_impulse=lag_impulse[test],
                        previous_log_activity=previous_log_activity[test],
                        log_duration=log_duration[test],
                    )
                    predictions[model] = base_prediction[test] + correction

                test_months = month[test]
                for model, prediction in predictions.items():
                    residual = observed[test] - prediction
                    bar_rmse, bar_mae = prediction_metrics(observed[test], prediction)
                    local_month_rows = []
                    for month_value in np.unique(test_months):
                        chosen = test_months == month_value
                        local_rmse, local_mae = prediction_metrics(
                            observed[test][chosen], prediction[chosen]
                        )
                        activity_correlation = safe_correlation(
                            residual[chosen], current_log_activity[test][chosen]
                        )
                        controlled_correlation = pressure_controlled_correlation(
                            current_log_activity[test][chosen],
                            residual[chosen],
                            test_pressure_labels[chosen],
                        )
                        row = {
                            **metadata,
                            "model": model,
                            "test_month": str(np.datetime64(int(month_value), "M")),
                            "observations": int(chosen.sum()),
                            "bar_rmse": local_rmse,
                            "bar_mae": local_mae,
                            "residual_current_activity_correlation": activity_correlation,
                            "pressure_controlled_residual_activity_correlation": controlled_correlation,
                            "residual_lag_impulse_correlation": safe_correlation(
                                residual[chosen], lag_impulse[test][chosen]
                            ),
                        }
                        monthly_rows.append(row)
                        local_month_rows.append(row)
                    evaluation_rows.append(
                        {
                            **metadata,
                            "model": model,
                            "observations": int(test.sum()),
                            "bar_rmse": bar_rmse,
                            "bar_mae": bar_mae,
                            "mean_monthly_rmse": float(
                                np.mean([row["bar_rmse"] for row in local_month_rows])
                            ),
                            "mean_monthly_mae": float(
                                np.mean([row["bar_mae"] for row in local_month_rows])
                            ),
                            "mean_monthly_residual_activity_correlation": float(
                                np.nanmean(
                                    [
                                        row["residual_current_activity_correlation"]
                                        for row in local_month_rows
                                    ]
                                )
                            ),
                            "mean_monthly_pressure_controlled_activity_correlation": float(
                                np.nanmean(
                                    [
                                        row[
                                            "pressure_controlled_residual_activity_correlation"
                                        ]
                                        for row in local_month_rows
                                    ]
                                )
                            ),
                            "mean_monthly_residual_lag_impulse_correlation": float(
                                np.nanmean(
                                    [
                                        row["residual_lag_impulse_correlation"]
                                        for row in local_month_rows
                                    ]
                                )
                            ),
                        }
                    )

                lag_cuts = np.unique(
                    np.quantile(
                        lag_impulse[train],
                        np.linspace(0, 1, config["lag_impulse_bins"] + 1)[1:-1],
                    )
                )
                activity_cuts = np.unique(
                    np.quantile(
                        previous_log_activity[train],
                        np.linspace(
                            0, 1, config["previous_activity_bands"] + 1
                        )[1:-1],
                    )
                )
                lag_labels = np.digitize(lag_impulse[test], lag_cuts, right=True)
                activity_labels = np.digitize(
                    previous_log_activity[test], activity_cuts, right=True
                )
                curve_residuals = {
                    model: observed[test] - prediction
                    for model, prediction in predictions.items()
                    if model in {"frozen_pb", "lag_impact", "lag_activity", "lag_activity_duration"}
                }
                for activity_label in range(len(activity_cuts) + 1):
                    for lag_label in range(len(lag_cuts) + 1):
                        chosen = (
                            (activity_labels == activity_label)
                            & (lag_labels == lag_label)
                        )
                        if chosen.sum() < config["minimum_bin_observations"]:
                            continue
                        relation = np.where(
                            side[test][chosen] * previous_side[test][chosen] > 0,
                            "same",
                            "opposite",
                        )
                        row = {
                            **metadata,
                            "previous_activity_band": activity_label + 1,
                            "lag_impulse_bin": lag_label + 1,
                            "observations": int(chosen.sum()),
                            "mean_lag_impulse": float(np.mean(lag_impulse[test][chosen])),
                            "mean_previous_log_activity": float(
                                np.mean(previous_log_activity[test][chosen])
                            ),
                            "same_sign_fraction": float(np.mean(relation == "same")),
                        }
                        for model, residual in curve_residuals.items():
                            row[f"mean_residual_{model}"] = float(
                                np.mean(residual[chosen])
                            )
                        curve_rows.append(row)
        del frame

    evaluation = pl.DataFrame(evaluation_rows)
    monthly = pl.DataFrame(monthly_rows)
    coefficients = pl.DataFrame(coefficient_rows)
    curves = pl.DataFrame(curve_rows)
    ledger = pl.DataFrame(ledger_rows)
    evaluation.write_csv(output / "oos_evaluation.csv")
    monthly.write_csv(output / "monthly_evaluation.csv")
    coefficients.write_csv(output / "lag_coefficients.csv")
    curves.write_csv(output / "lag_diagnostic_curves.csv")
    ledger.write_csv(output / "calibration_ledger.csv")
    manifest = {
        "experiment_id": config["experiment_id"],
        "config": str(args.config.resolve()),
        "base_scaling_config": str(base_config_path.resolve()),
        "base_scaling_results": str(base_results.resolve()),
        "cache_root": str(cache_root.resolve()),
        "calibration_sequence": [
            "Load the chronological pb_activity fit for the sample and fold.",
            "Freeze every P&B shape, scale, side and activity parameter.",
            "Calculate training-bar residuals from that frozen prediction.",
            "Fit only the linear lag-layer coefficients on those training residuals, with equal total weight per calendar month.",
            "Freeze the lag coefficients and training duration centre.",
            "Apply both frozen layers unchanged from test_start through test_end.",
        ],
        "outputs": [
            "calibration_ledger.csv",
            "lag_coefficients.csv",
            "oos_evaluation.csv",
            "monthly_evaluation.csv",
            "lag_diagnostic_curves.csv",
        ],
        "holdout_months_excluded": config["holdout_months_excluded"],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
