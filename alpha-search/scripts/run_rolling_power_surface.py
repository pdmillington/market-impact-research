"""Monthly rolling calibration and ageing diagnostics for the power surface."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))

from alpha_search.impact_surface import (  # noqa: E402
    fit_power_surface,
    predict_surface,
    surface_cells,
)
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


def _cell_rmse(
    z: np.ndarray,
    activity: np.ndarray,
    impact: np.ndarray,
    prediction: np.ndarray,
    training_cells,
    n_z: int,
    n_a: int,
) -> tuple[float, int, float]:
    residual = impact - prediction
    cells = surface_cells(
        z,
        activity,
        residual,
        imbalance_bins=n_z,
        activity_bins=n_a,
        imbalance_cuts=training_cells.imbalance_cuts,
        activity_cuts=training_cells.activity_cuts,
    )
    return (
        float(np.sqrt(np.mean(cells.mean_impact**2))),
        len(cells.mean_impact),
        float(np.max(np.abs(cells.mean_impact))),
    )


def analyse_construction(
    construction: str,
    path: Path,
    config: dict,
) -> tuple[list[dict], list[dict]]:
    print(f"Loading {construction}", flush=True)
    frame = (
        pl.scan_parquet(path)
        .sort("end_time_ms")
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
    n_z = int(config["surface"]["imbalance_bins"])
    n_a = int(config["surface"]["activity_bins"])
    first_time = int(time[0])
    final_exclusive = utc_ms("2026-06-01")
    fit_rows: list[dict] = []
    calibration_rows: list[dict] = []

    for calibration_date in month_starts(
        config["calibration_start"], config["calibration_end"]
    ):
        calibration_ms = utc_ms(calibration_date)
        if calibration_ms > int(time[-1]):
            break
        for window_value in config["training_windows_months"]:
            if window_value == "expanding":
                train_start_ms = first_time
                window_label = "expanding"
                window_months = None
            else:
                window_months = int(window_value)
                train_start = f"{add_months(calibration_date[:7], -window_months)}-01"
                train_start_ms = utc_ms(train_start)
                window_label = f"{window_months}m"
                if train_start_ms < first_time:
                    continue
            train_left = int(np.searchsorted(time, train_start_ms, side="left"))
            train_right = int(np.searchsorted(time, calibration_ms, side="left"))
            if train_right <= train_left:
                continue
            for side_value, side_name in [(-1, "sell"), (1, "buy")]:
                local = finite[train_left:train_right] & (
                    side[train_left:train_right] == side_value
                )
                if local.sum() < n_z * n_a * 3:
                    continue
                training_cells = surface_cells(
                    z[train_left:train_right][local],
                    activity[train_left:train_right][local],
                    impact[train_left:train_right][local],
                    imbalance_bins=n_z,
                    activity_bins=n_a,
                )
                fit = fit_power_surface(training_cells)
                fit_rows.append(
                    {
                        "construction": construction,
                        "calibration_date": calibration_date,
                        "training_window": window_label,
                        "training_window_months": window_months,
                        "side": side_name,
                        "train_start_ms": train_start_ms,
                        "train_end_ms": calibration_ms,
                        "training_observations": int(local.sum()),
                        "training_cells": fit.cells,
                        "amplitude": fit.amplitude,
                        "imbalance_exponent_p": fit.imbalance_exponent,
                        "activity_exponent_q": fit.activity_exponent,
                        "collapse_exponent_gamma": fit.collapse_exponent,
                        "train_cell_rmse": fit.cell_rmse,
                    }
                )
                for age in range(1, int(config["maximum_calibration_age_months"]) + 1):
                    test_start = f"{add_months(calibration_date[:7], age - 1)}-01"
                    test_end = f"{add_months(calibration_date[:7], age)}-01"
                    test_start_ms = utc_ms(test_start)
                    test_end_ms = utc_ms(test_end)
                    if test_end_ms > final_exclusive:
                        continue
                    left = int(np.searchsorted(time, test_start_ms, side="left"))
                    right = int(np.searchsorted(time, test_end_ms, side="left"))
                    selected = finite[left:right] & (side[left:right] == side_value)
                    if selected.sum() < n_z * n_a * 3:
                        continue
                    test_z = z[left:right][selected]
                    test_activity = activity[left:right][selected]
                    test_impact = impact[left:right][selected]
                    prediction = predict_surface(fit, test_z, test_activity)
                    cell_rmse, cells_used, maximum_bias = _cell_rmse(
                        test_z,
                        test_activity,
                        test_impact,
                        prediction,
                        training_cells,
                        n_z,
                        n_a,
                    )
                    residual = test_impact - prediction
                    calibration_rows.append(
                        {
                            "construction": construction,
                            "calibration_date": calibration_date,
                            "training_window": window_label,
                            "training_window_months": window_months,
                            "side": side_name,
                            "calibration_age_months": age,
                            "test_month": test_start[:7],
                            "test_observations": int(selected.sum()),
                            "test_cells": cells_used,
                            "test_cell_rmse": cell_rmse,
                            "test_maximum_absolute_cell_bias": maximum_bias,
                            "test_bar_rmse": float(np.sqrt(np.mean(residual**2))),
                            "test_mean_residual": float(np.mean(residual)),
                            "test_residual_log_activity_correlation": float(
                                np.corrcoef(residual, np.log(test_activity))[0, 1]
                            ),
                        }
                    )
    return fit_rows, calibration_rows


def add_cadence_fields(calibration: pl.DataFrame, config: dict) -> pl.DataFrame:
    anchor = datetime.strptime(config["cadence_comparison_start"], "%Y-%m-%d")
    dates = [datetime.strptime(value, "%Y-%m-%d") for value in calibration["calibration_date"]]
    origin_index = np.asarray(
        [(value.year - anchor.year) * 12 + value.month - anchor.month for value in dates]
    )
    return calibration.with_columns(
        pl.Series("months_from_cadence_anchor", origin_index)
    )


def write_cadence_summary(
    calibration_frame: pl.DataFrame,
    config: dict,
    destination: Path,
) -> None:
    anchor = datetime.strptime(config["cadence_comparison_start"], "%Y-%m-%d")
    common = datetime.strptime(config["common_window_comparison_start"], "%Y-%m-%d")
    common_index = (common.year - anchor.year) * 12 + common.month - anchor.month
    cadence_rows = []
    for cadence in config["cadences_months"]:
        selected = calibration_frame.filter(
            (pl.col("months_from_cadence_anchor") >= common_index)
            & ((pl.col("months_from_cadence_anchor") - common_index) % cadence == 0)
            & (pl.col("calibration_age_months") <= cadence)
        )
        summary = selected.group_by(
            "construction", "training_window", "side"
        ).agg(
            pl.len().alias("test_side_months"),
            pl.col("test_cell_rmse").mean().alias("mean_test_cell_rmse"),
            pl.col("test_cell_rmse").median().alias("median_test_cell_rmse"),
            pl.col("test_cell_rmse").quantile(0.90).alias("p90_test_cell_rmse"),
            pl.col("test_maximum_absolute_cell_bias").median()
            .alias("median_maximum_absolute_cell_bias"),
            pl.col("test_residual_log_activity_correlation").abs().median()
            .alias("median_abs_residual_activity_correlation"),
        ).with_columns(pl.lit(cadence).alias("recalibration_cadence_months"))
        cadence_rows.append(summary)
    pl.concat(cadence_rows).sort(
        "construction", "recalibration_cadence_months", "training_window", "side"
    ).write_csv(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "alpha-search" / "configs" / "rolling_power_surface_v1.json",
    )
    parser.add_argument(
        "--shared-data-root", type=Path, default=REPO / "shared-data"
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--summarise-existing", action="store_true",
        help="Rebuild cadence_summary.csv from an existing rolling_calibration.csv.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cache_root = (
        args.shared_data_root.expanduser().resolve()
        / "features"
        / "alpha"
        / f"symbol={config['symbol']}"
        / config["cache_version"]
    )
    cache_manifest = json.loads((cache_root / "cache_manifest.json").read_text())
    if cache_manifest.get("holdout_months_excluded") != config["holdout_months_excluded"]:
        raise ValueError("Cache holdout declaration does not match the rolling config.")
    output_root = args.output_root or (
        REPO / "alpha-search" / "reports" / config["experiment_id"]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if args.summarise_existing:
        calibration_frame = pl.read_csv(output_root / "rolling_calibration.csv")
        write_cadence_summary(
            calibration_frame, config, output_root / "cadence_summary.csv"
        )
        manifest_path = output_root / "run_manifest.json"
        run_manifest = json.loads(manifest_path.read_text())
        run_manifest["config"] = config
        run_manifest["config_path"] = str(args.config.resolve())
        run_manifest["summary_refreshed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")
        print(output_root / "cadence_summary.csv", flush=True)
        return
    fits: list[dict] = []
    calibration: list[dict] = []
    for construction in config["constructions"]:
        path = cache_root / construction / "part-000.parquet"
        new_fits, new_calibration = analyse_construction(construction, path, config)
        fits.extend(new_fits)
        calibration.extend(new_calibration)
    fit_frame = pl.DataFrame(fits).sort(
        "construction", "calibration_date", "training_window", "side"
    )
    calibration_frame = add_cadence_fields(
        pl.DataFrame(calibration), config
    ).sort(
        "construction", "calibration_date", "training_window", "side",
        "calibration_age_months"
    )
    fit_frame.write_csv(output_root / "rolling_fits.csv")
    calibration_frame.write_csv(output_root / "rolling_calibration.csv")

    write_cadence_summary(
        calibration_frame, config, output_root / "cadence_summary.csv"
    )

    run_manifest = {
        "experiment_id": config["experiment_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config.resolve()),
        "config": config,
        "cache_manifest": cache_manifest,
        "status": "complete",
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n"
    )
    print(output_root, flush=True)


if __name__ == "__main__":
    main()
