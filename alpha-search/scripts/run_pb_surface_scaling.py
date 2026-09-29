"""Fit nested scale-free P&B models on corrected fixed-event bars."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
from scipy.special import expit


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))

from alpha_search.impact_surface import (  # noqa: E402
    SurfaceCells,
    fit_power_surface,
    predict_surface,
    surface_cells,
)
from alpha_search.pb_scaling import (  # noqa: E402
    fit_pb_scaling,
    pb_shape,
    predict_pb_scaling,
)


def utc_ms(value: str) -> int:
    return int(
        datetime.strptime(value, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1_000
    )


def cells_as_rows(cells, *, metadata: dict) -> list[dict]:
    rows = []
    for index in range(len(cells.mean_impact)):
        rows.append(
            {
                **metadata,
                "imbalance_bin": int(cells.imbalance_bin[index]) + 1,
                "activity_bin": int(cells.activity_bin[index]) + 1,
                "observations": int(cells.observations[index]),
                "mean_relative_imbalance": float(cells.mean_imbalance[index]),
                "mean_relative_activity": float(cells.mean_activity[index]),
                "mean_normalised_impact": float(cells.mean_impact[index]),
                "impact_standard_error": float(cells.standard_error[index]),
            }
        )
    return rows


def prepare_cells(config: dict, cache_root: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows: list[dict] = []
    evolution_rows: list[dict] = []
    n_z = int(config["surface"]["imbalance_bins"])
    n_a = int(config["surface"]["activity_bins"])
    descriptive_end = utc_ms(config["descriptive_fit_end"])

    scenarios = []
    for sample in config["samples"]:
        sample_start = utc_ms(sample["start"])
        scenarios.append(
            {
                "sample": sample["name"],
                "fold": "full_fit",
                "train_start": sample_start,
                "train_end": descriptive_end,
                "test_start": None,
                "test_end": None,
            }
        )
        for fold in config["folds"]:
            scenarios.append(
                {
                    "sample": sample["name"],
                    "fold": str(fold["fold"]),
                    "train_start": sample_start,
                    "train_end": utc_ms(fold["train_end"]),
                    "test_start": utc_ms(fold["test_start"]),
                    "test_end": utc_ms(fold["test_end"]),
                }
            )

    for n_events in config["event_sizes"]:
        path = cache_root / f"event_{n_events}" / "part-000.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Loading event_{n_events}", flush=True)
        frame = (
            pl.scan_parquet(path)
            .select(
                "end_time_ms",
                "duration_seconds",
                "signed_imbalance",
                "relative_imbalance",
                "relative_activity",
                "normalised_signed_impact",
            )
            .filter(pl.col("end_time_ms") < descriptive_end)
            .collect()
            .rechunk()
        )
        evolution = (
            frame.lazy()
            .with_columns(
                pl.from_epoch("end_time_ms", time_unit="ms").dt.year().alias("year")
            )
            .group_by("year")
            .agg(
                pl.len().alias("bars"),
                pl.col("duration_seconds").median().alias("median_duration_seconds"),
                pl.col("relative_activity").median().alias("median_relative_activity"),
                pl.col("relative_imbalance").median().alias("median_relative_imbalance"),
                pl.col("normalised_signed_impact").abs().median()
                .alias("median_abs_normalised_impact"),
            )
            .sort("year")
            .collect()
            .with_columns(pl.lit(int(n_events)).alias("n_events"))
        )
        evolution_rows.extend(evolution.to_dicts())

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
        for scenario in scenarios:
            train = (
                finite
                & (time >= scenario["train_start"])
                & (time < scenario["train_end"])
            )
            for side_value, side_name in [(-1, "sell"), (1, "buy")]:
                chosen = train & (side == side_value)
                if chosen.sum() < n_z * n_a * 3:
                    continue
                training_cells = surface_cells(
                    z[chosen],
                    activity[chosen],
                    impact[chosen],
                    imbalance_bins=n_z,
                    activity_bins=n_a,
                )
                metadata = {
                    "sample": scenario["sample"],
                    "fold": scenario["fold"],
                    "split": "train",
                    "n_events": int(n_events),
                    "side": side_name,
                }
                rows.extend(cells_as_rows(training_cells, metadata=metadata))
                if scenario["test_start"] is None:
                    continue
                test = (
                    finite
                    & (time >= scenario["test_start"])
                    & (time < scenario["test_end"])
                    & (side == side_value)
                )
                if test.sum() < n_z * n_a * 3:
                    continue
                test_cells = surface_cells(
                    z[test],
                    activity[test],
                    impact[test],
                    imbalance_bins=n_z,
                    activity_bins=n_a,
                    imbalance_cuts=training_cells.imbalance_cuts,
                    activity_cuts=training_cells.activity_cuts,
                )
                metadata["split"] = "test"
                rows.extend(cells_as_rows(test_cells, metadata=metadata))
        del frame
    return pl.DataFrame(rows), pl.DataFrame(evolution_rows)


def fit_row(fit, *, sample: str, fold: str, training_cells: int) -> dict:
    return {
        "sample": sample,
        "fold": fold,
        "training_cells": training_cells,
        **asdict(fit),
        "tail_exponent": fit.tail_exponent,
    }


def evaluate_group(frame: pl.DataFrame, fit, *, split: str) -> tuple[list[dict], pl.DataFrame]:
    side = np.where(frame["side"].to_numpy() == "buy", 1, -1)
    n = frame["n_events"].to_numpy()
    z = frame["mean_relative_imbalance"].to_numpy()
    activity = frame["mean_relative_activity"].to_numpy()
    observed = frame["mean_normalised_impact"].to_numpy()
    predicted = predict_pb_scaling(fit, n, side, z, activity)
    residual = observed - predicted
    detailed = frame.with_columns(
        pl.Series("predicted_normalised_impact", predicted),
        pl.Series("residual", residual),
    )
    rows = []
    grouping = [("all", "all", np.ones(len(frame), dtype=bool))]
    for n_value in sorted(set(n)):
        grouping.append((str(int(n_value)), "all", n == n_value))
        for side_name in ("buy", "sell"):
            grouping.append(
                (
                    str(int(n_value)),
                    side_name,
                    (n == n_value) & (frame["side"].to_numpy() == side_name),
                )
            )
    for n_label, side_label, chosen in grouping:
        if not chosen.any():
            continue
        local_residual = residual[chosen]
        local_activity = activity[chosen]
        correlation = (
            float(np.corrcoef(local_residual, np.log(local_activity))[0, 1])
            if chosen.sum() > 2
            else np.nan
        )
        rows.append(
            {
                "split": split,
                "n_events": n_label,
                "side": side_label,
                "cells": int(chosen.sum()),
                "cell_rmse": float(np.sqrt(np.mean(local_residual**2))),
                "cell_mae": float(np.mean(np.abs(local_residual))),
                "mean_cell_residual": float(np.mean(local_residual)),
                "residual_log_activity_correlation": correlation,
            }
        )
    return rows, detailed


def fit_models(cells: pl.DataFrame, config: dict) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    fit_rows = []
    evaluation_rows = []
    detailed_rows = []
    scenarios = cells.select("sample", "fold").unique().sort(["sample", "fold"])
    for scenario in scenarios.iter_rows(named=True):
        sample = scenario["sample"]
        fold = scenario["fold"]
        training = cells.filter(
            (pl.col("sample") == sample)
            & (pl.col("fold") == fold)
            & (pl.col("split") == "train")
        )
        testing = cells.filter(
            (pl.col("sample") == sample)
            & (pl.col("fold") == fold)
            & (pl.col("split") == "test")
        )
        side = np.where(training["side"].to_numpy() == "buy", 1, -1)
        print(f"Fitting {sample}, fold {fold}", flush=True)
        for family in config["model_families"]:
            fit = fit_pb_scaling(
                training["n_events"].to_numpy(),
                side,
                training["mean_relative_imbalance"].to_numpy(),
                training["mean_relative_activity"].to_numpy(),
                training["mean_normalised_impact"].to_numpy(),
                family=family,
                reference_events=int(config["reference_events"]),
                n_starts=int(config["multi_starts"]),
            )
            fit_rows.append(
                fit_row(fit, sample=sample, fold=fold, training_cells=training.height)
            )
            for split, frame in [("train", training), ("test", testing)]:
                if frame.is_empty():
                    continue
                metrics, detailed = evaluate_group(frame, fit, split=split)
                for row in metrics:
                    evaluation_rows.append(
                        {"sample": sample, "fold": fold, "family": family, **row}
                    )
                if fold == "full_fit" and split == "train":
                    ratio = detailed["n_events"].to_numpy() / fit.reference_events
                    side_values = np.where(detailed["side"].to_numpy() == "buy", 1, -1)
                    q_scale = fit.q_reference * ratio**fit.q_exponent_xi
                    r0 = np.where(
                        side_values > 0, fit.r_buy_reference, fit.r_sell_reference
                    )
                    r_scale = r0 * ratio**fit.r_exponent_psi
                    activity_values = detailed[
                        "mean_relative_activity"
                    ].to_numpy()
                    activity_scale = fit.activity_reference * (
                        ratio**fit.activity_scale_exponent_eta
                    )
                    vertical_activity = (
                        activity_values / activity_scale
                    ) ** fit.activity_vertical_exponent_delta
                    scaled_pressure = (
                        detailed["mean_relative_imbalance"].to_numpy()
                        * activity_values**fit.activity_exponent_gamma
                        / q_scale
                    )
                    scaled_observed = (
                        detailed["mean_normalised_impact"].to_numpy()
                        / (r_scale * vertical_activity)
                    )
                    beta_fraction = np.clip(
                        fit.shape_beta / fit.central_exponent_p,
                        1e-8,
                        1.0 - 1e-8,
                    )
                    beta_logit = np.log(beta_fraction / (1.0 - beta_fraction))
                    log_relative_activity = np.log(
                        activity_values / activity_scale
                    )
                    local_beta = fit.central_exponent_p * expit(
                        beta_logit
                        + fit.shape_beta_activity_slope
                        * log_relative_activity
                    )
                    scaled_prediction = pb_shape(
                        scaled_pressure,
                        alpha=fit.shape_alpha,
                        beta=local_beta,
                        central_exponent=fit.central_exponent_p,
                    )
                    detailed = detailed.with_columns(
                        pl.lit(family).alias("family"),
                        pl.Series("scaled_pressure", scaled_pressure),
                        pl.Series("scaled_observed_impact", scaled_observed),
                        pl.Series("scaled_master_prediction", scaled_prediction),
                        pl.Series("scaled_local_beta", local_beta),
                        pl.Series(
                            "log_relative_activity", log_relative_activity
                        ),
                        pl.Series(
                            "scaled_residual", scaled_observed - scaled_prediction
                        ),
                    )
                    detailed_rows.extend(detailed.to_dicts())
    return (
        pl.DataFrame(fit_rows),
        pl.DataFrame(evaluation_rows),
        pl.DataFrame(detailed_rows),
    )


def frame_as_surface_cells(frame: pl.DataFrame) -> SurfaceCells:
    """Reconstitute the compact cell object needed by the existing fitter."""

    return SurfaceCells(
        imbalance_cuts=np.asarray([], dtype=float),
        activity_cuts=np.asarray([], dtype=float),
        mean_imbalance=frame["mean_relative_imbalance"].to_numpy(),
        mean_activity=frame["mean_relative_activity"].to_numpy(),
        mean_impact=frame["mean_normalised_impact"].to_numpy(),
        standard_error=frame["impact_standard_error"].to_numpy(),
        observations=frame["observations"].to_numpy(),
        imbalance_bin=frame["imbalance_bin"].to_numpy(),
        activity_bin=frame["activity_bin"].to_numpy(),
    )


def fit_independent_surface_benchmark(
    cells: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fit the current power surface independently at every scale and side."""

    fit_rows = []
    evaluation_rows = []
    scenarios = cells.select("sample", "fold").unique().sort(["sample", "fold"])
    for scenario in scenarios.iter_rows(named=True):
        sample = scenario["sample"]
        fold = scenario["fold"]
        all_residuals: dict[str, list[np.ndarray]] = {"train": [], "test": []}
        all_activities: dict[str, list[np.ndarray]] = {"train": [], "test": []}
        for n_events in sorted(cells["n_events"].unique().to_list()):
            for side_name in ("buy", "sell"):
                subset = cells.filter(
                    (pl.col("sample") == sample)
                    & (pl.col("fold") == fold)
                    & (pl.col("n_events") == n_events)
                    & (pl.col("side") == side_name)
                )
                training = subset.filter(pl.col("split") == "train")
                if training.is_empty():
                    continue
                fit = fit_power_surface(frame_as_surface_cells(training))
                fit_rows.append(
                    {
                        "sample": sample,
                        "fold": fold,
                        "n_events": n_events,
                        "side": side_name,
                        "amplitude": fit.amplitude,
                        "imbalance_exponent_p": fit.imbalance_exponent,
                        "activity_exponent_q": fit.activity_exponent,
                        "collapse_exponent_gamma": fit.collapse_exponent,
                        "train_cell_rmse": fit.cell_rmse,
                    }
                )
                for split in ("train", "test"):
                    frame = subset.filter(pl.col("split") == split)
                    if frame.is_empty():
                        continue
                    observed = frame["mean_normalised_impact"].to_numpy()
                    activity = frame["mean_relative_activity"].to_numpy()
                    predicted = predict_surface(
                        fit,
                        frame["mean_relative_imbalance"].to_numpy(),
                        activity,
                    )
                    residual = observed - predicted
                    all_residuals[split].append(residual)
                    all_activities[split].append(activity)
                    evaluation_rows.append(
                        {
                            "sample": sample,
                            "fold": fold,
                            "family": "independent_power_surface",
                            "split": split,
                            "n_events": str(n_events),
                            "side": side_name,
                            "cells": frame.height,
                            "cell_rmse": float(np.sqrt(np.mean(residual**2))),
                            "cell_mae": float(np.mean(np.abs(residual))),
                            "mean_cell_residual": float(np.mean(residual)),
                            "residual_log_activity_correlation": float(
                                np.corrcoef(residual, np.log(activity))[0, 1]
                            ),
                        }
                    )
        for split in ("train", "test"):
            if not all_residuals[split]:
                continue
            residual = np.concatenate(all_residuals[split])
            activity = np.concatenate(all_activities[split])
            evaluation_rows.append(
                {
                    "sample": sample,
                    "fold": fold,
                    "family": "independent_power_surface",
                    "split": split,
                    "n_events": "all",
                    "side": "all",
                    "cells": len(residual),
                    "cell_rmse": float(np.sqrt(np.mean(residual**2))),
                    "cell_mae": float(np.mean(np.abs(residual))),
                    "mean_cell_residual": float(np.mean(residual)),
                    "residual_log_activity_correlation": float(
                        np.corrcoef(residual, np.log(activity))[0, 1]
                    ),
                }
            )
    return pl.DataFrame(fit_rows), pl.DataFrame(evaluation_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "alpha-search" / "configs" / "pb_surface_scaling_v1.json",
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reuse-input-dir", type=Path)
    parser.add_argument("--reuse-benchmark-dir", type=Path)
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

    if args.reuse_input_dir:
        cells = pl.read_csv(
            args.reuse_input_dir / "scaling_cells.csv", infer_schema_length=None
        )
        evolution = pl.read_csv(
            args.reuse_input_dir / "market_evolution.csv", infer_schema_length=None
        )
        print(f"Reused prepared cells from {args.reuse_input_dir}", flush=True)
    else:
        cells, evolution = prepare_cells(config, cache_root)
    fits, evaluation, collapsed = fit_models(cells, config)
    if args.reuse_benchmark_dir:
        benchmark_fits = pl.read_csv(
            args.reuse_benchmark_dir / "independent_surface_fits.csv",
            infer_schema_length=None,
        )
        benchmark_evaluation = pl.read_csv(
            args.reuse_benchmark_dir / "scaling_evaluation.csv",
            infer_schema_length=None,
        ).filter(pl.col("family") == "independent_power_surface")
        print(f"Reused benchmark from {args.reuse_benchmark_dir}", flush=True)
    else:
        benchmark_fits, benchmark_evaluation = fit_independent_surface_benchmark(
            cells
        )
    evaluation = pl.concat([evaluation, benchmark_evaluation], how="vertical_relaxed")
    cells.write_csv(output / "scaling_cells.csv")
    evolution.write_csv(output / "market_evolution.csv")
    fits.write_csv(output / "scaling_fits.csv")
    benchmark_fits.write_csv(output / "independent_surface_fits.csv")
    evaluation.write_csv(output / "scaling_evaluation.csv")
    collapsed.write_csv(output / "collapsed_cells.csv")
    manifest = {
        "experiment_id": config["experiment_id"],
        "config": str(args.config.resolve()),
        "cache_root": str(cache_root.resolve()),
        "reuse_input_dir": (
            str(args.reuse_input_dir.resolve()) if args.reuse_input_dir else None
        ),
        "reuse_benchmark_dir": (
            str(args.reuse_benchmark_dir.resolve())
            if args.reuse_benchmark_dir
            else None
        ),
        "outputs": [
            "scaling_cells.csv",
            "market_evolution.csv",
            "scaling_fits.csv",
            "independent_surface_fits.csv",
            "scaling_evaluation.csv",
            "collapsed_cells.csv",
        ],
        "holdout_months_excluded": config["holdout_months_excluded"],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
