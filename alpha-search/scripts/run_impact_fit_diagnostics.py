"""Compare impact fit families without retaining all scales in memory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))

from alpha_search.impact_diagnostics import (  # noqa: E402
    calibration_metrics,
    calibration_rows,
    fit_trimmed_affine_log,
    fit_curve_families,
    frozen_tail_cuts,
    serialisable_parameters,
    tail_summary_rows,
)
from alpha_search.impact_model import fit_asymmetric_log_impact  # noqa: E402
from alpha_search.multiscale_impact import (  # noqa: E402
    build_trailing_impact_observations,
    resolve_measurement_windows,
)


def utc_ms(value: str) -> int:
    return int(
        datetime.fromisoformat(value)
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1_000
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--decision-events", type=int)
    parser.add_argument("--measurement-events", type=int)
    parser.add_argument("--allow-unverified-cache", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    config = manifest["config"]
    calibration_sampling = config.get("impact_curve_calibration")
    if calibration_sampling != "non_overlapping_measurement_blocks":
        raise ValueError(
            "This runner requires impact_curve_calibration="
            "'non_overlapping_measurement_blocks'."
        )
    shared_root = Path(manifest["shared_data_root"])
    cache_root = args.cache_root or (
        shared_root
        / "features"
        / "alpha"
        / f"symbol={manifest['symbol']}"
        / "short_event_screen_v1"
    )
    cache_manifest_path = cache_root / "cache_manifest.json"
    cache_manifest = (
        json.loads(cache_manifest_path.read_text())
        if cache_manifest_path.exists()
        else None
    )
    if cache_manifest is None and not args.allow_unverified_cache:
        raise RuntimeError(
            "The bar cache has no cache_manifest.json. Rebuild it, or use "
            "--allow-unverified-cache only for a deliberate legacy run."
        )
    if (
        cache_manifest is not None
        and cache_manifest.get("data_fingerprint") != manifest["data_fingerprint"]
    ):
        raise RuntimeError("Bar-cache and experiment data fingerprints do not match.")

    fit_rows: list[dict] = []
    bin_rows: list[dict] = []
    metric_rows: list[dict] = []
    tail_rows: list[dict] = []
    trim_rows: list[dict] = []
    sizes = [int(value) for value in config["event_bar_sizes"]]
    if args.decision_events is not None:
        if args.decision_events not in sizes:
            raise ValueError(
                f"Decision size {args.decision_events} is not configured: {sizes}."
            )
        sizes = [args.decision_events]
    if (args.decision_events is None) != (args.measurement_events is None):
        raise ValueError(
            "Supply --decision-events and --measurement-events together."
        )
    requested_windows = list(config["impact_residual_event_windows"])

    for size in sizes:
        windows = resolve_measurement_windows(size, requested_windows)
        if args.measurement_events is not None:
            if args.measurement_events not in windows:
                raise ValueError(
                    f"Measurement window {args.measurement_events} is not available "
                    f"for decision size {size}: {windows}."
                )
            windows = (args.measurement_events,)
        cache_path = cache_root / f"event_{size}" / "part-000.parquet"
        base = pl.scan_parquet(cache_path).select(
            "bar_number",
            "start_time_ms",
            "end_time_ms",
            "start_price",
            "end_price",
            "gross_volume",
            "signed_imbalance",
            "trailing_market_volume",
            "trailing_realised_volatility_bps",
        )
        for window in windows:
            print(f"event_{size}; measurement={window}", flush=True)
            frame = (
                build_trailing_impact_observations(
                    base, events_per_bar=size, measurement_events=window
                )
                .select(
                    "bar_number",
                    "measurement_start_time_ms",
                    "end_time_ms",
                    "signed_imbalance",
                    "measurement_return_bps",
                    "normalised_imbalance",
                    "normalised_signed_impact",
                    "is_non_overlapping_measurement",
                )
                .collect()
                .rechunk()
            )
            time = frame["end_time_ms"].to_numpy()
            measurement_start = frame["measurement_start_time_ms"].to_numpy()
            x = frame["normalised_imbalance"].to_numpy()
            y = frame["normalised_signed_impact"].to_numpy()
            side = frame["signed_imbalance"].fill_null(0).sign().to_numpy().astype(np.int8)
            aligned_return_bps = (
                side * frame["measurement_return_bps"].to_numpy()
            )
            calibration_endpoint = frame[
                "is_non_overlapping_measurement"
            ].to_numpy()
            valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (side != 0)

            for fold in manifest["folds"]:
                fold_id = int(fold["fold"])
                train_start = utc_ms(fold["train_start"])
                train_end = utc_ms(fold["train_end"]) - int(fold["embargo_days"]) * 86_400_000
                test_start = utc_ms(fold["test_start"])
                test_end = utc_ms(fold["test_end"])
                rolling_train_mask = valid & (time >= train_start) & (time < train_end)
                train_mask = (
                    rolling_train_mask
                    & calibration_endpoint
                    & (measurement_start >= train_start)
                )
                test_mask = (
                    valid
                    & calibration_endpoint
                    & (measurement_start >= test_start)
                    & (time < test_end)
                )
                fit = fit_asymmetric_log_impact(frame.filter(pl.Series(train_mask)))
                for side_value, side_label, fitted in (
                    (-1, "sell", fit.sell),
                    (1, "buy", fit.buy),
                ):
                    side_train = train_mask & (side == side_value)
                    side_test = test_mask & (side == side_value)
                    models, imbalance_cuts = fit_curve_families(
                        x,
                        y,
                        side_train,
                        raw_intercept=fitted.intercept,
                        raw_slope=fitted.slope,
                        raw_scale=fit.imbalance_scale,
                        bins=20,
                    )
                    tail_cuts = frozen_tail_cuts(x, side_train)
                    side_x = x[side_train]
                    common = {
                        "construction": f"event_{size}",
                        "decision_events": size,
                        "measurement_events": window,
                        "fold": fold_id,
                        "side": side_label,
                        "calibration_stride_decision_bars": window // size,
                        "calibration_sampling": calibration_sampling,
                        "training_x_min": float(np.min(side_x)),
                        "training_x_p01": float(np.quantile(side_x, 0.01)),
                        "training_x_p05": float(np.quantile(side_x, 0.05)),
                        "training_x_p50": float(np.quantile(side_x, 0.50)),
                        "training_x_p95": float(np.quantile(side_x, 0.95)),
                        "training_x_p99": float(np.quantile(side_x, 0.99)),
                        "training_x_p999": float(np.quantile(side_x, 0.999)),
                        "training_x_max": float(np.max(side_x)),
                    }
                    for model in models:
                        fit_rows.append({**common, **serialisable_parameters(model)})
                    for period, selected in (("train", side_train), ("test", side_test)):
                        rows = calibration_rows(
                            x, y, selected, imbalance_cuts, models
                        )
                        bin_rows.extend({**common, "period": period, **row} for row in rows)
                        metric_rows.extend(
                            {**common, "period": period, **row}
                            for row in calibration_metrics(rows)
                        )
                        tail_rows.extend(
                            {
                                **common,
                                "period": period,
                                **row,
                            }
                            for row in tail_summary_rows(
                                x,
                                y,
                                aligned_return_bps,
                                selected,
                                tail_cuts,
                            )
                        )
                    for upper_quantile in (0.95, 0.99, 1.0):
                        trimmed = fit_trimmed_affine_log(
                            x,
                            y,
                            side_train,
                            scale=fit.imbalance_scale,
                            upper_quantile=upper_quantile,
                        )
                        for period, selected in (
                            ("train", side_train),
                            ("test", side_test),
                        ):
                            rows = calibration_rows(
                                x,
                                y,
                                selected,
                                imbalance_cuts,
                                [trimmed],
                            )
                            metrics = calibration_metrics(rows)[0]
                            trim_rows.append(
                                {
                                    **common,
                                    "period": period,
                                    "upper_training_quantile": upper_quantile,
                                    "upper_x_cut": trimmed["upper_x_cut"],
                                    "fit_observations": trimmed["observations"],
                                    "intercept": trimmed["intercept"],
                                    "slope": trimmed["slope"],
                                    "scale": trimmed["scale"],
                                    **{
                                        key: value
                                        for key, value in metrics.items()
                                        if key != "model"
                                    },
                                }
                            )
                print(f"  fold {fold_id} complete", flush=True)
            del frame, time, x, y, side

    destination_root = args.output_root
    if args.decision_events is not None:
        destination_root = (
            destination_root
            / "fit_diagnostics"
            / f"event_{args.decision_events}_measurement_{args.measurement_events}"
        )
    destination_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fit_diagnostic_parameters.csv": pl.DataFrame(fit_rows),
        "fit_diagnostic_bins.csv": pl.DataFrame(bin_rows),
        "fit_diagnostic_metrics.csv": pl.DataFrame(metric_rows),
        "fit_diagnostic_tail_bins.csv": pl.DataFrame(tail_rows),
        "fit_diagnostic_trim_sensitivity.csv": pl.DataFrame(trim_rows),
    }
    for name, table in outputs.items():
        path = destination_root / name
        table.write_csv(path)
        print(path, flush=True)

    diagnostic_manifest = {
        "runner": "run_impact_fit_diagnostics.py",
        "source_manifest": str(args.manifest.resolve()),
        "data_fingerprint": manifest["data_fingerprint"],
        "cache_fingerprint_verified": cache_manifest is not None,
        "models": [
            "raw_log_affine",
            "binned_log_zero",
            "binned_power",
            "monotonic_bins",
        ],
        "training_imbalance_bins": 20,
        "tail_quantile_edges": [0, 0.01, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 0.99, 1],
        "trim_sensitivity_upper_quantiles": [0.95, 0.99, 1.0],
        "impact_curve_calibration": calibration_sampling,
        "impact_curve_application": "rolling_at_every_decision_bar_close",
        "notes": [
            "All fits use globally aligned, non-overlapping measurement blocks from the training period only.",
            "Test diagnostics use non-overlapping test blocks and imbalance cut points frozen from training.",
            "The fitted impact curve is applied separately to rolling measurements at every decision-bar close.",
            "The monotonic curve is a diagnostic benchmark, not a selected alpha model.",
        ],
    }
    (destination_root / "fit_diagnostic_manifest.json").write_text(
        json.dumps(diagnostic_manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
