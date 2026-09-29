"""Walk-forward screen of impact residuals measured on multiple event scales."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))

from alpha_search.evaluation import (  # noqa: E402
    day_block_summary,
    forward_clock_targets,
    frozen_quantiles,
    paired_day_difference,
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


def expected_impact(x: np.ndarray, side: np.ndarray, fit) -> np.ndarray:
    log_x = np.log1p(x / fit.imbalance_scale)
    result = np.full(len(x), np.nan)
    buy = side > 0
    sell = side < 0
    result[buy] = fit.buy.intercept + fit.buy.slope * log_x[buy]
    result[sell] = fit.sell.intercept + fit.sell.slope * log_x[sell]
    return result


def prefixed_summary(values: np.ndarray, day: np.ndarray, selected: np.ndarray) -> dict:
    summary = day_block_summary(values, day, selected)
    return {
        "observations": summary["observations"],
        "days": summary["days"],
        "mean_aligned_return_bps": summary["mean_return_bps"],
        "day_mean_aligned_return_bps": summary["day_mean_bps"],
        "day_standard_error_bps": summary["day_standard_error_bps"],
        "day_t_stat": summary["day_t_stat"],
        "positive_day_share": summary["positive_day_share"],
    }


def curve_bins(
    x: np.ndarray,
    y: np.ndarray,
    expected: np.ndarray,
    selected: np.ndarray,
    *,
    bins: int = 20,
) -> list[dict[str, float | int]]:
    mask = selected & np.isfinite(x) & np.isfinite(y) & np.isfinite(expected)
    if mask.sum() < bins:
        return []
    cuts = np.unique(np.quantile(x[mask], np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return []
    labels = np.digitize(x[mask], cuts[1:-1], right=True)
    rows = []
    for label in range(len(cuts) - 1):
        chosen = labels == label
        if not chosen.any():
            continue
        rows.append(
            {
                "impact_bin": label + 1,
                "observations": int(chosen.sum()),
                "mean_normalised_imbalance": float(x[mask][chosen].mean()),
                "mean_normalised_signed_impact": float(y[mask][chosen].mean()),
                "mean_expected_normalised_impact": float(expected[mask][chosen].mean()),
            }
        )
    return rows


def analyse_size(
    *,
    size: int,
    cache_path: Path,
    folds: list[dict],
    requested_windows: list[int | str],
    horizons: list[int],
    quantiles: int,
) -> dict[str, list[dict]]:
    windows = resolve_measurement_windows(size, requested_windows)
    print(f"Loading decision clock event_{size}; windows={windows}", flush=True)
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
    observations: dict[int, pl.DataFrame] = {}
    for window in windows:
        observations[window] = (
            build_trailing_impact_observations(
                base, events_per_bar=size, measurement_events=window
            )
            .select(
                "bar_number",
                "measurement_start_time_ms",
                "end_time_ms",
                "end_price",
                "signed_imbalance",
                "normalised_imbalance",
                "normalised_signed_impact",
                "trailing_market_volume",
                "trailing_realised_volatility_bps",
                "measurement_duration_seconds",
                "measurement_side",
                "decision_side",
                "side_agreement",
                "is_non_overlapping_measurement",
            )
            .collect()
            .rechunk()
        )

    reference = observations[windows[0]]
    time = reference["end_time_ms"].to_numpy()
    price = reference["end_price"].to_numpy()
    day = time // 86_400_000
    target_lookup = forward_clock_targets(time, price, horizons)
    output: dict[str, list[dict]] = {
        "fold_impact_fits": [],
        "fold_impact_curve_bins": [],
        "fold_residual_cuts": [],
        "fold_quintile_summary": [],
        "fold_extreme_contrast": [],
        "fold_scale_correlation": [],
    }

    for fold in folds:
        fold_id = int(fold["fold"])
        train_start = utc_ms(fold["train_start"])
        train_end = utc_ms(fold["train_end"]) - int(fold["embargo_days"]) * 86_400_000
        test_start = utc_ms(fold["test_start"])
        test_end = utc_ms(fold["test_end"])
        residuals: dict[int, np.ndarray] = {}
        sides: dict[int, np.ndarray] = {}

        for window, frame in observations.items():
            x = frame["normalised_imbalance"].to_numpy()
            y = frame["normalised_signed_impact"].to_numpy()
            side = frame["measurement_side"].fill_null(0).to_numpy().astype(np.int8)
            measurement_start = frame["measurement_start_time_ms"].to_numpy()
            calibration_endpoint = frame[
                "is_non_overlapping_measurement"
            ].to_numpy()
            valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (side != 0)
            train_mask = valid & (time >= train_start) & (time < train_end)
            test_mask = valid & (time >= test_start) & (time < test_end)
            calibration_mask = (
                train_mask
                & calibration_endpoint
                & (measurement_start >= train_start)
            )
            if calibration_mask.sum() < 1_000 or test_mask.sum() < 100:
                raise ValueError(
                    f"Fold {fold_id}, event_{size}, window {window} has insufficient "
                    "observations: "
                    f"calibration={calibration_mask.sum()}, "
                    f"rolling_train={train_mask.sum()}, test={test_mask.sum()}."
                )
            fit = fit_asymmetric_log_impact(
                frame.filter(pl.Series(calibration_mask))
            )
            expected = expected_impact(x, side, fit)
            residual = y - expected
            residuals[window] = residual
            sides[window] = side
            duration = frame["measurement_duration_seconds"].to_numpy() / 60.0
            agreement = frame["side_agreement"].fill_null(False).to_numpy()

            for side_value, side_label, fitted in (
                (-1, "sell", fit.sell),
                (1, "buy", fit.buy),
            ):
                side_calibration = calibration_mask & (side == side_value)
                side_train = train_mask & (side == side_value)
                side_test = test_mask & (side == side_value)
                side_x = x[side_calibration]
                output["fold_impact_fits"].append(
                    {
                        "construction": f"event_{size}",
                        "decision_events": size,
                        "measurement_events": window,
                        "fold": fold_id,
                        "side": side_label,
                        "observations": fitted.observations,
                        "rolling_training_observations": int(side_train.sum()),
                        "calibration_stride_decision_bars": window // size,
                        "calibration_sampling": "non_overlapping_measurement_blocks",
                        "training_x_min": float(np.min(side_x)),
                        "training_x_p01": float(np.quantile(side_x, 0.01)),
                        "training_x_p05": float(np.quantile(side_x, 0.05)),
                        "training_x_p50": float(np.quantile(side_x, 0.50)),
                        "training_x_p95": float(np.quantile(side_x, 0.95)),
                        "training_x_p99": float(np.quantile(side_x, 0.99)),
                        "training_x_p999": float(np.quantile(side_x, 0.999)),
                        "training_x_max": float(np.max(side_x)),
                        "imbalance_scale": fit.imbalance_scale,
                        "intercept": fitted.intercept,
                        "slope": fitted.slope,
                        "r_squared": fitted.r_squared,
                        "training_median_duration_minutes": float(
                            np.nanmedian(duration[side_calibration])
                        ),
                        "test_side_agreement_rate": float(agreement[side_test].mean()),
                    }
                )
                for row in curve_bins(x, y, expected, side_calibration):
                    output["fold_impact_curve_bins"].append(
                        {
                            "construction": f"event_{size}",
                            "decision_events": size,
                            "measurement_events": window,
                            "fold": fold_id,
                            "side": side_label,
                            **row,
                        }
                    )
            labels, cuts = frozen_quantiles(
                residual, train_mask, side, n_quantiles=quantiles
            )
            for side_value, values in cuts.items():
                for cut_number, value in enumerate(values, start=1):
                    output["fold_residual_cuts"].append(
                        {
                            "construction": f"event_{size}",
                            "decision_events": size,
                            "measurement_events": window,
                            "fold": fold_id,
                            "side": "buy" if side_value == 1 else "sell",
                            "cut_number": cut_number,
                            "cut_value": float(value),
                        }
                    )

            for horizon in horizons:
                future, _ = target_lookup[horizon]
                aligned = side * future
                for side_value, side_label in ((-1, "sell"), (1, "buy")):
                    eligible = test_mask & (side == side_value) & np.isfinite(future)
                    for q in range(1, quantiles + 1):
                        selected = eligible & (labels == q)
                        summary = prefixed_summary(aligned, day, selected)
                        output["fold_quintile_summary"].append(
                            {
                                "construction": f"event_{size}",
                                "decision_events": size,
                                "measurement_events": window,
                                "fold": fold_id,
                                "test_start": fold["test_start"],
                                "test_end": fold["test_end"],
                                "horizon_minutes": horizon,
                                "side": side_label,
                                "residual_quantile": q,
                                "signal_rate": int(selected.sum()) / int(eligible.sum()),
                                **summary,
                            }
                        )
                    result = paired_day_difference(
                        aligned,
                        day,
                        eligible & (labels == quantiles),
                        eligible & (labels == 1),
                    )
                    output["fold_extreme_contrast"].append(
                        {
                            "construction": f"event_{size}",
                            "decision_events": size,
                            "measurement_events": window,
                            "fold": fold_id,
                            "test_start": fold["test_start"],
                            "test_end": fold["test_end"],
                            "horizon_minutes": horizon,
                            "side": side_label,
                            "comparison": "q5_minus_q1_aligned_return",
                            **result,
                        }
                    )

        for index, first in enumerate(windows):
            for second in windows[index + 1 :]:
                first_side, second_side = sides[first], sides[second]
                base_test = (time >= test_start) & (time < test_end)
                for side_value, side_label in ((0, "all"), (-1, "sell"), (1, "buy")):
                    selected = (
                        base_test
                        & np.isfinite(residuals[first])
                        & np.isfinite(residuals[second])
                    )
                    if side_value:
                        selected &= (first_side == side_value) & (second_side == side_value)
                    count = int(selected.sum())
                    correlation = (
                        float(np.corrcoef(residuals[first][selected], residuals[second][selected])[0, 1])
                        if count > 1
                        else math.nan
                    )
                    output["fold_scale_correlation"].append(
                        {
                            "construction": f"event_{size}",
                            "decision_events": size,
                            "fold": fold_id,
                            "side": side_label,
                            "first_measurement_events": first,
                            "second_measurement_events": second,
                            "observations": count,
                            "pearson_correlation": correlation,
                        }
                    )
        print(f"  fold {fold_id} complete", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
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

    sizes = [int(value) for value in config["event_bar_sizes"]]
    horizons = [int(value) for value in config["clock_horizons_minutes"]]
    windows = list(config["impact_residual_event_windows"])
    quantiles = int(config.get("quantiles", 5))
    combined: dict[str, list[dict]] = {}
    for size in sizes:
        result = analyse_size(
            size=size,
            cache_path=cache_root / f"event_{size}" / "part-000.parquet",
            folds=manifest["folds"],
            requested_windows=windows,
            horizons=horizons,
            quantiles=quantiles,
        )
        for name, rows in result.items():
            combined.setdefault(name, []).extend(rows)

    args.output_root.mkdir(parents=True, exist_ok=True)
    tables = {name: pl.DataFrame(rows) for name, rows in combined.items()}
    contrasts = tables["fold_extreme_contrast"]
    tables["aggregate_extreme_contrast"] = (
        contrasts.group_by(
            "construction", "decision_events", "measurement_events",
            "horizon_minutes", "side", "comparison"
        )
        .agg(
            pl.len().alias("folds"),
            pl.col("day_mean_difference_bps").mean().alias("mean_spread_bps"),
            pl.col("day_mean_difference_bps").median().alias("median_spread_bps"),
            pl.col("day_mean_difference_bps").min().alias("minimum_spread_bps"),
            (pl.col("day_mean_difference_bps") < 0).mean().alias("reversion_fold_share"),
        )
        .sort("reversion_fold_share", "median_spread_bps", descending=[True, False])
    )
    for name, table in tables.items():
        path = args.output_root / f"{name}.csv"
        table.write_csv(path)
        print(path, flush=True)

    run_manifest = {
        "runner": "run_multiscale_impact_residual.py",
        "source_manifest": str(args.manifest.resolve()),
        "data_fingerprint": manifest["data_fingerprint"],
        "cache_root": str(cache_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "event_bar_sizes": sizes,
        "impact_residual_event_windows": windows,
        "clock_horizons_minutes": horizons,
        "quantiles": quantiles,
        "impact_curve_calibration": calibration_sampling,
        "impact_curve_application": "rolling_at_every_decision_bar_close",
        "cache_fingerprint_verified": cache_manifest is not None,
        "folds": manifest["folds"],
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
