"""Estimate dense event-response paths by realized execution-price sweep.

Version 2 deliberately treats the side-aligned first-to-last execution-price
displacement as the primary measure of event aggression. Fill count and price
level count remain in the output only as secondary structural diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))

from alpha_search.decay import (  # noqa: E402
    SWEEP_CLASS_ORDER,
    event_response_components,
    execution_span_bps,
    response_at_horizon,
    sweep_class,
)


HORIZONS_MS = (
    100,
    250,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    15_000,
    30_000,
    60_000,
    120_000,
    300_000,
    600_000,
    900_000,
    1_800_000,
)
SWEEP_QUANTILES = (0.25, 0.50, 0.75)
ZERO_TOLERANCE_BPS = 1e-12


def month_from_path(path: Path) -> str:
    return f"{path.parents[1].name.split('=')[1]}-{path.parent.name.split('=')[1]}"


def validated_event_paths(
    shared: Path,
    symbol: str,
    *,
    end_month: str | None = None,
) -> list[Path]:
    reports = sorted((shared / "metadata" / symbol).glob("????-??.json"))
    paths = []
    for report in reports:
        if end_month is not None and report.stem > end_month:
            continue
        year, month = report.stem.split("-")
        path = (
            shared
            / "events"
            / "timestamp_direction_v1"
            / f"symbol={symbol}"
            / f"year={year}"
            / f"month={month}"
            / "part-000.parquet"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(path)
    if not paths:
        raise FileNotFoundError("No validated event files were found.")
    return paths


def sweep_expression() -> pl.Expr:
    return (
        10_000.0
        * pl.col("aggressor_sign").cast(pl.Float64)
        * (pl.col("last_price") / pl.col("first_price")).log()
    )


def span_expression() -> pl.Expr:
    return 10_000.0 * (pl.col("max_price") / pl.col("min_price")).log()


def sweep_class_expression(cuts: np.ndarray) -> pl.Expr:
    sweep = pl.col("endpoint_sweep_bps")
    return (
        pl.when(sweep < -ZERO_TOLERANCE_BPS)
        .then(pl.lit("opposite_endpoint_move"))
        .when(sweep.abs() <= ZERO_TOLERANCE_BPS)
        .then(pl.lit("no_endpoint_sweep"))
        .when(sweep <= float(cuts[0]))
        .then(pl.lit("sweep_q1"))
        .when(sweep <= float(cuts[1]))
        .then(pl.lit("sweep_q2"))
        .when(sweep <= float(cuts[2]))
        .then(pl.lit("sweep_q3"))
        .otherwise(pl.lit("sweep_q4"))
    )


def calibrate_sweep_cuts(
    paths: list[Path],
    *,
    calibration_end_month: str,
    sample_modulus: int,
) -> tuple[np.ndarray, int]:
    calibration_paths = [
        path for path in paths if month_from_path(path) <= calibration_end_month
    ]
    if not calibration_paths:
        raise ValueError("No event partitions fall inside the calibration period.")
    sample = (
        pl.scan_parquet(calibration_paths)
        .filter((pl.col("first_trade_id") % sample_modulus) == 0)
        .select(sweep_expression().alias("endpoint_sweep_bps"))
        .filter(pl.col("endpoint_sweep_bps") > ZERO_TOLERANCE_BPS)
        .collect(engine="streaming")
    )
    values = sample["endpoint_sweep_bps"].to_numpy()
    if len(values) < 100:
        raise ValueError("Too few positive-sweep calibration events.")
    cuts = np.quantile(values, SWEEP_QUANTILES)
    if not np.all(np.diff(cuts) > 0):
        raise ValueError(
            "Training sweep quantiles are not distinct; use a denser calibration sample."
        )
    return cuts, len(values)


def category_masks(classes: np.ndarray, span_bps: np.ndarray) -> dict[str, np.ndarray]:
    positive = np.isin(classes, ["sweep_q1", "sweep_q2", "sweep_q3", "sweep_q4"])
    no_sweep = classes == "no_endpoint_sweep"
    return {
        "all_events": np.ones(len(classes), dtype=bool),
        "positive_sweep_all": positive,
        **{name: classes == name for name in SWEEP_CLASS_ORDER},
        "path_excursion_returned_to_start": no_sweep & (span_bps > ZERO_TOLERANCE_BPS),
    }


def append_daily_rows(
    rows: list[dict],
    *,
    month: str,
    horizon_ms: int,
    category: str,
    side_label: str,
    selected: np.ndarray,
    day: np.ndarray,
    quantity: np.ndarray,
    fill_count: np.ndarray,
    price_level_count: np.ndarray,
    endpoint_sweep_bps: np.ndarray,
    execution_span_bps: np.ndarray,
    entry_displacement: np.ndarray,
    immediate_response: np.ndarray,
    total_response: np.ndarray,
    post_event_response: np.ndarray,
) -> None:
    valid = selected & np.isfinite(total_response) & np.isfinite(post_event_response)
    if not valid.any():
        return
    for day_value in np.unique(day[valid]):
        chosen = valid & (day == day_value)
        rows.append(
            {
                "source_month": month,
                "utc_day": int(day_value),
                "horizon_ms": horizon_ms,
                "horizon_seconds": horizon_ms / 1_000.0,
                "category": category,
                "side": side_label,
                "sample_events": int(chosen.sum()),
                "mean_quantity": float(quantity[chosen].mean()),
                "mean_fill_count": float(fill_count[chosen].mean()),
                "mean_price_level_count": float(price_level_count[chosen].mean()),
                "mean_endpoint_sweep_bps": float(endpoint_sweep_bps[chosen].mean()),
                "mean_execution_span_bps": float(execution_span_bps[chosen].mean()),
                "mean_entry_displacement_bps": float(entry_displacement[chosen].mean()),
                "mean_immediate_response_bps": float(immediate_response[chosen].mean()),
                "mean_total_response_bps": float(total_response[chosen].mean()),
                "mean_post_event_response_bps": float(post_event_response[chosen].mean()),
            }
        )


def exact_category_counts(paths: list[Path], cuts: np.ndarray) -> pl.DataFrame:
    events = (
        pl.scan_parquet(paths)
        .with_columns(
            sweep_expression().alias("endpoint_sweep_bps"),
            span_expression().alias("execution_span_bps"),
        )
        .with_columns(
            sweep_class_expression(cuts).alias("category"),
            (
                (pl.col("first_price") == pl.col("last_price"))
                & (pl.col("execution_span_bps") > ZERO_TOLERANCE_BPS)
            ).alias("path_excursion_returned_to_start"),
        )
    )
    result = (
        events.group_by("category", "aggressor_sign")
        .agg(
            pl.len().alias("events"),
            pl.col("quantity").sum().alias("total_quantity"),
            pl.col("quantity").mean().alias("mean_quantity"),
            pl.col("endpoint_sweep_bps").mean().alias("mean_endpoint_sweep_bps"),
            pl.col("execution_span_bps").mean().alias("mean_execution_span_bps"),
            pl.col("fill_count").mean().alias("mean_fill_count"),
            pl.col("price_level_count").mean().alias("mean_price_level_count"),
            pl.col("path_excursion_returned_to_start").sum().alias(
                "path_excursion_events"
            ),
        )
        .collect(engine="streaming")
    )
    total_events = int(result["events"].sum())
    total_quantity = float(result["total_quantity"].sum())
    return result.with_columns(
        (pl.col("events") / total_events).alias("event_share"),
        (pl.col("total_quantity") / total_quantity).alias("quantity_share"),
    )


def analyse_month(
    path: Path,
    next_path: Path | None,
    *,
    sample_modulus: int,
    horizons_ms: tuple[int, ...],
    sweep_cuts_bps: np.ndarray,
) -> tuple[list[dict], dict]:
    month = month_from_path(path)
    market = (
        pl.scan_parquet(path)
        .select("timestamp_ms", "last_price")
        .collect()
        .rechunk()
    )
    current_time = market["timestamp_ms"].to_numpy()
    current_last = market["last_price"].to_numpy()
    maximum_horizon_ms = max(horizons_ms)
    if next_path is not None:
        next_prefix = (
            pl.scan_parquet(next_path)
            .filter(pl.col("timestamp_ms") <= int(current_time[-1]) + maximum_horizon_ms)
            .select("timestamp_ms", "last_price")
            .collect()
        )
        market_time = np.concatenate(
            [current_time, next_prefix["timestamp_ms"].to_numpy()]
        )
        market_last = np.concatenate(
            [current_last, next_prefix["last_price"].to_numpy()]
        )
    else:
        market_time = current_time
        market_last = current_last

    anchors = (
        pl.scan_parquet(path)
        .with_row_index("event_index")
        .filter((pl.col("first_trade_id") % sample_modulus) == 0)
        .select(
            "event_index",
            "timestamp_ms",
            "aggressor_sign",
            "quantity",
            "first_price",
            "last_price",
            "min_price",
            "max_price",
            "fill_count",
            "price_level_count",
        )
        .collect()
        .filter(pl.col("event_index") > 0)
        .rechunk()
    )
    index = anchors["event_index"].to_numpy().astype(np.int64)
    anchor_time = anchors["timestamp_ms"].to_numpy()
    side = anchors["aggressor_sign"].to_numpy().astype(np.int8)
    quantity = anchors["quantity"].to_numpy()
    first = anchors["first_price"].to_numpy()
    last = anchors["last_price"].to_numpy()
    minimum = anchors["min_price"].to_numpy()
    maximum = anchors["max_price"].to_numpy()
    fills = anchors["fill_count"].to_numpy()
    levels = anchors["price_level_count"].to_numpy()
    previous = current_last[index - 1]
    if not np.allclose(last, current_last[index], rtol=0.0, atol=0.0):
        raise AssertionError("Sample event indexes do not align with the market-price array.")

    entry, endpoint_sweep, immediate = event_response_components(
        previous, first, last, side
    )
    span = execution_span_bps(minimum, maximum)
    classes = sweep_class(endpoint_sweep, sweep_cuts_bps)
    masks = category_masks(classes, span)
    day = anchor_time // 86_400_000
    rows: list[dict] = []
    for horizon_ms in horizons_ms:
        future_index = np.searchsorted(
            market_time, anchor_time + horizon_ms, side="left"
        )
        valid = future_index < len(market_time)
        future = np.full(len(anchors), np.nan)
        future[valid] = market_last[future_index[valid]]
        total = np.full(len(anchors), np.nan)
        post = np.full(len(anchors), np.nan)
        total[valid], post[valid] = response_at_horizon(
            previous[valid], last[valid], future[valid], side[valid]
        )
        for category, category_mask in masks.items():
            for side_value, side_label in ((0, "all"), (-1, "sell"), (1, "buy")):
                selected = category_mask if side_value == 0 else (
                    category_mask & (side == side_value)
                )
                append_daily_rows(
                    rows,
                    month=month,
                    horizon_ms=horizon_ms,
                    category=category,
                    side_label=side_label,
                    selected=selected,
                    day=day,
                    quantity=quantity,
                    fill_count=fills,
                    price_level_count=levels,
                    endpoint_sweep_bps=endpoint_sweep,
                    execution_span_bps=span,
                    entry_displacement=entry,
                    immediate_response=immediate,
                    total_response=total,
                    post_event_response=post,
                )
    sample_summary = {
        "source_month": month,
        "source_events": len(current_time),
        "sample_modulus": sample_modulus,
        "sample_events": len(anchors),
        "sample_share": len(anchors) / len(current_time),
        "positive_sweep_events": int(np.isin(classes, ["sweep_q1", "sweep_q2", "sweep_q3", "sweep_q4"]).sum()),
        "no_endpoint_sweep_events": int((classes == "no_endpoint_sweep").sum()),
        "opposite_endpoint_move_events": int((classes == "opposite_endpoint_move").sum()),
        "path_excursion_returned_to_start_events": int(
            ((classes == "no_endpoint_sweep") & (span > ZERO_TOLERANCE_BPS)).sum()
        ),
    }
    return rows, sample_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--shared-data-root",
        type=Path,
        default=REPO / "shared-data",
        help="Portable shared-data root; may be on an external SSD.",
    )
    parser.add_argument("--sample-modulus", type=int, default=500)
    parser.add_argument("--calibration-end-month", default="2022-12")
    parser.add_argument(
        "--end-month",
        default="2026-05",
        help="Last analysed month. The default preserves Jun-Aug 2026 as holdout.",
    )
    parser.add_argument(
        "--horizons-ms",
        nargs="+",
        type=int,
        default=HORIZONS_MS,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "alpha-search" / "reports" / "event_decay_sweep_v2",
    )
    args = parser.parse_args()
    if args.sample_modulus <= 0:
        raise ValueError("sample-modulus must be positive.")
    horizons_ms = tuple(sorted(set(args.horizons_ms)))
    if not horizons_ms or any(value <= 0 for value in horizons_ms):
        raise ValueError("horizons-ms must contain positive integers.")
    if args.calibration_end_month > args.end_month:
        raise ValueError("calibration-end-month cannot follow end-month.")

    paths = validated_event_paths(
        args.shared_data_root.expanduser().resolve(),
        args.symbol,
        end_month=args.end_month,
    )
    print("Calibrating positive endpoint-sweep cut points", flush=True)
    cuts, calibration_events = calibrate_sweep_cuts(
        paths,
        calibration_end_month=args.calibration_end_month,
        sample_modulus=args.sample_modulus,
    )
    print(
        "Sweep cuts (bps): " + ", ".join(f"{value:.8f}" for value in cuts),
        flush=True,
    )
    print("Calculating exact sweep-class counts", flush=True)
    counts = exact_category_counts(paths, cuts)
    daily_rows: list[dict] = []
    sample_rows: list[dict] = []
    for position, path in enumerate(paths):
        next_path = paths[position + 1] if position + 1 < len(paths) else None
        rows, sample = analyse_month(
            path,
            next_path,
            sample_modulus=args.sample_modulus,
            horizons_ms=horizons_ms,
            sweep_cuts_bps=cuts,
        )
        daily_rows.extend(rows)
        sample_rows.append(sample)
        print(
            f"{sample['source_month']}: {sample['sample_events']:,} sampled events",
            flush=True,
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    cut_table = pl.DataFrame(
        {
            "boundary": ["q25_q50", "q50_q75", "q75_q100"],
            "training_quantile": list(SWEEP_QUANTILES),
            "endpoint_sweep_bps": cuts,
        }
    )
    tables = {
        "sweep_quantile_cuts.csv": cut_table,
        "event_sweep_class_counts.csv": counts,
        "decay_sample_summary.csv": pl.DataFrame(sample_rows),
        "decay_daily.csv": pl.DataFrame(daily_rows),
    }
    for filename, table in tables.items():
        destination = args.output_root / filename
        table.write_csv(destination)
        print(destination, flush=True)
    manifest = {
        "version": "event_decay_sweep_v2",
        "symbol": args.symbol,
        "event_rule": "timestamp_direction_v1",
        "first_month": month_from_path(paths[0]),
        "last_month": month_from_path(paths[-1]),
        "calibration_end_month": args.calibration_end_month,
        "calibration_positive_sweep_events": calibration_events,
        "sample_modulus": args.sample_modulus,
        "horizons_ms": list(horizons_ms),
        "sweep_quantiles": list(SWEEP_QUANTILES),
        "sweep_cuts_bps": cuts.tolist(),
        "primary_classification": "side_aligned_first_to_last_execution_price_displacement",
        "secondary_diagnostics": ["quantity", "fill_count", "price_level_count", "min_max_execution_span"],
        "holdout_note": "Jun-Aug 2026 excluded by default and not inspected.",
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
