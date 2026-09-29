"""Generic walk-forward screen for conditioners of a primary reversion rule."""

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
    normal_p_value,
    paired_day_difference,
)
from alpha_search.impact_model import (  # noqa: E402
    add_asymmetric_impact_features,
    fit_asymmetric_log_impact,
)


FITTED_FEATURES = {
    "expected_normalised_impact",
    "impact_residual",
    "marginal_impact_bps_per_btc",
}


def utc_ms(value: str) -> int:
    return int(
        datetime.fromisoformat(value)
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1_000
    )


def _fit_row(construction: str, fold: int, fit) -> list[dict]:
    rows = []
    for side_label, fitted in (("buy", fit.buy), ("sell", fit.sell)):
        rows.append(
            {
                "construction": construction,
                "fold": fold,
                "side": side_label,
                "observations": fitted.observations,
                "imbalance_scale": fit.imbalance_scale,
                "intercept": fitted.intercept,
                "slope": fitted.slope,
                "r_squared": fitted.r_squared,
            }
        )
    return rows


def analyse_size(
    *,
    size: int,
    cache_path: Path,
    folds: list[dict],
    primary_feature: str,
    conditioners: list[str],
    horizons: list[int],
    quantiles: int,
) -> dict[str, list[dict]]:
    print(f"Loading event_{size}", flush=True)
    requested = set(conditioners) | {primary_feature}
    base = pl.scan_parquet(cache_path).sort("bar_number")
    schema = set(base.collect_schema().names())
    missing = requested.difference(schema | FITTED_FEATURES)
    if missing:
        raise KeyError(f"event_{size} cache lacks requested features: {sorted(missing)}")
    selected_columns = sorted(
        {
            "end_time_ms",
            "end_price",
            "signed_imbalance",
            "normalised_imbalance",
            "normalised_signed_impact",
            "trailing_market_volume",
            "trailing_realised_volatility_bps",
            primary_feature,
            *[name for name in conditioners if name not in FITTED_FEATURES],
        }
    )
    frame = base.select(selected_columns).collect().rechunk()
    time = frame["end_time_ms"].to_numpy()
    price = frame["end_price"].to_numpy()
    side = np.sign(frame["signed_imbalance"].to_numpy()).astype(np.int8)
    day = time // 86_400_000
    target_lookup = forward_clock_targets(time, price, horizons)

    output: dict[str, list[dict]] = {
        "impact_fits": [],
        "fold_signal_summary": [],
        "fold_conditioner_increment": [],
        "fold_conditioning_grid": [],
        "fold_feature_cuts": [],
    }
    for fold in folds:
        fold_id = int(fold["fold"])
        train_start = utc_ms(fold["train_start"])
        train_end = utc_ms(fold["train_end"]) - int(fold["embargo_days"]) * 86_400_000
        test_start = utc_ms(fold["test_start"])
        test_end = utc_ms(fold["test_end"])
        train_mask = (time >= train_start) & (time < train_end) & (side != 0)
        test_mask = (time >= test_start) & (time < test_end) & (side != 0)
        if train_mask.sum() < 1_000 or test_mask.sum() < 100:
            raise ValueError(
                f"Fold {fold_id} has insufficient event_{size} observations: "
                f"train={train_mask.sum()}, test={test_mask.sum()}."
            )

        fit = fit_asymmetric_log_impact(frame.filter(pl.Series(train_mask)))
        enriched = add_asymmetric_impact_features(frame.lazy(), fit).collect()
        output["impact_fits"].extend(_fit_row(f"event_{size}", fold_id, fit))
        primary = enriched[primary_feature].to_numpy()
        primary_q, primary_cuts = frozen_quantiles(
            primary, train_mask, side, n_quantiles=quantiles
        )
        for side_value, cuts in primary_cuts.items():
            for cut_number, value in enumerate(cuts, start=1):
                output["fold_feature_cuts"].append(
                    {
                        "construction": f"event_{size}", "fold": fold_id,
                        "feature": primary_feature, "side": side_value,
                        "cut_number": cut_number, "cut_value": float(value),
                    }
                )

        for conditioner in conditioners:
            values = enriched[conditioner].to_numpy()
            conditioner_q, conditioner_cuts = frozen_quantiles(
                values, train_mask, side, n_quantiles=quantiles
            )
            for side_value, cuts in conditioner_cuts.items():
                for cut_number, value in enumerate(cuts, start=1):
                    output["fold_feature_cuts"].append(
                        {
                            "construction": f"event_{size}", "fold": fold_id,
                            "feature": conditioner, "side": side_value,
                            "cut_number": cut_number, "cut_value": float(value),
                        }
                    )
            for horizon in horizons:
                raw_future, _ = target_lookup[horizon]
                strategy_return = -side * raw_future
                for side_value, side_label in ((-1, "sell"), (1, "buy")):
                    eligible = test_mask & (side == side_value)
                    primary_signal = eligible & (primary_q == quantiles)
                    conditioned = primary_signal & (conditioner_q == quantiles)
                    low_condition = primary_signal & (conditioner_q == 1)
                    for strategy, selected in (
                        ("primary_reversion_q_high", primary_signal),
                        ("primary_reversion_conditioner_q_high", conditioned),
                    ):
                        summary = day_block_summary(
                            strategy_return, day, selected
                        )
                        output["fold_signal_summary"].append(
                            {
                                "construction": f"event_{size}",
                                "fold": fold_id,
                                "test_start": fold["test_start"],
                                "test_end": fold["test_end"],
                                "horizon_minutes": horizon,
                                "side": side_label,
                                "primary_feature": primary_feature,
                                "conditioner": conditioner,
                                "strategy": strategy,
                                "signal_rate": summary["observations"] / int(eligible.sum()),
                                **summary,
                                "normal_approx_p_value": normal_p_value(
                                    float(summary["day_t_stat"])
                                ),
                            }
                        )
                    for comparison, first, second in (
                        (
                            "high_conditioner_minus_unconditioned",
                            conditioned,
                            primary_signal,
                        ),
                        (
                            "high_conditioner_minus_low_conditioner",
                            conditioned,
                            low_condition,
                        ),
                    ):
                        result = paired_day_difference(
                            strategy_return, day, first, second
                        )
                        output["fold_conditioner_increment"].append(
                            {
                                "construction": f"event_{size}",
                                "fold": fold_id,
                                "test_start": fold["test_start"],
                                "test_end": fold["test_end"],
                                "horizon_minutes": horizon,
                                "side": side_label,
                                "primary_feature": primary_feature,
                                "conditioner": conditioner,
                                "comparison": comparison,
                                **result,
                                "normal_approx_p_value": normal_p_value(
                                    float(result["day_t_stat"])
                                ),
                            }
                        )
                    for primary_bin in range(1, quantiles + 1):
                        for conditioner_bin in range(1, quantiles + 1):
                            selected = (
                                eligible
                                & (primary_q == primary_bin)
                                & (conditioner_q == conditioner_bin)
                            )
                            summary = day_block_summary(
                                strategy_return, day, selected
                            )
                            output["fold_conditioning_grid"].append(
                                {
                                    "construction": f"event_{size}",
                                    "fold": fold_id,
                                    "test_start": fold["test_start"],
                                    "test_end": fold["test_end"],
                                    "horizon_minutes": horizon,
                                    "side": side_label,
                                    "primary_feature": primary_feature,
                                    "conditioner": conditioner,
                                    "primary_quantile": primary_bin,
                                    "conditioner_quantile": conditioner_bin,
                                    **summary,
                                }
                            )
        print(f"  fold {fold_id} complete", flush=True)
    return output


def aggregate_folds(table: pl.DataFrame, *, value: str, keys: list[str]) -> pl.DataFrame:
    return (
        table.group_by(*keys)
        .agg(
            pl.len().alias("folds"),
            pl.col(value).mean().alias(f"mean_{value}"),
            pl.col(value).median().alias(f"median_{value}"),
            pl.col(value).min().alias(f"minimum_{value}"),
            (pl.col(value) > 0).mean().alias("positive_fold_share"),
        )
        .sort("positive_fold_share", f"median_{value}", descending=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--allow-unverified-cache",
        action="store_true",
        help="Allow legacy caches without a cache_manifest.json fingerprint.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    config = manifest["config"]
    shared_root = Path(manifest["shared_data_root"])
    cache_root = args.cache_root or (
        shared_root / "features" / "alpha" / f"symbol={manifest['symbol']}"
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
            "The bar cache has no cache_manifest.json. Rebuild it with the current "
            "builder, or use --allow-unverified-cache only for a deliberate legacy run."
        )
    if (
        cache_manifest is not None
        and cache_manifest.get("data_fingerprint") != manifest["data_fingerprint"]
    ):
        raise RuntimeError(
            "Bar-cache fingerprint does not match the experiment data fingerprint. "
            "Rebuild the cache from the manifest's shared-data root."
        )
    sizes = [int(value) for value in config["event_bar_sizes"]]
    horizons = [int(value) for value in config["clock_horizons_minutes"]]
    conditioners = list(config["conditioners"])
    primary = str(config["primary_feature"])
    quantiles = int(config.get("quantiles", 5))
    combined: dict[str, list[dict]] = {}
    for size in sizes:
        result = analyse_size(
            size=size,
            cache_path=cache_root / f"event_{size}" / "part-000.parquet",
            folds=manifest["folds"],
            primary_feature=primary,
            conditioners=conditioners,
            horizons=horizons,
            quantiles=quantiles,
        )
        for name, rows in result.items():
            combined.setdefault(name, []).extend(rows)

    args.output_root.mkdir(parents=True, exist_ok=True)
    tables = {name: pl.DataFrame(rows) for name, rows in combined.items()}
    increment_keys = [
        "construction", "horizon_minutes", "side", "primary_feature",
        "conditioner", "comparison",
    ]
    signal_keys = [
        "construction", "horizon_minutes", "side", "primary_feature",
        "conditioner", "strategy",
    ]
    tables["aggregate_conditioner_increment"] = aggregate_folds(
        tables["fold_conditioner_increment"],
        value="day_mean_difference_bps",
        keys=increment_keys,
    )
    tables["aggregate_signal_summary"] = aggregate_folds(
        tables["fold_signal_summary"], value="day_mean_bps", keys=signal_keys
    )
    for name, table in tables.items():
        destination = args.output_root / f"{name}.csv"
        table.write_csv(destination)
        print(destination, flush=True)
    run_manifest = {
        "runner": "run_walk_forward_conditioners.py",
        "source_manifest": str(args.manifest.resolve()),
        "data_fingerprint": manifest["data_fingerprint"],
        "cache_root": str(cache_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "event_bar_sizes": sizes,
        "clock_horizons_minutes": horizons,
        "primary_feature": primary,
        "conditioners": conditioners,
        "quantiles": quantiles,
        "cache_fingerprint_verified": cache_manifest is not None,
        "folds": manifest["folds"],
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
