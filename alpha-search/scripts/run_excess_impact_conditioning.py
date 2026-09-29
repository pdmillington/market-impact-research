"""Benchmark short-horizon reversion and test excess-impact conditioning."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))

from alpha_search.evaluation import (  # noqa: E402
    benjamini_hochberg,
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


SIZES = (200, 500, 1_000)
HORIZONS_MINUTES = (5, 10, 15, 20, 30)
PERIODS = ("validation", "later_sandbox")


def utc_ms(value: str) -> int:
    return int(
        datetime.strptime(value, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1_000
    )


def _period_masks(
    end_time_ms: np.ndarray,
    train_end: int,
    validation_end: int,
) -> dict[str, np.ndarray]:
    return {
        "train": end_time_ms < train_end,
        "validation": (end_time_ms >= train_end) & (end_time_ms < validation_end),
        "later_sandbox": end_time_ms >= validation_end,
    }


def _strategy_definitions(
    *,
    side: np.ndarray,
    raw_pretrend: np.ndarray,
    absolute_pretrend_quintile: np.ndarray,
    aligned_pretrend_quintile: np.ndarray,
    residual_quintile: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return strategy selection and position direction arrays."""

    return {
        "prior_return_contrarian_q5": (
            absolute_pretrend_quintile == 5,
            -np.sign(raw_pretrend),
        ),
        "aligned_pretrend_reversion_q5": (
            aligned_pretrend_quintile == 5,
            -side,
        ),
        "excess_impact_reversion_q5": (
            residual_quintile == 5,
            -side,
        ),
        "aligned_pretrend_q5_conditioned_excess_q5": (
            (aligned_pretrend_quintile == 5) & (residual_quintile == 5),
            -side,
        ),
    }


def analyse_size(
    *,
    size: int,
    path: Path,
    train_end: int,
    validation_end: int,
) -> dict[str, list[dict]]:
    print(f"Loading event_{size}", flush=True)
    base = pl.scan_parquet(path).sort("bar_number")
    first_ms = int(base.select(pl.col("end_time_ms").min()).collect().item())
    usable = base.filter(pl.col("end_time_ms") >= first_ms + 86_400_000)
    fit = fit_asymmetric_log_impact(
        usable.filter(pl.col("end_time_ms") < train_end)
    )
    frame = (
        add_asymmetric_impact_features(usable, fit)
        .select(
            "end_time_ms",
            "end_price",
            "signed_imbalance",
            "aligned_pretrend_12000_events",
            "normalised_imbalance",
            "normalised_signed_impact",
            "expected_normalised_impact",
            "impact_residual",
        )
        .collect()
        .rechunk()
    )

    time = frame["end_time_ms"].to_numpy()
    price = frame["end_price"].to_numpy()
    side = np.sign(frame["signed_imbalance"].to_numpy()).astype(np.int8)
    aligned_pretrend = frame["aligned_pretrend_12000_events"].to_numpy()
    raw_pretrend = side * aligned_pretrend
    residual = frame["impact_residual"].to_numpy()
    day = time // 86_400_000
    month = time.astype("datetime64[ms]").astype("datetime64[M]").astype(str)
    period_masks = _period_masks(time, train_end, validation_end)

    sided_training = period_masks["train"] & (side != 0)
    aligned_q, aligned_cuts = frozen_quantiles(
        aligned_pretrend, sided_training, side, n_quantiles=5
    )
    residual_q, residual_cuts = frozen_quantiles(
        residual, sided_training, side, n_quantiles=5
    )
    absolute_q, absolute_cuts = frozen_quantiles(
        np.abs(raw_pretrend),
        period_masks["train"],
        np.zeros(len(frame), dtype=np.int8),
        n_quantiles=5,
    )

    rows: dict[str, list[dict]] = {
        "impact_fits": [],
        "impact_curve_bins": [],
        "residual_monthly": [],
        "quantile_cuts": [],
        "horizon_accuracy": [],
        "quintile_curves": [],
        "conditioning_grid": [],
        "signal_summary": [],
        "incremental_conditioning": [],
        "monthly_signal_summary": [],
    }
    for side_label, side_fit in (("buy", fit.buy), ("sell", fit.sell)):
        rows["impact_fits"].append(
            {
                "construction": f"event_{size}",
                "side": side_label,
                "observations": side_fit.observations,
                "imbalance_scale": fit.imbalance_scale,
                "intercept": side_fit.intercept,
                "slope": side_fit.slope,
                "r_squared": side_fit.r_squared,
            }
        )
    normalised_imbalance = frame["normalised_imbalance"].to_numpy()
    observed_impact = frame["normalised_signed_impact"].to_numpy()
    expected_impact = frame["expected_normalised_impact"].to_numpy()
    for side_value, side_label in ((-1, "sell"), (1, "buy")):
        training_selected = (
            period_masks["train"]
            & (side == side_value)
            & np.isfinite(normalised_imbalance)
            & np.isfinite(observed_impact)
            & np.isfinite(expected_impact)
        )
        edges = np.unique(
            np.quantile(normalised_imbalance[training_selected], np.linspace(0, 1, 21))
        )
        bin_id = np.digitize(
            normalised_imbalance[training_selected], edges[1:-1], right=True
        )
        selected_x = normalised_imbalance[training_selected]
        selected_y = observed_impact[training_selected]
        selected_expected = expected_impact[training_selected]
        for value in range(len(edges) - 1):
            chosen = bin_id == value
            if not chosen.any():
                continue
            rows["impact_curve_bins"].append(
                {
                    "construction": f"event_{size}",
                    "side": side_label,
                    "bin": value + 1,
                    "observations": int(chosen.sum()),
                    "mean_normalised_imbalance": float(selected_x[chosen].mean()),
                    "mean_observed_impact": float(selected_y[chosen].mean()),
                    "mean_fitted_impact": float(selected_expected[chosen].mean()),
                    "mean_residual": float(
                        (selected_y[chosen] - selected_expected[chosen]).mean()
                    ),
                }
            )
        for month_value in np.unique(month[side == side_value]):
            chosen = (
                (side == side_value)
                & (month == month_value)
                & np.isfinite(residual)
            )
            if not chosen.any():
                continue
            values = residual[chosen]
            rows["residual_monthly"].append(
                {
                    "construction": f"event_{size}",
                    "side": side_label,
                    "month": month_value,
                    "observations": int(chosen.sum()),
                    "mean_residual": float(values.mean()),
                    "median_residual": float(np.median(values)),
                    "residual_p10": float(np.quantile(values, 0.10)),
                    "residual_p90": float(np.quantile(values, 0.90)),
                }
            )
    for variable, cuts_by_group in (
        ("aligned_pretrend_12000_events", aligned_cuts),
        ("impact_residual", residual_cuts),
        ("absolute_pretrend_12000_events", absolute_cuts),
    ):
        for group, cuts in cuts_by_group.items():
            for quantile, cut in enumerate(cuts, start=1):
                rows["quantile_cuts"].append(
                    {
                        "construction": f"event_{size}",
                        "variable": variable,
                        "side": group,
                        "upper_edge_for_quantile": quantile,
                        "cut_value": float(cut),
                    }
                )

    strategies = _strategy_definitions(
        side=side,
        raw_pretrend=raw_pretrend,
        absolute_pretrend_quintile=absolute_q,
        aligned_pretrend_quintile=aligned_q,
        residual_quintile=residual_q,
    )
    targets = forward_clock_targets(time, price, HORIZONS_MINUTES)
    for horizon, (raw_future_return, elapsed) in targets.items():
        aligned_future_return = side * raw_future_return
        for period, period_mask in period_masks.items():
            valid_elapsed = elapsed[period_mask & np.isfinite(elapsed)]
            rows["horizon_accuracy"].append(
                {
                    "construction": f"event_{size}",
                    "horizon_minutes": horizon,
                    "period": period,
                    "observations": int(len(valid_elapsed)),
                    "elapsed_p10_minutes": float(np.quantile(valid_elapsed, 0.10)),
                    "elapsed_median_minutes": float(np.median(valid_elapsed)),
                    "elapsed_p90_minutes": float(np.quantile(valid_elapsed, 0.90)),
                    "median_endpoint_delay_seconds": float(
                        60.0 * np.median(valid_elapsed - horizon)
                    ),
                    "p90_endpoint_delay_seconds": float(
                        60.0 * np.quantile(valid_elapsed - horizon, 0.90)
                    ),
                }
            )
        for period in PERIODS:
            period_mask = period_masks[period]
            for side_value, side_label in ((-1, "sell"), (1, "buy")):
                side_mask = period_mask & (side == side_value)
                for variable, values, quantiles in (
                    ("aligned_pretrend_12000_events", aligned_pretrend, aligned_q),
                    ("impact_residual", residual, residual_q),
                ):
                    for quantile in range(1, 6):
                        selected = side_mask & (quantiles == quantile)
                        summary = day_block_summary(
                            aligned_future_return, day, selected
                        )
                        finite = selected & np.isfinite(values)
                        rows["quintile_curves"].append(
                            {
                                "construction": f"event_{size}",
                                "horizon_minutes": horizon,
                                "period": period,
                                "side": side_label,
                                "variable": variable,
                                "quintile": quantile,
                                "mean_variable": (
                                    float(values[finite].mean())
                                    if finite.any()
                                    else math.nan
                                ),
                                **summary,
                            }
                        )
                for pretrend_quantile in range(1, 6):
                    for excess_quantile in range(1, 6):
                        selected = (
                            side_mask
                            & (aligned_q == pretrend_quantile)
                            & (residual_q == excess_quantile)
                        )
                        summary = day_block_summary(
                            aligned_future_return, day, selected
                        )
                        rows["conditioning_grid"].append(
                            {
                                "construction": f"event_{size}",
                                "horizon_minutes": horizon,
                                "period": period,
                                "side": side_label,
                                "pretrend_quintile": pretrend_quantile,
                                "excess_impact_quintile": excess_quantile,
                                **summary,
                            }
                        )

            for strategy, (selection, position) in strategies.items():
                strategy_return = position * raw_future_return
                strategy_sides = ((0, "all"),) if strategy.startswith(
                    "prior_return"
                ) else ((-1, "sell"), (1, "buy"))
                for side_value, side_label in strategy_sides:
                    selected = period_mask & selection
                    if side_value:
                        selected &= side == side_value
                    summary = day_block_summary(strategy_return, day, selected)
                    denominator = int(
                        (period_mask & ((side == side_value) if side_value else True)).sum()
                    )
                    rows["signal_summary"].append(
                        {
                            "construction": f"event_{size}",
                            "horizon_minutes": horizon,
                            "period": period,
                            "side": side_label,
                            "strategy": strategy,
                            "signal_rate": (
                                summary["observations"] / denominator
                                if denominator
                                else math.nan
                            ),
                            **summary,
                            "normal_approx_p_value": normal_p_value(
                                float(summary["day_t_stat"])
                            ),
                        }
                    )
                    chosen_months = np.unique(month[selected & np.isfinite(strategy_return)])
                    for month_value in chosen_months:
                        month_selected = selected & (month == month_value)
                        month_values = strategy_return[
                            month_selected & np.isfinite(strategy_return)
                        ]
                        rows["monthly_signal_summary"].append(
                            {
                                "construction": f"event_{size}",
                                "horizon_minutes": horizon,
                                "period": period,
                                "side": side_label,
                                "strategy": strategy,
                                "month": month_value,
                                "observations": int(len(month_values)),
                                "mean_return_bps": float(month_values.mean()),
                            }
                        )

            for side_value, side_label in ((-1, "sell"), (1, "buy")):
                common = period_mask & (side == side_value) & (aligned_q == 5)
                strategy_return = -side * raw_future_return
                comparisons = {
                    "high_excess_q5_minus_low_excess_q1": (
                        common & (residual_q == 5),
                        common & (residual_q == 1),
                    ),
                    "high_excess_q5_minus_unconditioned": (
                        common & (residual_q == 5),
                        common,
                    ),
                }
                for comparison, (first, second) in comparisons.items():
                    result = paired_day_difference(
                        strategy_return, day, first, second
                    )
                    rows["incremental_conditioning"].append(
                        {
                            "construction": f"event_{size}",
                            "horizon_minutes": horizon,
                            "period": period,
                            "side": side_label,
                            "comparison": comparison,
                            **result,
                            "normal_approx_p_value": normal_p_value(
                                float(result["day_t_stat"])
                            ),
                        }
                    )
        print(f"  horizon={horizon}m complete", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-end", default="2020-09-01")
    parser.add_argument("--validation-end", default="2021-07-01")
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=(
            REPO
            / "shared-data"
            / "features"
            / "alpha"
            / "symbol=BTCUSDT"
            / "short_event_screen_v1"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "alpha-search" / "reports" / "excess_impact_v1",
    )
    args = parser.parse_args()

    unknown = set(args.sizes).difference(SIZES)
    if unknown:
        raise ValueError(f"Unsupported event sizes: {sorted(unknown)}")
    combined: dict[str, list[dict]] = {}
    for size in args.sizes:
        result = analyse_size(
            size=size,
            path=args.cache_root / f"event_{size}" / "part-000.parquet",
            train_end=utc_ms(args.train_end),
            validation_end=utc_ms(args.validation_end),
        )
        for name, records in result.items():
            combined.setdefault(name, []).extend(records)

    signal = pl.DataFrame(combined["signal_summary"])
    validation = signal["period"] == "validation"
    adjusted = np.full(signal.height, np.nan)
    adjusted[validation.to_numpy()] = benjamini_hochberg(
        signal.filter(validation)["normal_approx_p_value"].to_numpy()
    )
    combined["signal_summary"] = signal.with_columns(
        pl.Series("validation_fdr_q_value", adjusted)
    ).to_dicts()

    incremental = pl.DataFrame(combined["incremental_conditioning"])
    validation = incremental["period"] == "validation"
    adjusted = np.full(incremental.height, np.nan)
    adjusted[validation.to_numpy()] = benjamini_hochberg(
        incremental.filter(validation)["normal_approx_p_value"].to_numpy()
    )
    combined["incremental_conditioning"] = incremental.with_columns(
        pl.Series("validation_fdr_q_value", adjusted)
    ).to_dicts()

    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, records in combined.items():
        destination = args.output_root / f"{name}.csv"
        pl.DataFrame(records).write_csv(destination)
        print(destination, flush=True)


if __name__ == "__main__":
    main()
