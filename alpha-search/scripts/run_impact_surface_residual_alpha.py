"""Fit chronological impact surfaces and screen their residuals for gross alpha."""

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

from alpha_search.evaluation import (  # noqa: E402
    benjamini_hochberg,
    day_block_summary,
    forward_clock_targets,
    normal_p_value,
    paired_day_difference,
)
from alpha_search.impact_surface import (  # noqa: E402
    ImpactSurfaceFit,
    fit_log_activity_surface,
    fit_normalised_flow_log,
    fit_normalised_flow_power,
    fit_power_surface,
    fit_relative_imbalance_power,
    predict_surface,
    surface_cells,
)


def utc_ms(value: str) -> int:
    return int(
        datetime.strptime(value, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1_000
    )


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    selected = np.isfinite(x) & np.isfinite(y)
    if selected.sum() < 3:
        return math.nan
    left, right = x[selected], y[selected]
    if left.std() == 0 or right.std() == 0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    return pl.Series(values).rank(method="average").to_numpy()


def _daily_ic(
    factor: np.ndarray,
    target: np.ndarray,
    day: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float | int]:
    mask = selected & np.isfinite(factor) & np.isfinite(target)
    selected_factor = factor[mask]
    selected_target = target[mask]
    selected_day = day[mask]
    correlations = []
    _, starts, counts = np.unique(
        selected_day, return_index=True, return_counts=True
    )
    for start, count in zip(starts, counts, strict=True):
        if count >= 3:
            stop = start + count
            correlation = _pearson(
                selected_factor[start:stop], selected_target[start:stop]
            )
            if math.isfinite(correlation):
                correlations.append(correlation)
    if not correlations:
        return {
            "ic_days": 0,
            "day_mean_ic": math.nan,
            "day_ic_standard_error": math.nan,
            "day_ic_t_stat": math.nan,
        }
    values = np.asarray(correlations)
    standard_error = (
        float(values.std(ddof=1) / math.sqrt(len(values)))
        if len(values) > 1
        else math.nan
    )
    mean = float(values.mean())
    return {
        "ic_days": len(values),
        "day_mean_ic": mean,
        "day_ic_standard_error": standard_error,
        "day_ic_t_stat": (
            mean / standard_error
            if math.isfinite(standard_error) and standard_error > 0
            else math.nan
        ),
    }


def _model_fits(cells, names: list[str]) -> dict[str, ImpactSurfaceFit]:
    factories = {
        "relative_imbalance_power": fit_relative_imbalance_power,
        "normalised_flow_power": fit_normalised_flow_power,
        "normalised_flow_log": fit_normalised_flow_log,
        "surface_power": fit_power_surface,
        "surface_log_activity": fit_log_activity_surface,
    }
    unknown = sorted(set(names).difference(factories))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    return {name: factories[name](cells) for name in names}


def _cell_calibration_rmse(
    z: np.ndarray,
    a: np.ndarray,
    residual: np.ndarray,
    training_cells,
    imbalance_bins: int,
    activity_bins: int,
) -> tuple[float, int]:
    cells = surface_cells(
        z,
        a,
        residual,
        imbalance_bins=imbalance_bins,
        activity_bins=activity_bins,
        imbalance_cuts=training_cells.imbalance_cuts,
        activity_cuts=training_cells.activity_cuts,
    )
    return float(np.sqrt(np.mean(cells.mean_impact**2))), len(cells.mean_impact)


def _fit_record(
    *,
    construction: str,
    fold: int,
    side_name: str,
    model: str,
    fit: ImpactSurfaceFit,
    train_observations: int,
    test_observations: int,
    test_bar_rmse: float,
    test_cell_rmse: float,
    test_cells: int,
    residual_activity_correlation: float,
) -> dict:
    return {
        "construction": construction,
        "fold": fold,
        "side": side_name,
        "model": model,
        "train_observations": train_observations,
        "test_observations": test_observations,
        "training_cells": fit.cells,
        "amplitude": fit.amplitude,
        "imbalance_exponent_p": fit.imbalance_exponent,
        "activity_exponent_q": fit.activity_exponent,
        "collapse_exponent_gamma": fit.collapse_exponent,
        "imbalance_scale": fit.imbalance_scale,
        "curvature": fit.curvature,
        "train_cell_rmse": fit.cell_rmse,
        "test_cell_rmse": test_cell_rmse,
        "test_cells": test_cells,
        "test_bar_rmse": test_bar_rmse,
        "test_residual_log_activity_correlation": residual_activity_correlation,
    }


def analyse_construction(
    construction: str,
    path: Path,
    config: dict,
) -> dict[str, list[dict]]:
    print(f"Loading {construction}", flush=True)
    frame = (
        pl.scan_parquet(path)
        .sort("end_time_ms")
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
    day = time // 86_400_000
    targets = forward_clock_targets(time, price, config["future_horizons_minutes"])
    surface_config = config["surface"]
    n_z = int(surface_config["imbalance_bins"])
    n_a = int(surface_config["activity_bins"])
    model_names = list(surface_config["models"])
    n_quantiles = int(config["residual_quantiles"])
    rows: dict[str, list[dict]] = {
        "surface_fits": [],
        "surface_cells": [],
        "residual_cuts": [],
        "residual_ic": [],
        "residual_quantiles": [],
        "residual_strategies": [],
        "horizon_accuracy": [],
    }

    for fold_config in config["folds"]:
        fold = int(fold_config["fold"])
        train_end = utc_ms(fold_config["train_end"])
        test_start = utc_ms(fold_config["test_start"])
        test_end = utc_ms(fold_config["test_end"])
        training_period = time < train_end
        test_period = (time >= test_start) & (time < test_end)
        print(
            f"  fold {fold}: train < {fold_config['train_end']}; "
            f"test {fold_config['test_start']} to {fold_config['test_end']}",
            flush=True,
        )
        for horizon, (_, elapsed) in targets.items():
            chosen = test_period & np.isfinite(elapsed)
            rows["horizon_accuracy"].append(
                {
                    "construction": construction,
                    "fold": fold,
                    "horizon_minutes": horizon,
                    "observations": int(chosen.sum()),
                    "elapsed_p10_minutes": float(np.quantile(elapsed[chosen], 0.10)),
                    "elapsed_median_minutes": float(np.median(elapsed[chosen])),
                    "elapsed_p90_minutes": float(np.quantile(elapsed[chosen], 0.90)),
                }
            )

        for side_value, side_name in [(-1, "sell"), (1, "buy")]:
            finite = (
                np.isfinite(z)
                & np.isfinite(activity)
                & np.isfinite(impact)
                & (z > 0)
                & (activity > 0)
                & (side == side_value)
            )
            train = training_period & finite
            test = test_period & finite
            cells = surface_cells(
                z[train],
                activity[train],
                impact[train],
                imbalance_bins=n_z,
                activity_bins=n_a,
            )
            for index in range(len(cells.mean_impact)):
                rows["surface_cells"].append(
                    {
                        "construction": construction,
                        "fold": fold,
                        "side": side_name,
                        "imbalance_bin": int(cells.imbalance_bin[index]) + 1,
                        "activity_bin": int(cells.activity_bin[index]) + 1,
                        "observations": int(cells.observations[index]),
                        "mean_relative_imbalance": float(cells.mean_imbalance[index]),
                        "mean_relative_activity": float(cells.mean_activity[index]),
                        "mean_normalised_impact": float(cells.mean_impact[index]),
                        "impact_standard_error": float(cells.standard_error[index]),
                    }
                )
            fits = _model_fits(cells, model_names)
            for model_name, fit in fits.items():
                training_prediction = predict_surface(fit, z, activity)
                residual = impact - training_prediction
                cuts = np.quantile(
                    residual[train], np.linspace(0, 1, n_quantiles + 1)[1:-1]
                )
                quantile = np.zeros(len(frame), dtype=np.int8)
                applicable = finite & np.isfinite(residual)
                quantile[applicable] = (
                    np.digitize(residual[applicable], cuts, right=True) + 1
                )
                for cut_number, cut in enumerate(cuts, start=1):
                    rows["residual_cuts"].append(
                        {
                            "construction": construction,
                            "fold": fold,
                            "side": side_name,
                            "model": model_name,
                            "upper_edge_for_quantile": cut_number,
                            "cut_value": float(cut),
                        }
                    )
                test_cell_rmse, test_cells = _cell_calibration_rmse(
                    z[test], activity[test], residual[test], cells, n_z, n_a
                )
                rows["surface_fits"].append(
                    _fit_record(
                        construction=construction,
                        fold=fold,
                        side_name=side_name,
                        model=model_name,
                        fit=fit,
                        train_observations=int(train.sum()),
                        test_observations=int(test.sum()),
                        test_bar_rmse=float(np.sqrt(np.mean(residual[test] ** 2))),
                        test_cell_rmse=test_cell_rmse,
                        test_cells=test_cells,
                        residual_activity_correlation=_pearson(
                            residual[test], np.log(activity[test])
                        ),
                    )
                )

                for horizon, (future_return, _) in targets.items():
                    aligned_future = side_value * future_return
                    selected = test & np.isfinite(future_return)
                    raw_ic = _pearson(residual[selected], aligned_future[selected])
                    rank_ic = _pearson(
                        _rank(residual[selected]), _rank(aligned_future[selected])
                    )
                    day_ic = _daily_ic(residual, aligned_future, day, selected)
                    p_value = normal_p_value(float(day_ic["day_ic_t_stat"]))
                    rows["residual_ic"].append(
                        {
                            "construction": construction,
                            "fold": fold,
                            "side": side_name,
                            "model": model_name,
                            "horizon_minutes": horizon,
                            "observations": int(selected.sum()),
                            "pearson_ic": raw_ic,
                            "spearman_ic": rank_ic,
                            **day_ic,
                            "day_ic_p_value": p_value,
                        }
                    )
                    for quantile_value in range(1, n_quantiles + 1):
                        chosen = selected & (quantile == quantile_value)
                        summary = day_block_summary(aligned_future, day, chosen)
                        rows["residual_quantiles"].append(
                            {
                                "construction": construction,
                                "fold": fold,
                                "side": side_name,
                                "model": model_name,
                                "horizon_minutes": horizon,
                                "residual_quantile": quantile_value,
                                "mean_residual": (
                                    float(residual[chosen].mean())
                                    if chosen.any()
                                    else math.nan
                                ),
                                **summary,
                            }
                        )
                    high = selected & (quantile == n_quantiles)
                    low = selected & (quantile == 1)
                    for strategy_name, strategy_return, chosen in [
                        ("high_residual_reversion", -aligned_future, high),
                        ("low_residual_continuation", aligned_future, low),
                    ]:
                        rows["residual_strategies"].append(
                            {
                                "construction": construction,
                                "fold": fold,
                                "side": side_name,
                                "model": model_name,
                                "horizon_minutes": horizon,
                                "strategy": strategy_name,
                                **day_block_summary(strategy_return, day, chosen),
                            }
                        )
                    rows["residual_strategies"].append(
                        {
                            "construction": construction,
                            "fold": fold,
                            "side": side_name,
                            "model": model_name,
                            "horizon_minutes": horizon,
                            "strategy": "aligned_q5_minus_q1",
                            "observations": int(high.sum() + low.sum()),
                            "days": math.nan,
                            "mean_return_bps": math.nan,
                            "day_mean_bps": math.nan,
                            "day_standard_error_bps": math.nan,
                            "day_t_stat": math.nan,
                            "positive_day_share": math.nan,
                            **paired_day_difference(
                                aligned_future, day, high, low
                            ),
                        }
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "alpha-search" / "configs" / "impact_surface_residual_alpha_v1.json",
    )
    parser.add_argument(
        "--shared-data-root", type=Path, default=REPO / "shared-data"
    )
    parser.add_argument("--output-root", type=Path)
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
        raise ValueError("Cache holdout declaration does not match the experiment config.")
    output_root = args.output_root or (
        REPO / "alpha-search" / "reports" / config["experiment_id"]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    combined: dict[str, list[dict]] = {}
    for construction in config["constructions"]:
        path = cache_root / construction / "part-000.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        result = analyse_construction(construction, path, config)
        for name, records in result.items():
            combined.setdefault(name, []).extend(records)

    ic = pl.DataFrame(combined["residual_ic"]).with_columns(
        pl.Series(
            "day_ic_fdr_bh",
            benjamini_hochberg(
                [row["day_ic_p_value"] for row in combined["residual_ic"]]
            ),
        )
    )
    combined.pop("residual_ic")
    for name, records in combined.items():
        pl.DataFrame(records).write_csv(output_root / f"{name}.csv")
    ic.write_csv(output_root / "residual_ic.csv")
    run_manifest = {
        "experiment_id": config["experiment_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config.resolve()),
        "config": config,
        "cache_manifest": cache_manifest,
        "output_files": sorted(path.name for path in output_root.glob("*.csv")),
        "status": "complete",
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n"
    )
    print(output_root, flush=True)


if __name__ == "__main__":
    main()
