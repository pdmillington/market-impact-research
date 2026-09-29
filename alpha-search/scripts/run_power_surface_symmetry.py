"""Compare pooled and side-specific power surfaces out of period."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))

from alpha_search.impact_surface import (  # noqa: E402
    fit_common_shape_power_surface,
    fit_power_surface,
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


def cell_rmse(z, activity, impact, prediction, training_cells, n_z, n_a):
    residual_cells = surface_cells(
        z,
        activity,
        impact - prediction,
        imbalance_bins=n_z,
        activity_bins=n_a,
        imbalance_cuts=training_cells.imbalance_cuts,
        activity_cuts=training_cells.activity_cuts,
    )
    return float(np.sqrt(np.mean(residual_cells.mean_impact**2)))


def analyse(path: Path, construction: str, config: dict) -> list[dict]:
    print(f"Loading {construction}", flush=True)
    frame = (
        pl.scan_parquet(path)
        .sort("end_time_ms")
        .select(
            "end_time_ms", "signed_imbalance", "relative_imbalance",
            "relative_activity", "normalised_signed_impact",
        )
        .collect()
    )
    time = frame["end_time_ms"].to_numpy()
    side = np.sign(frame["signed_imbalance"].to_numpy()).astype(np.int8)
    z = frame["relative_imbalance"].to_numpy()
    activity = frame["relative_activity"].to_numpy()
    impact = frame["normalised_signed_impact"].to_numpy()
    finite = (
        np.isfinite(z) & np.isfinite(activity) & np.isfinite(impact)
        & (z > 0) & (activity > 0) & (side != 0)
    )
    n_z = int(config["surface"]["imbalance_bins"])
    n_a = int(config["surface"]["activity_bins"])
    rows = []
    for fold in config["folds"]:
        train = finite & (time < utc_ms(fold["train_end"]))
        test = (
            finite & (time >= utc_ms(fold["test_start"]))
            & (time < utc_ms(fold["test_end"]))
        )
        cells = {}
        separate = {}
        for side_value, side_name in [(-1, "sell"), (1, "buy")]:
            selected = train & (side == side_value)
            cells[side_name] = surface_cells(
                z[selected], activity[selected], impact[selected],
                imbalance_bins=n_z, activity_bins=n_a,
            )
            separate[side_name] = fit_power_surface(cells[side_name])
        common_shape = fit_common_shape_power_surface(
            cells["buy"], cells["sell"], common_amplitude=False
        )
        common_surface = fit_common_shape_power_surface(
            cells["buy"], cells["sell"], common_amplitude=True
        )
        specifications = {
            "separate_shape_and_level": {
                "p_buy": separate["buy"].imbalance_exponent,
                "p_sell": separate["sell"].imbalance_exponent,
                "q_buy": separate["buy"].activity_exponent,
                "q_sell": separate["sell"].activity_exponent,
                "amplitude_buy": separate["buy"].amplitude,
                "amplitude_sell": separate["sell"].amplitude,
            },
            "common_shape_separate_level": {
                "p_buy": common_shape.imbalance_exponent,
                "p_sell": common_shape.imbalance_exponent,
                "q_buy": common_shape.activity_exponent,
                "q_sell": common_shape.activity_exponent,
                "amplitude_buy": common_shape.buy_amplitude,
                "amplitude_sell": common_shape.sell_amplitude,
            },
            "common_shape_and_level": {
                "p_buy": common_surface.imbalance_exponent,
                "p_sell": common_surface.imbalance_exponent,
                "q_buy": common_surface.activity_exponent,
                "q_sell": common_surface.activity_exponent,
                "amplitude_buy": common_surface.buy_amplitude,
                "amplitude_sell": common_surface.sell_amplitude,
            },
        }
        for specification, parameters in specifications.items():
            side_rmse = {}
            for side_value, side_name in [(-1, "sell"), (1, "buy")]:
                chosen = test & (side == side_value)
                p = parameters[f"p_{side_name}"]
                q = parameters[f"q_{side_name}"]
                amplitude = parameters[f"amplitude_{side_name}"]
                prediction = amplitude * z[chosen] ** p * activity[chosen] ** q
                side_rmse[side_name] = cell_rmse(
                    z[chosen], activity[chosen], impact[chosen], prediction,
                    cells[side_name], n_z, n_a,
                )
            rows.append(
                {
                    "construction": construction,
                    "fold": int(fold["fold"]),
                    "specification": specification,
                    **parameters,
                    "gamma_buy": parameters["q_buy"] / parameters["p_buy"],
                    "gamma_sell": parameters["q_sell"] / parameters["p_sell"],
                    "buy_test_cell_rmse": side_rmse["buy"],
                    "sell_test_cell_rmse": side_rmse["sell"],
                    "combined_test_cell_rmse": float(
                        np.sqrt((side_rmse["buy"] ** 2 + side_rmse["sell"] ** 2) / 2)
                    ),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "alpha-search" / "configs" / "impact_surface_residual_alpha_v1.json",
    )
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO / "alpha-search" / "reports" / "rolling_power_surface_v1",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cache_root = (
        args.shared_data_root.expanduser().resolve() / "features" / "alpha"
        / f"symbol={config['symbol']}" / config["cache_version"]
    )
    rows = []
    for construction in config["constructions"]:
        rows.extend(analyse(
            cache_root / construction / "part-000.parquet", construction, config
        ))
    args.output_root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).sort("construction", "fold", "specification").write_csv(
        args.output_root / "symmetry_comparison.csv"
    )
    print(args.output_root / "symmetry_comparison.csv")


if __name__ == "__main__":
    main()
