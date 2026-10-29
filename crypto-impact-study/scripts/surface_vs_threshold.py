"""Like-for-like comparison of the activity surface and the threshold model (revision log A3, A3c-1).

Data: the 4,000-event and five-minute bars of StateDependentEffectiveLiquidity
(shared builder, 2021-01..2026-08), with the same response: signed bar return over
trailing 24 h realised volatility of the same bars (time-based 24 h warm-up).

Common cell grid (the paper's): 5 quintiles of log event intensity z x 20
equal-count imbalance bins within each quintile. Every model is fitted to these
cells with the same inverse-variance weights and least-squares loss
(crypto_impact.effective_liquidity):

    pooled_log         2 parameters
    state_log          4
    activity_tail      7   (event intensity z)
    surface_volume     3   R F(q a^gamma / Q), a = V / V_24h
    surface_intensity  3   same, a = lambda / lambda_med
    surface_volume_scaled 3  surface_volume with scales following the bar's event
                             count N_t: Q = Q0 (N_t/N_med)^xi, R = R0 (N_t/N_med)^psi,
                             xi and psi from the identified multi-N fit. For 4,000-event
                             bars N_t is constant and this equals surface_volume; for
                             five-minute bars N_t varies with activity.

The surface's master-curve shape (alpha, beta) is held at a multi-N estimate: at a
single bar size the shape and the scales Q, R are not separately identified (free
fits degenerate to a power law or a straight line with alpha, beta at their bounds).
Only Q, R and gamma are local, as in the multi-N surface; the shape (and, for
surface_volume_scaled, xi and psi) comes from the multi-N surface on N = 250..16,000
(pb_blocks_v1):
- in sample: the headline full-sample fit (activity_surface_n250_2021_2026);
- out of sample: a multi-N fit on the fold's training data only, so that no test-period
  data reach the surface through the fixed parameters. Training blocks are those whose
  start and end days both lie outside the test period (blocked) or before the test year
  (expanding); cell bin edges come from the training blocks; xi and psi are the slopes
  over the identified N (at least 10% of cells beyond the bend), as for the headline.

where q = |Q| / V. Scores: weighted RMSE, tail WRMSE (top imbalance bin of each
quintile) and high-activity tail WRMSE, in sample; and out of sample in two designs:
- blocked: the 5 contiguous day blocks used in StateDependentEffectiveLiquidity;
- expanding by year: train from 2021-01 to the end of year Y-1, test year Y
  (2022..2026), as in the surface's own out-of-sample test.
Test cells are built from the test data alone. Results are reported block by block
so that the post-2024 change (revision log A3b) is visible.

Outputs: reports/redo/surface_vs_threshold/{in_sample,parameters,out_of_sample,fold_shapes}.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "crypto-impact-study" / "src"), str(REPO / "crypto-impact-study" / "scripts"),
                str(REPO / "shared-methodology" / "src")]

import run_activity_surface as run  # noqa: E402
from crypto_impact import effective_liquidity as el  # noqa: E402
from crypto_impact.pb_surface import aggregate_cells, fit_activity_surface, grid_cuts, monthly_cell_stats  # noqa: E402
from surface_identification_and_stability import identified_exponents  # noqa: E402
from crypto_impact.monthly_eda import build_fixed_event_bars, build_time_bars  # noqa: E402

OUT = REPO / "crypto-impact-study" / "reports" / "redo" / "surface_vs_threshold"
DATA_DIR = REPO / "shared-data" / "features" / "academic" / "symbol=BTCUSDT" / "paper_events_v2"
START, END = "2021-01", "2026-08"
MODELS = ["pooled_log", "state_log", "activity_tail", "surface_volume", "surface_intensity", "surface_volume_scaled"]
EXTRA = ("log_q", "log_a", "log_n")
N_STARTS = 16
TEST_YEARS = (2022, 2023, 2024, 2025, 2026)
COMMON = REPO / "crypto-impact-study" / "reports" / "redo" / "activity_surface_n250_2021_2026"  # headline fit, N = 250..16,000


N_HEADLINE = (250, 500, 1000, 2000, 4000, 8000, 16000)


def common_shape() -> dict:
    fits = pd.read_csv(COMMON / "fits.csv").set_index("model")
    return {"log_alpha": float(np.log(fits.loc["activity_surface", "alpha"])),
            "log_beta": float(np.log(fits.loc["activity_surface", "beta"]))}


def common_scaling() -> dict:
    exponents = pd.read_csv(COMMON / "scaling_exponents_identified.csv").set_index("model")
    return {"xi": float(exponents.loc["activity_surface", "xi"]), "psi": float(exponents.loc["activity_surface", "psi"])}


def fold_shapes(folds: dict) -> dict:
    """Multi-N surface (N = 250..16,000) fitted on each fold's training blocks only.

    `folds` maps (design, label) -> (first test day, last test day) as datetime64[D].
    Returns (design, label) -> (shape, scaling, summary row). One N is in memory at a time.
    """

    stats = {key: [] for key in folds}
    train_blocks = {key: 0 for key in folds}
    for n in N_HEADLINE:
        arr = run.load_arrays(REPO / "shared-data", n, "2020-12", "2021-01", "2026-08")
        start_day = arr["day"].astype("datetime64[D]")
        end_day = arr["end_day"].astype("datetime64[D]")
        for key, (first, last) in folds.items():
            design, label = key
            if design == "blocked":
                train = ((end_day < first) | (start_day > last))
            else:
                train = end_day < np.datetime64(f"{label}-01-01")
            ok = train & np.isfinite(arr["y_vol"])
            cuts = {s: grid_cuts(arr["z"][ok & (arr["side"] == s) & (arr["z"] > 0)],
                                 arr["a"][ok & (arr["side"] == s) & np.isfinite(arr["a"])], run.N_Z, run.N_A)
                    for s in (-1, 1)}
            fold_stats = monthly_cell_stats(np.zeros(int(ok.sum()), dtype=np.int32), arr["side"][ok], arr["z"][ok],
                                            arr["a"][ok], arr["y_vol"][ok], cuts)
            fold_stats.insert(0, "n_events", n)
            stats[key].append(fold_stats)
            train_blocks[key] += int(ok.sum())
        del arr, start_day, end_day
        print("fold shapes: loaded N =", n, flush=True)

    result = {}
    for key, frames in stats.items():
        cells = aggregate_cells(pd.concat(frames, ignore_index=True))
        fit = fit_activity_surface(cells, n_starts=8)
        exps = identified_exponents(fit, cells)
        shape = {"log_alpha": float(np.log(fit.alpha)), "log_beta": float(np.log(fit.beta))}
        scaling = {"xi": float(exps["xi_identified"]), "psi": float(exps["psi_identified"])}
        if not np.all(np.isfinite(list(scaling.values()))):
            raise RuntimeError(f"fewer than 3 identified N in fold {key}")
        row = {"design": key[0], "test_block": key[1], "test_start": folds[key][0], "test_end": folds[key][1],
               "train_blocks": train_blocks[key], "cells": len(cells), "gamma": fit.gamma, "alpha": fit.alpha,
               "beta": fit.beta, "xi": scaling["xi"], "psi": scaling["psi"], "identified_n": exps["identified_n"],
               "converged": fit.converged}
        result[key] = (shape, scaling, row)
        print("fold shape", key, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)}, flush=True)
    return result


def analysis_samples() -> dict[str, pd.DataFrame]:
    files = [p for p in sorted(DATA_DIR.glob("BTCUSDT-trades-*.parquet")) if START <= p.stem[-7:] <= END]
    event_bars = build_fixed_event_bars(files, events_per_bar=4000)
    time_bars = build_time_bars(files, ["5min"])["5min"]
    time_bars["start_time"] = time_bars.index
    time_bars["end_time"] = time_bars.index + pd.Timedelta("5min")
    time_bars["duration_seconds"] = 300.0
    samples = {}
    for name, bars in (("event_bars", event_bars), ("time_bars", time_bars)):
        data, _ = el.add_model_scales(el.prepare_normalised_analysis(bars))
        # Day of the bar's last instant: a five-minute bar's end is exclusive, so the 23:55 bar
        # belongs to the day it trades in, not the next (by end_time, 2026-08-31 23:55 would fall
        # on 2026-09-01). Changes the day only for bars ending exactly at midnight.
        data["block_day"] = (pd.to_datetime(data["end_time"], utc=True) - pd.Timedelta(1, "ns")).dt.floor("D")
        data["log_q"] = np.log(data["absolute_imbalance_ratio"])
        data["log_a"] = np.log(data["gross_volume"] / data["trailing_market_volume"])
        data["log_n"] = np.log(data["trade_count"] / data["trade_count"].median())
        samples[name] = data.replace([np.inf, -np.inf], np.nan).dropna(subset=list(EXTRA))
    return samples


def fit_all(cells: pd.DataFrame, random_state: int, shape: dict, scaling: dict) -> dict:
    fits = el.fit_model_sequence(cells, n_starts=N_STARTS, random_state=random_state,
                                 models=["pooled_log", "state_log", "activity_tail"])
    for k, model in enumerate(("surface_volume", "surface_intensity", "surface_volume_scaled")):
        fixed = {**shape, **scaling} if model == "surface_volume_scaled" else shape
        fits[model] = el.fit_one_model(model, cells, fits, n_starts=N_STARTS, random_state=random_state + 10 + k,
                                       fixed_params=fixed)
    return fits


def score(fit_params: dict, cells: pd.DataFrame) -> dict:
    rows = {}
    tail = (cells["imbalance_bin"] == cells["imbalance_bin"].max()).to_numpy()
    high_tail = tail & (cells["activity_bin"] == cells["activity_bin"].max()).to_numpy()
    se = cells["standard_error"].to_numpy()
    for model, params in fit_params.items():
        residual = cells["mean_impact"].to_numpy() - el.predict_model(model, params, cells)
        rows[model] = {"wrmse": el.weighted_rmse(residual, se),
                       "tail_wrmse": el.weighted_rmse(residual[tail], se[tail]),
                       "high_activity_tail_wrmse": el.weighted_rmse(residual[high_tail], se[high_tail])}
    return rows


SHAPE = common_shape()
SCALING = common_scaling()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("in-sample master-curve shape: alpha, beta =", [round(float(np.exp(v)), 4) for v in SHAPE.values()],
          "scaling:", {k: round(v, 4) for k, v in SCALING.items()}, flush=True)
    surface_kwargs = dict(extra_columns=EXTRA)
    samples = analysis_samples()

    def designs_for(data: pd.DataFrame) -> list:
        days = np.asarray(sorted(data["block_day"].dropna().unique()))
        designs = [("blocked", k, block) for k, block in enumerate(np.array_split(days, 5))]
        years = pd.DatetimeIndex(days).year
        return designs + [("expanding_year", year, days[years == year]) for year in TEST_YEARS]

    def day(value) -> np.datetime64:
        return np.datetime64(pd.Timestamp(value).date(), "D")

    folds = {}
    for data in samples.values():
        for design, label, test_days in designs_for(data):
            span = (day(test_days[0]), day(test_days[-1]))
            if folds.setdefault((design, label), span) != span:
                raise RuntimeError(f"test periods differ between constructions for {(design, label)}")
    shapes = fold_shapes(folds)
    pd.DataFrame([row for _, _, row in shapes.values()]).to_csv(OUT / "fold_shapes.csv", index=False)

    in_sample, parameters, oos = [], [], []
    for c_number, (construction, data) in enumerate(samples.items()):
        cells, _ = el.build_response_surface(data, **surface_kwargs)
        fits = fit_all(cells, random_state=40_000 + 1_000 * c_number, shape=SHAPE, scaling=SCALING)
        for model in MODELS:
            row = el.fit_summary_row(fits[model])
            in_sample.append({"construction": construction, **row})
            parameters.append({"construction": construction, "model": model,
                               **el.transformed_parameters(model, fits[model]["params"])})
        print(construction, "in-sample done", flush=True)

        for design, label, test_days in designs_for(data):
            if design == "blocked":
                train = data.loc[~data["block_day"].isin(test_days)]
            else:
                train = data.loc[pd.DatetimeIndex(data["block_day"]).year < label]
            test = data.loc[data["block_day"].isin(test_days)]
            train_cells, _ = el.build_response_surface(train, **surface_kwargs)
            test_cells, _ = el.build_response_surface(test, **surface_kwargs)
            shape, scaling, _ = shapes[(design, label)]
            train_fits = fit_all(train_cells, random_state=50_000 + 1_000 * c_number + int(label),
                                 shape=shape, scaling=scaling)
            for model, result in score({m: f["params"] for m, f in train_fits.items()}, test_cells).items():
                oos.append({"construction": construction, "design": design, "test_block": label,
                            "test_start": pd.Timestamp(test_days[0]).date(), "test_end": pd.Timestamp(test_days[-1]).date(),
                            "model": model, "parameters": len(train_fits[model]["free"]), **result})
            print(construction, design, label, flush=True)

    pd.DataFrame(in_sample).to_csv(OUT / "in_sample.csv", index=False)
    pd.DataFrame(parameters).to_csv(OUT / "parameters.csv", index=False)
    pd.DataFrame(oos).to_csv(OUT / "out_of_sample.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(in_sample)[["construction", "model", "parameters", "weighted_rmse", "tail_wrmse",
                                   "high_activity_tail_wrmse", "bound_parameter_names"]].round(5).to_string(index=False))
    o = pd.DataFrame(oos)
    print(o.pivot_table(index=["construction", "design", "test_block"], columns="model", values="wrmse").round(5).to_string())


if __name__ == "__main__":
    main()
