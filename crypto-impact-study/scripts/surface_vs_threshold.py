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

The surface's master-curve shape (alpha, beta) is held at the common multi-N
estimate (headline fit, activity_surface_n250_2021_2026/fits.csv): at a single bar
size the shape and the scales Q, R are not separately identified (free fits
degenerate to a power law or a straight line with alpha, beta at their bounds). Only
Q, R and gamma are local, as in the multi-N surface; the shape is estimated from
N = 250..16,000.

where q = |Q| / V. Scores: weighted RMSE, tail WRMSE (top imbalance bin of each
quintile) and high-activity tail WRMSE, in sample; and out of sample in two designs:
- blocked: the 5 contiguous day blocks used in StateDependentEffectiveLiquidity;
- expanding by year: train from 2021-01 to the end of year Y-1, test year Y
  (2022..2026), as in the surface's own out-of-sample test.
Test cells are built from the test data alone. Results are reported block by block
so that the post-2024 change (revision log A3b) is visible.

Outputs: reports/redo/surface_vs_threshold/{in_sample,parameters,out_of_sample}.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "crypto-impact-study" / "src"), str(REPO / "shared-methodology" / "src")]

from crypto_impact import effective_liquidity as el  # noqa: E402
from crypto_impact.monthly_eda import build_fixed_event_bars, build_time_bars  # noqa: E402

OUT = REPO / "crypto-impact-study" / "reports" / "redo" / "surface_vs_threshold"
DATA_DIR = REPO / "shared-data" / "features" / "academic" / "symbol=BTCUSDT" / "paper_events_v2"
START, END = "2021-01", "2026-08"
MODELS = ["pooled_log", "state_log", "activity_tail", "surface_volume", "surface_intensity", "surface_volume_scaled"]
EXTRA = ("log_q", "log_a", "log_n")
N_STARTS = 16
TEST_YEARS = (2022, 2023, 2024, 2025, 2026)
COMMON = REPO / "crypto-impact-study" / "reports" / "redo" / "activity_surface_n250_2021_2026"  # headline fit, N = 250..16,000


def common_shape() -> dict:
    fits = pd.read_csv(COMMON / "fits.csv").set_index("model")
    return {"log_alpha": float(np.log(fits.loc["activity_surface", "alpha"])),
            "log_beta": float(np.log(fits.loc["activity_surface", "beta"]))}


def common_scaling() -> dict:
    exponents = pd.read_csv(COMMON / "scaling_exponents_identified.csv").set_index("model")
    return {"xi": float(exponents.loc["activity_surface", "xi"]), "psi": float(exponents.loc["activity_surface", "psi"])}


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


def fit_all(cells: pd.DataFrame, random_state: int) -> dict:
    fits = el.fit_model_sequence(cells, n_starts=N_STARTS, random_state=random_state,
                                 models=["pooled_log", "state_log", "activity_tail"])
    for k, model in enumerate(("surface_volume", "surface_intensity", "surface_volume_scaled")):
        fixed = {**SHAPE, **SCALING} if model == "surface_volume_scaled" else SHAPE
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
    print("fixed master-curve shape:", {k: round(float(np.exp(v)), 4) for k, v in SHAPE.items()},
          "scaling:", {k: round(v, 4) for k, v in SCALING.items()}, flush=True)
    surface_kwargs = dict(extra_columns=EXTRA)
    in_sample, parameters, oos = [], [], []
    for c_number, (construction, data) in enumerate(analysis_samples().items()):
        cells, _ = el.build_response_surface(data, **surface_kwargs)
        fits = fit_all(cells, random_state=40_000 + 1_000 * c_number)
        for model in MODELS:
            row = el.fit_summary_row(fits[model])
            in_sample.append({"construction": construction, **row})
            parameters.append({"construction": construction, "model": model,
                               **el.transformed_parameters(model, fits[model]["params"])})
        print(construction, "in-sample done", flush=True)

        days = np.asarray(sorted(data["block_day"].dropna().unique()))
        designs = [("blocked", k, block) for k, block in enumerate(np.array_split(days, 5))]
        years = pd.DatetimeIndex(days).year
        designs += [("expanding_year", year, days[years == year]) for year in TEST_YEARS]
        for design, label, test_days in designs:
            if design == "blocked":
                train = data.loc[~data["block_day"].isin(test_days)]
            else:
                train = data.loc[pd.DatetimeIndex(data["block_day"]).year < label]
            test = data.loc[data["block_day"].isin(test_days)]
            train_cells, _ = el.build_response_surface(train, **surface_kwargs)
            test_cells, _ = el.build_response_surface(test, **surface_kwargs)
            train_fits = fit_all(train_cells, random_state=50_000 + 1_000 * c_number + int(label))
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
