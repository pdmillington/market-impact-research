"""Paper tables for the activity-adjusted surface (revision log A3, A3b-d, A8-A11).

Reads saved outputs only (no fitting here), so every number in these tables traces to:
- reports/redo/activity_surface_n250_2021_2026/   headline fit, N = 250..16,000
  (run_activity_surface.py --tag activity_surface_n250_2021_2026 --n-values 250,...,16000,
   then diagnose_activity_surface.py with the same tag);
- reports/redo/activity_surface_2021_2026/        all-N fit including N = 100 (robustness);
- reports/redo/surface_stability/                 day-block intervals; gamma by N and by year
  (surface_identification_and_stability.py);
- reports/redo/surface_vs_threshold/              like-for-like comparison (surface_vs_threshold.py);
- reports/redo/event_price_rule/                  event price rules (event_price_rule_robustness.py);
- reports/redo/surface_concavity/                 delta within activity deciles (surface_concavity_by_activity.py).

Two error measures appear and are not comparable with each other:
- "standardised RMSE": root mean squared (residual / standard error) over the multi-N
  surface cells (crypto_impact.pb_surface.weighted_rmse); dimensionless.
- "WRMSE (x10^-3)": inverse-variance weighted RMSE in response units over the 5 x 20
  event-intensity grid of the threshold model (crypto_impact.effective_liquidity.weighted_rmse).

Writes reports/redo/paper/tables/surface_*.tex and .csv:
- surface_fit                   T1 headline and all-N fits, eta with month- and day-block CIs (complete record)
- surface_fit_headline          T1 paper table: headline N set only, month-block CI
- surface_oos_by_year           T2 expanding-window out-of-sample error by test year
- surface_vs_threshold          T3 like-for-like comparison (in sample and mean out of sample)
- surface_vs_threshold_by_block T3 appendix: out-of-sample error block by block
- surface_eta_by_n / _by_year   T4 eta stability with day-block CIs
- surface_robustness            T5 price rules, raw returns, 2019-20, N set
- surface_displacement          T5 high-activity displacement under both activity measures
- surface_power_law_fit         A3g (additional): 7-parameter power-law-scale surface and 6-parameter benchmarks
- surface_power_law_oos_by_year A3g: out-of-sample standardised RMSE by year
- surface_power_law_rolling     A3g: trailing 12-month vs expanding, 3-month tests, block-level MSE
- surface_concavity             T6 delta within relative-activity deciles vs pooled
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
REDO = REPO / "crypto-impact-study" / "reports" / "redo"
HEADLINE = REDO / "activity_surface_n250_2021_2026"
ALL_N = REDO / "activity_surface_2021_2026"
STABILITY = REDO / "surface_stability"
COMPARISON = REDO / "surface_vs_threshold"
PRICE_RULE = REDO / "event_price_rule"
CONCAVITY = REDO / "surface_concavity"
POWER_LAW = REDO / "surface_power_law"
TABLES = REDO / "paper" / "tables"

MODEL_LABEL = {"activity_surface": "Activity surface", "pb_gamma_1": r"P\&B (\(\eta=1\))",
               "imbalance_ratio_gamma_0": r"Imbalance ratio (\(\eta=0\))"}
SURFACE_FOR = {"event_bars": "surface_volume", "time_bars": "surface_volume_scaled"}
COMPARISON_LABEL = {"pooled_log": "Pooled log", "state_log": "State log", "activity_tail": "Activity tail (threshold)",
                    "surface_volume": r"Surface, \(\nu=V/V_{24h}\)", "surface_volume_scaled": r"Surface, scales in \(N_t\)",
                    "surface_intensity": r"Surface, activity \(\lambda/\lambda_{\mathrm{med}}\)"}
CONSTRUCTION_LABEL = {"event_bars": "4,000-event", "time_bars": "Five-minute"}


def num(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    text = f"{value:.{digits}f}"
    return f"$-${text[1:]}" if text.startswith("-") else text


def interval(lo, hi, digits: int = 3) -> str:
    return f"[{num(lo, digits)}, {num(hi, digits)}]"


def write(name: str, frame: pd.DataFrame, header: list[str], rows: list[list[str]], align: str,
          second_header: list[str] | None = None, breaks: set[int] = frozenset()) -> None:
    """CSV of the underlying numbers plus a booktabs tabular of the formatted rows."""

    frame.to_csv(TABLES / f"{name}.csv", index=False)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(header) + r" \\"]
    if second_header:
        lines.append(" & ".join(second_header) + r" \\")
    lines.append(r"\midrule")
    for k, row in enumerate(rows):
        if k in breaks:
            lines.append(r"\addlinespace")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (TABLES / f"{name}.tex").write_text("\n".join(lines))


def month_block(source: Path) -> tuple[float, float]:
    return tuple(np.quantile(pd.read_csv(source / "gamma_bootstrap.csv")["gamma"], [0.025, 0.975]))


def fit_table() -> None:
    """T1: the common surface and its benchmarks on the headline and the all-N sets."""

    ident = pd.read_csv(STABILITY / "identified_n_fits.csv").set_index(["n_set", "model"])
    day = pd.read_csv(STABILITY / "gamma_bootstrap.csv").set_index("n_set")
    records, rows, breaks = [], [], set()
    for n_set, source, label in (("N_250_16000", HEADLINE, "250--16,000"), ("all_N", ALL_N, "100--16,000")):
        fits = pd.read_csv(source / "fits.csv").set_index("model")
        month_lo, month_hi = month_block(source)
        if rows:
            breaks.add(len(rows))
        for model in ("activity_surface", "pb_gamma_1", "imbalance_ratio_gamma_0"):
            f, i = fits.loc[model], ident.loc[(n_set, model)]
            surface = model == "activity_surface"
            rec = {"n_set": n_set, "model": model, "gamma": f["gamma"],
                   "gamma_month_block_2_5": month_lo if surface else np.nan,
                   "gamma_month_block_97_5": month_hi if surface else np.nan,
                   "gamma_day_block_2_5": day.loc[n_set, "gamma_p025"] if surface else np.nan,
                   "gamma_day_block_97_5": day.loc[n_set, "gamma_p975"] if surface else np.nan,
                   "alpha": f["alpha"], "beta": f["beta"], "xi_identified": i["xi_identified"],
                   "psi_identified": i["psi_identified"], "identified_n": i["identified_n"],
                   "parameters": int(i["parameters"]), "cells": int(f["cells"]), "standardised_rmse": f["weighted_rmse"]}
            records.append(rec)
            rows.append([label, MODEL_LABEL[model], num(rec["gamma"]),
                         interval(month_lo, month_hi) if surface else "--",
                         interval(rec["gamma_day_block_2_5"], rec["gamma_day_block_97_5"]) if surface else "--",
                         num(rec["alpha"], 2), num(rec["beta"], 3), num(rec["xi_identified"]), num(rec["psi_identified"]),
                         str(rec["parameters"]), f"{rec['cells']:,}", num(rec["standardised_rmse"], 2)])
    write("surface_fit", pd.DataFrame(records),
          [r"\(N\)", "Model", r"\(\eta\)", "95\\% CI", "95\\% CI", r"\(\alpha\)", r"\(\beta\)", r"\(\xi\)", r"\(\psi\)",
           "Param.", "Cells", "Std."],
          rows, "llrccrrrrrrr",
          second_header=["", "", "", "(month blocks)", "(day blocks)", "", "", "", "", "", "", "RMSE"], breaks=breaks)

    # The paper's main surface table: headline N set only, month-block interval only. Same numbers as the
    # N_250_16000 rows of surface_fit (no refitting); restricted values of eta printed as integers.
    head = [r for r in records if r["n_set"] == "N_250_16000"]
    frame = pd.DataFrame([{"model": r["model"], "eta": r["gamma"], "eta_fixed": r["model"] != "activity_surface",
                           "eta_month_block_2_5": r["gamma_month_block_2_5"],
                           "eta_month_block_97_5": r["gamma_month_block_97_5"], "alpha": r["alpha"], "beta": r["beta"],
                           "xi": r["xi_identified"], "psi": r["psi_identified"], "parameters": r["parameters"],
                           "standardised_rmse": r["standardised_rmse"]} for r in head])
    rows = []
    for r in frame.itertuples():
        eta = f"{r.eta:.0f}" if r.eta_fixed else num(r.eta, 3)
        ci = "---" if r.eta_fixed else interval(r.eta_month_block_2_5, r.eta_month_block_97_5, 3)
        rows.append([MODEL_LABEL[r.model], eta, ci, num(r.alpha, 2), num(r.beta, 3), num(r.xi, 3), num(r.psi, 3),
                     str(r.parameters), num(r.standardised_rmse, 2)])
    write("surface_fit_headline", frame,
          ["Model", r"\(\hat\eta\)", "95\\% CI", r"\(\alpha\)", r"\(\beta\)", r"\(\xi\)", r"\(\psi\)",
           "Param.", "Std. RMSE"],
          rows, "lrcrrrrrr", second_header=["", "", "(month blocks)", "", "", "", "", "", ""])


def oos_table() -> None:
    """T2: train 2021 to Y-1, test Y; headline N set."""

    oos = pd.read_csv(HEADLINE / "oos.csv")
    common = oos.loc[~oos["model"].str.startswith("independent")].pivot(index="test_year", columns="model",
                                                                          values="test_weighted_rmse")
    gamma = oos.loc[oos["model"] == "activity_surface"].set_index("test_year")["gamma"]
    separate = (oos.loc[oos["model"] == "independent_N_activity_surface"].groupby("test_year")["test_weighted_rmse"]
                .apply(lambda s: float(np.sqrt(np.mean(s ** 2)))))
    table = common.assign(surface_gamma_trained=gamma, independent_N_activity_surface_rms=separate).reset_index()
    rows = [[f"{int(r.test_year)}" + (" (Jan--Aug)" if r.test_year == 2026 else ""), num(r.surface_gamma_trained),
             num(r.activity_surface, 2), num(r.pb_gamma_1, 2), num(r.imbalance_ratio_gamma_0, 2),
             num(r.independent_N_activity_surface_rms, 2)] for r in table.itertuples()]
    write("surface_oos_by_year", table,
          ["Test year", r"Trained \(\eta\)", "Activity surface", r"P\&B (\(\eta=1\))", r"\(\eta=0\)", "Separate per \\(N\\)"],
          rows, "lrrrrr")


def comparison_tables() -> None:
    """T3: like-for-like on the 5 x 20 event-intensity grid; T3 appendix block by block."""

    ins = pd.read_csv(COMPARISON / "in_sample.csv")
    oos = pd.read_csv(COMPARISON / "out_of_sample.csv")
    mean_oos = oos.groupby(["construction", "model", "design"])["wrmse"].mean().unstack("design")
    records, rows, breaks = [], [], set()
    for construction in ("event_bars", "time_bars"):
        models = ["pooled_log", "state_log", "activity_tail", SURFACE_FOR[construction], "surface_intensity"]
        if construction == "time_bars":
            models.insert(4, "surface_volume")
        if rows:
            breaks.add(len(rows))
        for model in models:
            i = ins.loc[(ins["construction"] == construction) & (ins["model"] == model)].iloc[0]
            o = mean_oos.loc[(construction, model)]
            rec = {"construction": construction, "model": model, "headline_surface": model == SURFACE_FOR[construction],
                   "free_parameters": int(i["parameters"]), "in_sample_wrmse": i["weighted_rmse"],
                   "in_sample_tail_wrmse": i["tail_wrmse"], "oos_blocked_mean_wrmse": o["blocked"],
                   "oos_expanding_year_mean_wrmse": o["expanding_year"]}
            records.append(rec)
            label = COMPARISON_LABEL[model] + (r"\(^\ast\)" if rec["headline_surface"] else "")
            rows.append([CONSTRUCTION_LABEL[construction], label, str(rec["free_parameters"]),
                         *[num(1e3 * rec[k], 2) for k in ("in_sample_wrmse", "in_sample_tail_wrmse",
                                                           "oos_blocked_mean_wrmse", "oos_expanding_year_mean_wrmse")]])
    write("surface_vs_threshold", pd.DataFrame(records),
          ["Bars", "Model", "Param.", "In-sample", "In-sample tail", "Out of sample", "Out of sample"],
          rows, "llrrrrr", second_header=["", "", "", r"(\(\times 10^{-3}\))", r"(\(\times 10^{-3}\))",
                                         r"blocked (\(\times 10^{-3}\))", r"by year (\(\times 10^{-3}\))"], breaks=breaks)

    oos["surface"] = oos["construction"].map(SURFACE_FOR)
    keep = oos.loc[(oos["model"] == oos["surface"]) | oos["model"].isin(["state_log", "activity_tail"])].copy()
    keep["model"] = np.where(keep["model"] == keep["surface"], "activity_surface", keep["model"])
    wide = keep.pivot_table(index=["construction", "design", "test_block", "test_start", "test_end"], columns="model",
                            values="wrmse").reset_index()
    rows, breaks, last = [], set(), None
    for r in wide.itertuples():
        if last is not None and (r.construction, r.design) != last:
            breaks.add(len(rows))
        last = (r.construction, r.design)
        design = "Blocked" if r.design == "blocked" else "By year"
        winner = min(("activity_surface", "activity_tail", "state_log"), key=lambda m: getattr(r, m))
        cells = [num(1e3 * getattr(r, m), 2) for m in ("activity_surface", "activity_tail", "state_log")]
        cells = [rf"\textbf{{{c}}}" if m == winner else c for c, m in zip(cells, ("activity_surface", "activity_tail", "state_log"))]
        rows.append([CONSTRUCTION_LABEL[r.construction], design, f"{r.test_start} to {r.test_end}", *cells])
    write("surface_vs_threshold_by_block", wide, ["Bars", "Design", "Test period", "Surface", "Activity tail", "State log"],
          rows, "lllrrr", second_header=["", "", "", r"(\(\times 10^{-3}\))", r"(\(\times 10^{-3}\))", r"(\(\times 10^{-3}\))"],
          breaks=breaks)


def stability_tables() -> None:
    """T4: gamma by N (each N fitted alone) and by calendar year (headline N set), day-block CIs."""

    by_n = pd.read_csv(STABILITY / "gamma_by_n.csv")
    rows = [[f"{int(r.n_events):,}" + (r"\(^\dagger\)" if r.n_events < 250 else ""), num(r.gamma),
             interval(r.gamma_p025, r.gamma_p975), num(r.alpha, 2), num(r.beta, 3)] for r in by_n.itertuples()]
    write("surface_eta_by_n", by_n, [r"\(N\)", r"\(\eta\)", "95\\% CI (day blocks)", r"\(\alpha\)", r"\(\beta\)"],
          rows, "rrcrr")
    by_year = pd.read_csv(STABILITY / "gamma_by_year.csv")
    rows = [[f"{int(r.year)}" + (" (Jan--Aug)" if r.year == 2026 else ""), f"{int(r.days)}", num(r.gamma),
             interval(r.gamma_p025, r.gamma_p975)] for r in by_year.itertuples()]
    write("surface_eta_by_year", by_year, ["Year", "Days", r"\(\eta\)", "95\\% CI (day blocks)"], rows, "lrrc")


def robustness_tables() -> None:
    """T5: gamma under alternative event prices, responses, samples and N sets; displacement."""

    records = []

    def add(spec: str, n_set: str, fits: pd.DataFrame, source: str) -> None:
        f = fits.set_index("model")
        records.append({"specification": spec, "n_set": n_set, "gamma": f.loc["activity_surface", "gamma"],
                        "alpha": f.loc["activity_surface", "alpha"], "beta": f.loc["activity_surface", "beta"],
                        "standardised_rmse": f.loc["activity_surface", "weighted_rmse"],
                        "pb_standardised_rmse": f.loc["pb_gamma_1", "weighted_rmse"], "cells": int(f.loc["activity_surface", "cells"]),
                        "source": source})

    head = pd.read_csv(HEADLINE / "fits.csv")
    add("Headline: \\(y_t\\) (volatility-normalised), 2021-01 to 2026-08", "N_250_16000", head.loc[~head["model"].str.startswith("independent")],
        "activity_surface_n250_2021_2026/fits.csv")
    rob = pd.read_csv(HEADLINE / "robustness.csv")
    add("Raw log return (no volatility scaling)", "N_250_16000", rob.loc[rob["response"] == "log_return (raw)"],
        "activity_surface_n250_2021_2026/robustness.csv")
    add("Earlier sample, 2019-10 to 2020-12", "N_250_16000", rob.loc[rob["sample"] == "2019-10..2020-12"],
        "activity_surface_n250_2021_2026/robustness.csv")
    full = pd.read_csv(ALL_N / "fits.csv")
    add("Including \\(N=100\\)", "all_N", full.loc[~full["model"].str.startswith("independent")],
        "activity_surface_2021_2026/fits.csv")
    rules = pd.read_csv(PRICE_RULE / "surface_fits.csv")
    rule_label = {"within_last": "last fill (paper rule)", "continuous_last": "last fill, continuous runs",
                  "continuous_vwap": "VWAP, continuous runs", "within_first": "first fill"}
    for rule, label in rule_label.items():
        add(f"Event price: {label}", "all_N (rebuilt)", rules.loc[rules["rule"] == rule], "event_price_rule/surface_fits.csv")
    frame = pd.DataFrame(records)
    n_label = {"N_250_16000": "250--16,000", "all_N": "100--16,000", "all_N (rebuilt)": "100--16,000\\(^\\ddagger\\)"}
    rows = [[r.specification, n_label[r.n_set], num(r.gamma), num(r.alpha, 2), num(r.beta, 3), num(r.standardised_rmse, 2),
             num(r.pb_standardised_rmse, 2)] for r in frame.itertuples()]
    write("surface_robustness", frame, ["Specification", r"\(N\)", r"\(\eta\)", r"\(\alpha\)", r"\(\beta\)",
                                        "Std. RMSE", r"P\&B std. RMSE"], rows, "llrrrrr", breaks={4})

    disp = pd.read_csv(HEADLINE / "displacement.csv")
    wide = disp.pivot_table(index=["activity_measure", "n_events"], columns="model",
                            values=["ordinary_activity", "high_activity", "difference_high_minus_ordinary"]).reset_index()
    wide.columns = ["_".join(c for c in col if c) for col in wide.columns]
    measure = {"volume_share": r"\(\nu=V/V_{24h}\)", "event_intensity": "Event intensity"}
    rows, breaks = [], set()
    for k, r in enumerate(wide.sort_values(["activity_measure", "n_events"], ascending=[False, True]).itertuples()):
        if k and r.n_events == wide["n_events"].min():
            breaks.add(k)
        rows.append([measure[r.activity_measure], f"{int(r.n_events):,}",
                     *[num(getattr(r, f"{q}_{m}"), 3) for m in ("pb_gamma_1", "activity_surface")
                       for q in ("ordinary_activity", "high_activity", "difference_high_minus_ordinary")]])
    write("surface_displacement", wide, ["Activity state", r"\(N\)", "P\\&B", "", "", "Surface", "", ""], rows, "lrrrrrrr",
          second_header=["", "", "Ordinary", "High", "Diff.", "Ordinary", "High", "Diff."], breaks=breaks)


def concavity_table() -> None:
    """T6: exponent delta in q within relative-activity deciles vs delta in x = |Q|/V_24h pooled."""

    d = pd.read_csv(CONCAVITY / "delta_by_decile.csv")
    records, rows = [], []
    for n, g in d.groupby("n_events"):
        within = g.loc[g["group"].str.startswith("decile")]
        pooled = g.loc[g["group"] == "pooled_x"].iloc[0]
        top = g.loc[g["group"] == "decile_10"].iloc[0]
        rec = {"n_events": n, "pooled_delta": pooled["delta"], "pooled_p025": pooled["delta_p025"],
               "pooled_p975": pooled["delta_p975"], "within_min": within["delta"].min(), "within_max": within["delta"].max(),
               "within_median": within["delta"].median(), "deciles_1_9_max": within.loc[within["group"] != "decile_10", "delta"].max(),
               "top_decile_delta": top["delta"], "top_decile_p025": top["delta_p025"], "top_decile_p975": top["delta_p975"],
               "share_beyond_bend": g["share_beyond_bend"].iat[0]}
        records.append(rec)
        rows.append([f"{int(n):,}", num(rec["pooled_delta"], 2), interval(rec["pooled_p025"], rec["pooled_p975"], 2),
                     num(rec["within_median"], 2), f"{num(rec['within_min'], 2)}--{num(rec['within_max'], 2)}",
                     num(rec["top_decile_delta"], 2), interval(rec["top_decile_p025"], rec["top_decile_p975"], 2),
                     num(rec["share_beyond_bend"], 2)])
    write("surface_concavity", pd.DataFrame(records),
          [r"\(N\)", r"Pooled \(\delta\)", "95\\% CI", r"Within-decile \(\delta\)", "Range", r"Top decile \(\delta\)",
           "95\\% CI", "Share beyond"],
          rows, "rrcrcrcr", second_header=["", r"(in \(|Q|/V_{24h}\))", "", "(median, in \\(q\\))", "", "", "", "bend"])


def power_law_tables() -> None:
    """A3g (additional work): scales restricted to power laws in N inside the fit (surface_power_law_scales.py)."""

    fits = pd.read_csv(POWER_LAW / "fits.csv").set_index("model")
    boot = pd.read_csv(POWER_LAW / "bootstrap.csv")
    head = pd.read_csv(HEADLINE / "fits.csv").set_index("model")
    head_exp = pd.read_csv(HEADLINE / "scaling_exponents_identified.csv").set_index("model")
    month_lo, month_hi = month_block(HEADLINE)
    p_lo, p_hi = np.quantile(boot["gamma"], [0.025, 0.975])
    specs = [  # (label, scales, parameters, eta, fixed, ci, alpha, beta, xi, psi, rmse, source)
        ("Activity surface", "free", 17, fits.loc["surface_free", "gamma"], False, (month_lo, month_hi),
         fits.loc["surface_free", "alpha"], fits.loc["surface_free", "beta"], head_exp.loc["activity_surface", "xi"],
         head_exp.loc["activity_surface", "psi"], fits.loc["surface_free", "standardised_rmse"], "surface_power_law/fits.csv"),
        ("Activity surface", "power law", 7, fits.loc["surface_power", "gamma"], False, (p_lo, p_hi),
         fits.loc["surface_power", "alpha"], fits.loc["surface_power", "beta"], fits.loc["surface_power", "xi"],
         fits.loc["surface_power", "psi"], fits.loc["surface_power", "standardised_rmse"], "surface_power_law/fits.csv"),
        (MODEL_LABEL["pb_gamma_1"], "free", 16, 1.0, True, None, head.loc["pb_gamma_1", "alpha"], head.loc["pb_gamma_1", "beta"],
         head_exp.loc["pb_gamma_1", "xi"], head_exp.loc["pb_gamma_1", "psi"], head.loc["pb_gamma_1", "weighted_rmse"],
         "activity_surface_n250_2021_2026/fits.csv"),
        (MODEL_LABEL["pb_gamma_1"], "power law", 6, 1.0, True, None, fits.loc["pb_power", "alpha"], fits.loc["pb_power", "beta"],
         fits.loc["pb_power", "xi"], fits.loc["pb_power", "psi"], fits.loc["pb_power", "standardised_rmse"], "surface_power_law/fits.csv"),
        (MODEL_LABEL["imbalance_ratio_gamma_0"], "free", 16, 0.0, True, None, head.loc["imbalance_ratio_gamma_0", "alpha"],
         head.loc["imbalance_ratio_gamma_0", "beta"], head_exp.loc["imbalance_ratio_gamma_0", "xi"],
         head_exp.loc["imbalance_ratio_gamma_0", "psi"], head.loc["imbalance_ratio_gamma_0", "weighted_rmse"],
         "activity_surface_n250_2021_2026/fits.csv"),
        (MODEL_LABEL["imbalance_ratio_gamma_0"], "power law", 6, 0.0, True, None, fits.loc["ratio_power", "alpha"],
         fits.loc["ratio_power", "beta"], fits.loc["ratio_power", "xi"], fits.loc["ratio_power", "psi"],
         fits.loc["ratio_power", "standardised_rmse"], "surface_power_law/fits.csv"),
    ]
    records, rows = [], []
    for (lab, scales, k, eta, fixed, ci, alpha, beta, xi, psi, rmse, source) in specs:
        records.append({"model": lab.replace("\\", ""), "scales": scales, "parameters": k, "eta": eta, "eta_fixed": fixed,
                        "eta_ci_2_5": np.nan if ci is None else ci[0], "eta_ci_97_5": np.nan if ci is None else ci[1],
                        "alpha": alpha, "beta": beta, "xi": xi, "xi_psi_fitted_inside": scales == "power law",
                        "psi": psi, "standardised_rmse": rmse, "source": source})
        mark = "" if scales == "power law" else r"\(^\dagger\)"
        rows.append([lab, scales, str(k), f"{eta:.0f}" if fixed else num(eta, 3),
                     "---" if ci is None else interval(ci[0], ci[1], 3), num(alpha, 2), num(beta, 3),
                     num(xi, 3) + mark, num(psi, 3) + mark, num(rmse, 2)])
    write("surface_power_law_fit", pd.DataFrame(records),
          ["Model", "Scales", "Param.", r"\(\hat\eta\)", "95\\% CI", r"\(\alpha\)", r"\(\beta\)", r"\(\xi\)", r"\(\psi\)", "Std."],
          rows, "llrrcrrrrr", second_header=["", "", "", "", "(month blocks)", "", "", "", "", "RMSE"], breaks={2, 4})

    oos = pd.read_csv(POWER_LAW / "oos_by_year.csv").pivot(index="test_year", columns="model", values="test_standardised_rmse")
    head_oos = pd.read_csv(HEADLINE / "oos.csv")
    pb_free = head_oos.loc[head_oos["model"] == "pb_gamma_1"].set_index("test_year")["test_weighted_rmse"]
    table = oos.assign(pb_free=pb_free).reset_index()
    rows = [[f"{int(r.test_year)}" + (" (Jan--Aug)" if r.test_year == 2026 else ""), num(r.surface_free, 2),
             num(r.surface_power, 2), num(r.pb_free, 2), num(r.pb_power, 2), num(r.ratio_power, 2)] for r in table.itertuples()]
    write("surface_power_law_oos_by_year", table,
          ["Test year", "Surface", "Surface", r"P\&B (\(\eta=1\))", r"P\&B (\(\eta=1\))", r"\(\eta=0\)"], rows, "lrrrrr",
          second_header=["", "free (17)", "power law (7)", "free (16)", "power law (6)", "power law (6)"])

    summary = json.loads((POWER_LAW / "rolling_summary.json").read_text())
    records, rows = [], []
    for name, res in summary["comparisons"].items():
        raw, clip = res["raw"], res["clipped"]
        records.append({"comparison": name, "months": raw["months"], "mean_D_raw": raw["mean_D"], "ci_raw_2_5": raw["ci_2_5"],
                        "ci_raw_97_5": raw["ci_97_5"], "share_months_positive_raw": raw["share_months_positive"],
                        "mean_D_clipped": clip["mean_D"], "ci_clipped_2_5": clip["ci_2_5"], "ci_clipped_97_5": clip["ci_97_5"]})
        pct = lambda v: num(100 * v, 2)
        rows.append([name, str(raw["months"]), pct(raw["mean_D"]), f"[{pct(raw['ci_2_5'])}, {pct(raw['ci_97_5'])}]",
                     f"{100 * raw['share_months_positive']:.0f}\\%", pct(clip["mean_D"]),
                     f"[{pct(clip['ci_2_5'])}, {pct(clip['ci_97_5'])}]"])
    write("surface_power_law_rolling", pd.DataFrame(records),
          ["Comparison (first vs second)", "Months", "Mean D", "95\\% CI", "Months D \\(>0\\)", "Mean D", "95\\% CI"],
          rows, "lrrcrrc", second_header=["", "", "(\\%)", "", "", "clipped (\\%)", ""])


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    fit_table()
    oos_table()
    comparison_tables()
    stability_tables()
    robustness_tables()
    concavity_table()
    power_law_tables()
    print(sorted(p.name for p in TABLES.glob("surface_*.tex")))


if __name__ == "__main__":
    main()
