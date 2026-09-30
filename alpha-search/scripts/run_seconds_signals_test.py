"""Registered test of H-014 (sweep toxicity), H-015 (level-depletion break) and
H-016 (burst inverted U) on BTCUSDT events. See HYPOTHESIS_LEDGER.md (common
seconds-scale protocol) and configs/seconds_signals_test_v1.json.

Primary mode uses the trade-based mid proxy over the registered test months.
`--true-mid` re-runs the main statistics with bookTicker quote mids on the
registered true-mid months that have been parsed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "scripts"))
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from crypto_market_data.book_ticker import canonical_month  # noqa: E402
from crypto_market_data.months import iter_months  # noqa: E402
from explore_seconds_signals import (  # noqa: E402
    DAY_MS, HIT_BINS, NOTIONAL_BINS, forward_fill, load_events, month_start_ms,
    trailing_count, trailing_mean_at,
)
from run_residual_decomposition import ols, spearman, t_stat  # noqa: E402


# ----------------------------------------------------------------- features


QUOTE_CHUNK_ROWS = 100_000_000


def true_mids_at(root: Path, month: str, symbol: str, queries: list[np.ndarray]) -> tuple[list[np.ndarray], int, int]:
    """Quote mid in force at each query time (last quote with time <= query).

    Busy months hold up to ~1e9 quote updates, so the month is streamed in chunks
    (sorted within each chunk) and all query arrays are answered in one pass.
    Chunks are consecutive in exchange sequence, so a query falling between two
    chunks takes the last quote of the earlier one.
    """

    scan = pl.scan_parquet(canonical_month(root, symbol, month)).select(
        "transaction_time", "update_id", "best_bid_price", "best_ask_price")
    total = scan.select(pl.len()).collect().item()
    outputs = [np.full(len(q), np.nan) for q in queries]
    first_time = last_time = None
    carry_time, carry_mid = None, None
    for offset in range(0, total, QUOTE_CHUNK_ROWS):
        chunk = scan.slice(offset, QUOTE_CHUNK_ROWS).sort("transaction_time", "update_id").collect()
        times = chunk["transaction_time"].to_numpy()
        mids = 0.5 * (chunk["best_bid_price"].to_numpy() + chunk["best_ask_price"].to_numpy())
        del chunk
        if carry_time is not None:
            times = np.concatenate([[carry_time], times])
            mids = np.concatenate([[carry_mid], mids])
        first_time = int(times[0]) if first_time is None else first_time
        chunk_end = int(times[-1])
        is_last = offset + QUOTE_CHUNK_ROWS >= total
        for query, output in zip(queries, outputs):
            # Answer queries up to the end of this chunk (or all remaining, for the last chunk).
            upper = np.searchsorted(query, np.inf if is_last else chunk_end, side="right")
            lower = np.searchsorted(query, times[0], side="left")
            if upper > lower:
                index = np.searchsorted(times, query[lower:upper], side="right") - 1
                output[lower:upper] = mids[index]
        carry_time, carry_mid = times[-1], mids[-1]
        last_time = chunk_end
    return outputs, first_time, last_time


def month_features(root: Path, month: str, taus: list[int], true_mid: bool = False,
                   symbol: str = "BTCUSDT") -> dict:
    events = load_events(root, month, symbol)
    t = events["timestamp_ms"].to_numpy()
    s = events["aggressor_sign"].to_numpy().astype(np.float64)
    qty = events["quantity"].to_numpy()
    vwap = events["vwap"].to_numpy()
    first = events["first_price"].to_numpy()
    last = events["last_price"].to_numpy()
    levels = events["price_level_count"].to_numpy()
    del events

    in_month = t >= month_start_ms(month)
    if true_mid:
        queries = [t, t - 1] + [t + tau * 1_000 for tau in taus]
        answers, q_first, q_last = true_mids_at(root, month, symbol, queries)
        mid, pre_mid = answers[0], answers[1]
        future = dict(zip(taus, answers[2:]))
        in_month &= (t >= q_first + 3_600_000) & (t <= q_last - max(taus) * 1_000)
    else:
        ask = forward_fill(last, s > 0)
        bid = forward_fill(last, s < 0)
        mid = 0.5 * (ask + bid)
        pre_mid = np.concatenate([[np.nan], mid[:-1]])
        future = {tau: mid[np.searchsorted(t, t + tau * 1_000, side="right") - 1] for tau in taus}

    features = {"t": t, "s": s, "levels": levels, "in_month": in_month,
                "notional": qty * vwap}
    features["continuation"] = {tau: s * 1e4 * np.log(future[tau] / mid) for tau in taus}
    features["markout"] = {tau: -s * 1e4 * np.log(future[tau] / vwap) for tau in taus}
    features["pre_move"] = {tau: 1e4 * np.log(future[tau] / pre_mid) for tau in taus}

    rate_24h = trailing_count(t, np.ones(len(t), bool), DAY_MS) / 86_400.0
    features["activity"] = trailing_count(t, np.ones(len(t), bool), 300_000) / 300.0 / np.maximum(rate_24h, 1e-9)

    # Resiliency (E-003): trailing mean 5-second reversion fraction of past sweeps.
    mid_5 = future[5] if 5 in future else mid[np.searchsorted(t, t + 5_000, side="right") - 1]
    impact = s * 1e4 * np.log(mid / pre_mid)
    reverted = s * 1e4 * np.log(mid / mid_5)
    usable = (levels >= 2) & (impact > 0) & np.isfinite(reverted)
    features["resiliency"] = trailing_mean_at(
        t, t[usable] + 5_000, np.clip(reverted[usable] / impact[usable], -2.0, 2.0), 1_800_000)

    # Level runs and absorption (E-003 definitions).
    same = np.zeros(len(t), bool)
    same[1:] = (s[1:] == s[:-1]) & (levels[1:] == 1) & (levels[:-1] == 1) & (first[1:] == last[:-1])
    run_id = np.cumsum(~same)
    run_start = np.flatnonzero(~same)
    features["hits"] = np.arange(len(t)) - run_start[run_id - 1] + 1
    cumulative = np.cumsum(qty)
    run_qty = cumulative - np.concatenate([[0.0], cumulative])[run_start[run_id - 1]]
    depth = {}
    for side in (1.0, -1.0):
        chosen = (levels >= 2) & (s == side)
        depth[side] = np.exp(trailing_mean_at(t, t[chosen], np.log(qty[chosen] / levels[chosen]), 1_800_000))
    features["absorption"] = run_qty / np.where(s > 0, depth[1.0], depth[-1.0])

    buys = trailing_count(t, s > 0, 5_000)
    sells = trailing_count(t, s < 0, 5_000)
    features["burst"] = (buys - sells) / np.maximum(rate_24h * 5.0, 1e-9)
    return features


def tercile(values: np.ndarray, cuts) -> np.ndarray:
    labels = np.digitize(values, cuts) + 1
    labels[~np.isfinite(values)] = 0
    return labels


# ------------------------------------------------------------ residual fit


def fit_resiliency_residual(root: Path, config: dict, path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    xs, ys = [], []
    rng = np.random.default_rng(2022)
    for month in config["residual_fit_months"]:
        f = month_features(root, month, [5])
        ok = f["in_month"] & np.isfinite(f["resiliency"]) & (f["activity"] > 0)
        chosen = rng.choice(np.flatnonzero(ok), size=min(500_000, int(ok.sum())), replace=False)
        xs.append(np.log(f["activity"][chosen]))
        ys.append(f["resiliency"][chosen])
        print(f"residual fit {month}", flush=True)
    x, y = np.concatenate(xs), np.concatenate(ys)
    design = np.column_stack([np.ones(len(x)), x])
    beta = ols(design, y)
    residual = y - design @ beta
    fit = {"intercept": float(beta[0]), "slope_log_activity": float(beta[1]),
           "residual_tercile_cuts": np.quantile(residual, [1 / 3, 2 / 3]).tolist(),
           "correlation_resiliency_log_activity": float(np.corrcoef(x, y)[0, 1])}
    path.write_text(json.dumps(fit, indent=2) + "\n")
    return fit


# --------------------------------------------------------------- statistics


def month_statistics(f: dict, config: dict, fit: dict | None, month: str) -> dict:
    cuts = config["frozen_cuts"]
    taus = config["taus"]
    s, levels, hits = f["s"], f["levels"], f["hits"]
    m = f["in_month"]
    rng = np.random.default_rng(int(month.replace("-", "")))
    sample = np.sort(rng.choice(np.flatnonzero(m), size=min(config["ic_sample_per_month"], int(m.sum())),
                                replace=False))
    row = {"month": month, "events": int(m.sum())}

    def ic(x, y, mask=None):
        index = sample if mask is None else sample[mask[sample]]
        xs, ys = x[index], y[index]
        ok = np.isfinite(xs) & np.isfinite(ys)
        return spearman(xs[ok], ys[ok])

    def mean(values, mask):
        chosen = values[mask & np.isfinite(values)]
        return float(chosen.mean()) if len(chosen) else float("nan")

    activity_state = tercile(f["activity"], cuts["activity"])
    # H-014: sweep toxicity.
    for tau in taus:
        row[f"H014_ic_tau{tau}"] = ic(s * levels, s * f["continuation"][tau])
    notional_class = np.digitize(f["notional"], NOTIONAL_BINS)
    within = [ic(s * levels, s * f["continuation"][5], notional_class == c) for c in range(5)]
    row["H014_ic_size_controlled"] = float(np.nanmean(within))
    sweep = m & (levels >= 2)
    markout5 = f["markout"][5]
    row["H014_interaction_activity"] = (mean(markout5, sweep & (activity_state == 3))
                                        - mean(markout5, sweep & (activity_state == 1)))
    if fit is not None:
        residual = f["resiliency"] - (fit["intercept"] + fit["slope_log_activity"]
                                      * np.log(np.maximum(f["activity"], 1e-12)))
        residual_state = tercile(residual, fit["residual_tercile_cuts"])
        row["H014_secondary_resiliency_orth"] = (mean(markout5, sweep & (residual_state == 3))
                                                 - mean(markout5, sweep & (residual_state == 1)))

    # H-015: level-depletion break (single-level events).
    single = levels == 1
    for tau in taus:
        row[f"H015_ic_tau{tau}"] = ic(s * hits, s * f["continuation"][tau], single)
    row["H015_ic_hits_vs_markout"] = ic(hits.astype(float), markout5, single)
    absorption_state = tercile(f["absorption"], cuts["absorption"])
    runs = m & single & (hits >= 3)
    row["H015_interaction_absorption"] = (mean(f["continuation"][5], runs & (absorption_state == 3))
                                          - mean(f["continuation"][5], runs & (absorption_state == 1)))

    # H-016: burst inverted U.
    burst = f["burst"]
    decile = np.digitize(burst, cuts["burst"]) + 1
    sign = np.sign(burst)
    moderate = m & np.isin(decile, config["burst_moderate_deciles"]) & (sign != 0)
    extreme = m & np.isin(decile, config["burst_extreme_deciles"]) & (sign != 0)
    for tau in (1, 5, 15):
        row[f"H016_A_tau{tau}"] = mean(sign * f["pre_move"][tau], moderate)
    for tau in (15, 60):
        row[f"H016_B_tau{tau}"] = mean(sign * f["pre_move"][tau], extreme)
    row["H016_interaction_activity"] = (mean(sign * f["pre_move"][5], moderate & (activity_state == 1))
                                        - mean(sign * f["pre_move"][5], moderate & (activity_state == 3)))

    # Protective value (diagnostic): maker markout of same-side fills within 5 s after a flag.
    flags = config["protective_value_flags"]
    window = int(flags["window_seconds"]) * 1_000
    for name, flag in (("H014", levels >= flags["H-014_min_levels"]),
                       ("H015", single & (hits >= flags["H-015_min_hits"]))):
        after = np.zeros(len(s), bool)
        for side in (1.0, -1.0):
            recent = trailing_count(f["t"], flag & (s == side), window) > 0
            after |= recent & (s == side)
        row[f"{name}_protective_markout_after_flag"] = mean(markout5, m & after)
        row[f"{name}_protective_markout_other"] = mean(markout5, m & ~after)
        row[f"{name}_protective_share_after_flag"] = float((m & after).sum() / m.sum())
    return row


EXPECTED_SIGN = {
    "H014_ic_tau1": 1, "H014_ic_tau5": 1, "H014_ic_tau15": 1, "H014_ic_tau60": 1,
    "H014_ic_size_controlled": 1, "H014_interaction_activity": 1, "H014_secondary_resiliency_orth": 1,
    "H015_ic_tau1": 1, "H015_ic_tau5": 1, "H015_ic_tau15": 1, "H015_ic_tau60": 1,
    "H015_ic_hits_vs_markout": -1, "H015_interaction_absorption": 1,
    "H016_A_tau1": 1, "H016_A_tau5": 1, "H016_A_tau15": 1,
    "H016_B_tau15": -1, "H016_B_tau60": -1, "H016_interaction_activity": 1,
}


def summarise(monthly: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for metric, sign in EXPECTED_SIGN.items():
        if metric not in monthly.columns:
            continue
        values = monthly[metric].to_numpy().astype(float)
        values = values[np.isfinite(values)]
        rows.append({"metric": metric, "expected_sign": sign, "months": len(values),
                     "mean": float(values.mean()), "t": t_stat(values),
                     "share_expected_sign": float(np.mean(np.sign(values) == sign))})
    return pl.DataFrame(rows)


def evaluate(summary: pl.DataFrame) -> pl.DataFrame:
    get = {r["metric"]: r for r in summary.iter_rows(named=True)}

    def passes(metric, share):
        r = get[metric]
        return bool(r["mean"] * r["expected_sign"] > 0 and r["share_expected_sign"] > share)

    def robust(prefix):
        return sum(get[f"{prefix}_ic_tau{t}"]["mean"] > 0 for t in (1, 15, 60)) >= 2

    rows = [
        ("H-014", "1 main effect (IC tau 5, >80% months)", passes("H014_ic_tau5", 0.8)),
        ("H-014", "2 main effect positive at >=2 of 3 robustness horizons", robust("H014")),
        ("H-014", "3 activity interaction (>60% months)", passes("H014_interaction_activity", 0.6)),
    ]
    if "H014_secondary_resiliency_orth" in get:
        rows.append(("H-014", "4 secondary: resiliency orthogonal to activity (>60% months)",
                     passes("H014_secondary_resiliency_orth", 0.6)))
    rows += [
        ("H-015", "1 main effect (IC tau 5, >80% months)", passes("H015_ic_tau5", 0.8)),
        ("H-015", "2 main effect positive at >=2 of 3 robustness horizons", robust("H015")),
        ("H-015", "3 absorption interaction (>60% months)", passes("H015_interaction_absorption", 0.6)),
        ("H-016", "1 primary A continuation (>70% months)", passes("H016_A_tau5", 0.7)),
        ("H-016", "2 primary B exhaustion (>60% months)", passes("H016_B_tau60", 0.6)),
        ("H-016", "3 activity interaction (>60% months)", passes("H016_interaction_activity", 0.6)),
    ]
    return pl.DataFrame(rows, schema=["hypothesis", "condition", "passes"], orient="row")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO / "alpha-search" / "configs" / "seconds_signals_test_v1.json")
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--true-mid", action="store_true")
    parser.add_argument("--symbol", default="BTCUSDT",
                        help="Confirmation runs apply the frozen BTC specification unchanged.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    out = args.output_dir or (REPO / "alpha-search" / "reports" / config["experiment_id"])
    if args.symbol != "BTCUSDT":
        out = out / f"confirmation_{args.symbol}"
    out.mkdir(parents=True, exist_ok=True)
    root = args.shared_data_root

    if args.true_mid:
        months = [m for m in config["true_mid_months"] if canonical_month(root, args.symbol, m).exists()]
        fit, label = None, "true_mid"
    else:
        months = [m for m in iter_months(config["test_start_month"], config["test_end_month"])
                  if m not in config["excluded_months"]]
        btc_out = args.output_dir or (REPO / "alpha-search" / "reports" / config["experiment_id"])
        fit = fit_resiliency_residual(root, config, btc_out / "resiliency_residual_fit.json")
        label = "proxy"
    monthly_path = out / f"monthly_{label}.parquet"
    done = pl.read_parquet(monthly_path) if monthly_path.exists() else None
    rows = [] if done is None else done.to_dicts()
    finished = {r["month"] for r in rows}
    for month in months:
        if month in finished:
            continue
        rows.append(month_statistics(month_features(root, month, config["taus"], args.true_mid, args.symbol),
                                     config, fit, month))
        pl.DataFrame(rows).write_parquet(monthly_path)  # checkpoint after every month
        print(f"{label} {month} done", flush=True)
    monthly = pl.DataFrame(rows).sort("month")
    summary = summarise(monthly)
    summary.write_csv(out / f"summary_{label}.csv")
    if not args.true_mid:
        evaluate(summary).write_csv(out / "evaluation.csv")
    manifest = {"experiment_id": config["experiment_id"], "symbol": args.symbol, "mode": label, "months": months,
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / f"run_manifest_{label}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out, flush=True)


if __name__ == "__main__":
    main()
