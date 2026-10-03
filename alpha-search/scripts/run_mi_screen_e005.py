"""E-005: mutual-information screen of hourly execution signatures (development window only).

Registered protocol (HYPOTHESIS_LEDGER.md, E-005 and H-013). Implementation choices
fixed before any return was read (recorded in the ledger build notes):

- feature window W equals the horizon H (8 h features -> 8 h targets; 24 -> 24);
- decisions are non-overlapping: every 8 h (00/08/16 UTC) or daily (00 UTC);
- development window: decisions with T >= 2021-01-01 and T + H <= 2024-01-01;
- conditioning set Z = {source flow_imbalance_W, source own return_W}, plus the
  target's own past return_W when the target differs from the source; the
  benchmark feature flow_imbalance is screened marginally (it is in Z);
- estimator: Gaussian-copula (conditional) MI in bits (Ince et al. 2017); the
  sign target uses the Gaussian-mixture form for a discrete variable; zero
  returns are dropped;
- null: 1,000 circular shifts of the forward return against all predictors
  (feature and Z move together), shifts of at least 30 days; p = (1 + #null >=
  observed) / 1001; excess = observed - mean(null);
- BH at q = 0.10 over the 6 signatures x 16 source/target pairs x 2 horizons
  (the benchmark flow_imbalance cells are reported but excluded from the family);
- the crowding side table is a separate BH family and cannot feed the selection.

The selection rule for H-013 is applied mechanically and written to selection.json.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
from scipy.special import ndtri
from scipy.stats import rankdata


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "scripts"))
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

import build_signature_features as sig  # noqa: E402
from crypto_market_data.series import DATASETS  # noqa: E402


HOUR_MS = 3_600_000
DEV_START = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1_000)
DEV_END = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1_000)
HORIZONS = {"8h": 8, "24h": 24}
SIGNATURES = ("ml_share", "ml_imbalance", "sweep_pressure", "absorption_imbalance", "burst_skew", "large_imbalance")
BENCHMARK = "flow_imbalance"
CROWDING = ("funding_z30d", "premium_8h", "basis_1h", "oi_chg_24h", "toptrader_ls_z30d", "global_ls_z30d", "taker_ls_1h")
INSTRUMENTS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
TARGETS = INSTRUMENTS + ("ALT",)
N_SHIFTS, MIN_SHIFT_DAYS, SEED = 1_000, 30, 20261001
Q_BH, P_INCREMENT, MAX_SELECTED = 0.10, 0.05, 3
LN2 = np.log(2.0)


# ---------------------------------------------------------------- estimators


def copnorm(x: np.ndarray) -> np.ndarray:
    """Rank-transform each column to standard normal scores."""

    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    return ndtri(rankdata(x, axis=0) / (x.shape[0] + 1))


def gauss_entropy(v: np.ndarray) -> float:
    """Differential entropy (nats) of a Gaussian with the sample covariance of the columns of v."""

    d = v.shape[1]
    cov = np.atleast_2d(np.cov(v, rowvar=False))
    sign, logdet = np.linalg.slogdet(cov)
    return 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)


def mi_continuous(x: np.ndarray, y: np.ndarray, z: np.ndarray | None) -> float:
    """I(X; Y | Z) in bits for copula-normalised continuous columns."""

    if z is None or z.shape[1] == 0:
        return (gauss_entropy(x) + gauss_entropy(y) - gauss_entropy(np.hstack([x, y]))) / LN2
    return (gauss_entropy(np.hstack([x, z])) + gauss_entropy(np.hstack([y, z])) - gauss_entropy(z)
            - gauss_entropy(np.hstack([x, y, z]))) / LN2


def mi_discrete(x: np.ndarray, y: np.ndarray, z: np.ndarray | None) -> float:
    """I(X; Y | Z) in bits, Y discrete (Gaussian mixture form)."""

    xz = x if z is None or z.shape[1] == 0 else np.hstack([x, z])
    total = gauss_entropy(xz) - (0.0 if xz is x else gauss_entropy(z))
    conditional = 0.0
    for value in np.unique(y):
        mask = y == value
        part = gauss_entropy(xz[mask]) - (0.0 if xz is x else gauss_entropy(z[mask]))
        conditional += mask.mean() * part
    return (total - conditional) / LN2


# ---------------------------------------------------------------- data


def decision_grid(horizon: int) -> np.ndarray:
    times = np.arange(DEV_START, DEV_END - horizon * HOUR_MS + 1, horizon * HOUR_MS, dtype=np.int64)
    return times


def instrument_series(root: Path, symbol: str, label: str, horizon: int, times: np.ndarray) -> dict:
    panel = pl.read_parquet(sig.panel_path(root, symbol))
    index = {int(t): i for i, t in enumerate(panel["decision_ms"].to_numpy())}
    rows = np.array([index.get(int(t), -1) for t in times])
    later = np.array([index.get(int(t + horizon * HOUR_MS), -1) for t in times])
    take = lambda col, r: np.where(r >= 0, panel[col].to_numpy()[np.clip(r, 0, None)], np.nan)  # noqa: E731
    out = {f: take(f"{f}_{label}_z", rows) for f in SIGNATURES + (BENCHMARK, "own_return")}
    close_now, close_later = take("close", rows), take("close", later)
    out["forward"] = close_later / close_now - 1.0
    return out


def crowding_series(root: Path, symbol: str, times: np.ndarray) -> dict:
    suffix = "" if symbol == "BTCUSDT" else f"_{symbol}"
    panel = pl.read_parquet(REPO / "alpha-search" / "reports" / f"funding_positioning_v1{suffix}" / "hourly_panel.parquet")
    t = panel["time_s"].to_numpy() * 1_000
    out = {}
    for name in CROWDING:
        values = panel[name].to_numpy().astype(float)
        values = np.where(np.isfinite(values), values, np.nan)
        z = sig.rolling_z(values)
        lookup = dict(zip(t.tolist(), z))
        out[name] = np.array([lookup.get(int(x), np.nan) for x in times])
    return out


def alt_basket(root: Path, horizon: int, label: str, times: np.ndarray) -> dict:
    members = pl.read_parquet(root / "features" / "alpha" / "alt_panel_v1" / "membership.parquet")
    month = np.array([datetime.fromtimestamp(t / 1_000, tz=timezone.utc).strftime("%Y-%m") for t in times])
    forward = np.zeros(len(times))
    past = np.zeros(len(times))
    count_f = np.zeros(len(times))
    count_p = np.zeros(len(times))
    for symbol in members["symbol"].unique().to_list():
        months = set(members.filter(pl.col("symbol") == symbol)["month"].to_list())
        member = np.array([m in months for m in month])
        if not member.any():
            continue
        table = pl.read_parquet(DATASETS["perp_klines_1h"].table(root, symbol), columns=["open_time_ms", "close"])
        close = dict(zip((table["open_time_ms"].to_numpy() + HOUR_MS).tolist(), table["close"].to_numpy()))
        now = np.array([close.get(int(t), np.nan) for t in times])
        later = np.array([close.get(int(t + horizon * HOUR_MS), np.nan) for t in times])
        before = np.array([close.get(int(t - horizon * HOUR_MS), np.nan) for t in times])
        f = later / now - 1.0
        b = now / before - 1.0
        ok_f, ok_b = member & np.isfinite(f), member & np.isfinite(b)
        forward += np.where(ok_f, f, 0.0)
        count_f += ok_f
        past += np.where(ok_b, b, 0.0)
        count_p += ok_b
    return {"forward": np.where(count_f > 0, forward / np.maximum(count_f, 1), np.nan),
            "own_return": np.where(count_p > 0, past / np.maximum(count_p, 1), np.nan)}


# ---------------------------------------------------------------- screen


def shifts(n: int, min_shift: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(min_shift, n - min_shift, size=N_SHIFTS)


def cell_statistics(x: np.ndarray, z: np.ndarray | None, r: np.ndarray, min_shift: int, rng) -> dict:
    """Observed and circular-shift-null MI for direction, volatility and all dependence."""

    keep = np.isfinite(x) & np.isfinite(r) & (r != 0)
    if z is not None:
        keep &= np.all(np.isfinite(z), axis=1)
    x, r = x[keep], r[keep]
    z = None if z is None else z[keep]
    n = len(x)
    cx = copnorm(x)
    cz = None if z is None else copnorm(z)
    targets = {"direction": np.sign(r).astype(int), "volatility": copnorm(np.abs(r)), "all": copnorm(r)}

    def stat(kind: str, y):
        return mi_discrete(cx, y, cz) if kind == "direction" else mi_continuous(cx, y, cz)

    offsets = shifts(n, min_shift, rng)
    out = {"n": n, "spearman": float(np.corrcoef(rankdata(x), rankdata(r))[0, 1])}
    for kind, y in targets.items():
        observed = stat(kind, y)
        null = np.array([stat(kind, np.roll(y, k, axis=0)) for k in offsets])
        out[f"{kind}_mi"] = observed
        out[f"{kind}_excess"] = observed - null.mean()
        out[f"{kind}_p"] = (1 + np.sum(null >= observed)) / (N_SHIFTS + 1)
    deciles = np.digitize(rankdata(x) / (n + 1), np.linspace(0.1, 0.9, 9))
    out["decile_mean_bps"] = [float(1e4 * r[deciles == d].mean()) for d in range(10)]
    return out


def benjamini_hochberg(p: np.ndarray, q: float) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    passed = np.zeros(len(p), bool)
    below = np.flatnonzero(ranked <= q)
    if len(below):
        passed[order[: below.max() + 1]] = True
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    parser.add_argument("--output-dir", type=Path, default=REPO / "alpha-search" / "reports" / "e005_mi_screen")
    args = parser.parse_args()
    root, out = args.shared_data_root, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows, side_rows, data = [], [], {}
    for label, horizon in HORIZONS.items():
        times = decision_grid(horizon)
        min_shift = MIN_SHIFT_DAYS * 24 // horizon
        series = {s: instrument_series(root, s, label, horizon, times) for s in INSTRUMENTS}
        series["ALT"] = alt_basket(root, horizon, label, times)
        crowd = {s: crowding_series(root, s, times) for s in INSTRUMENTS}
        market = {f: np.nanmean([series[s][f] for s in INSTRUMENTS], axis=0) for f in SIGNATURES + (BENCHMARK, "own_return")}
        market_crowd = {f: np.nanmean([crowd[s][f] for s in INSTRUMENTS], axis=0) for f in CROWDING}
        sources = {s: series[s] for s in INSTRUMENTS} | {"MARKET": market}
        crowd_sources = crowd | {"MARKET": market_crowd}
        data[label] = (times, sources, series)
        for source, features in sources.items():
            for target in TARGETS:
                if source == target == "ALT":
                    continue
                r = series[target]["forward"]
                z_cols = [features[BENCHMARK], features["own_return"]]
                if target != source:
                    z_cols.append(series[target]["own_return"])
                z = np.column_stack(z_cols)
                for feature in SIGNATURES + (BENCHMARK,):
                    conditional = feature != BENCHMARK
                    stats = cell_statistics(features[feature], z if conditional else None, r, min_shift, rng)
                    rows.append({"horizon": label, "source": source, "target": target, "feature": feature,
                                 "pair": "own" if source == target else ("market" if source == "MARKET" else "cross"),
                                 "conditional": conditional, **stats})
                for feature in CROWDING:
                    stats = cell_statistics(crowd_sources[source][feature], z, r, min_shift, rng)
                    side_rows.append({"horizon": label, "source": source, "target": target, "feature": feature, **stats})
                print(label, source, "->", target, flush=True)

    cells = pl.DataFrame(rows)
    family = cells["conditional"].to_numpy()
    passed = np.zeros(cells.height, bool)
    passed[family] = benjamini_hochberg(cells.filter(pl.col("conditional"))["direction_p"].to_numpy(), Q_BH)
    cells = cells.with_columns(pl.Series("bh_pass", passed))
    side = pl.DataFrame(side_rows)
    side = side.with_columns(pl.Series("bh_pass", benjamini_hochberg(side["direction_p"].to_numpy(), Q_BH)))
    cells.write_parquet(out / "cells.parquet")
    side.write_parquet(out / "crowding_side_table.parquet")
    flat = lambda f: f.drop("decile_mean_bps")  # noqa: E731
    flat(cells).write_csv(out / "cells.csv")
    flat(side).write_csv(out / "crowding_side_table.csv")

    # H-013 selection rule (frozen): greedy, conditional on already-chosen features for the same target.
    candidates = cells.filter(pl.col("bh_pass")).sort("direction_excess", descending=True)
    chosen: list[dict] = []
    log = []
    if candidates.height:
        first = candidates.row(0, named=True)
        chosen.append(first)
        log.append({"step": 1, "cell": {k: first[k] for k in ("horizon", "source", "target", "feature")},
                    "direction_excess": first["direction_excess"], "direction_p": first["direction_p"]})
        while len(chosen) < MAX_SELECTED:
            used = {c["feature"] for c in chosen}
            best = None
            for cand in candidates.iter_rows(named=True):
                if cand["feature"] in used or any(all(cand[k] == c[k] for k in ("horizon", "source", "target", "feature")) for c in chosen):
                    continue
                label = cand["horizon"]
                times, sources, series = data[label]
                features = sources[cand["source"]]
                z_cols = [features[BENCHMARK], features["own_return"]]
                if cand["target"] != cand["source"]:
                    z_cols.append(series[cand["target"]]["own_return"])
                for c in chosen:
                    if c["target"] == cand["target"] and c["horizon"] == label:
                        z_cols.append(data[label][1][c["source"]][c["feature"]])
                stats = cell_statistics(features[cand["feature"]], np.column_stack(z_cols), series[cand["target"]]["forward"],
                                        MIN_SHIFT_DAYS * 24 // HORIZONS[label], rng)
                if stats["direction_p"] < P_INCREMENT and (best is None or stats["direction_excess"] > best[1]["direction_excess"]):
                    best = (cand, stats)
            if best is None:
                break
            chosen.append(best[0])
            log.append({"step": len(chosen), "cell": {k: best[0][k] for k in ("horizon", "source", "target", "feature")},
                        "incremental_direction_excess": best[1]["direction_excess"], "incremental_direction_p": best[1]["direction_p"]})
    selection = {"frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rule": "H-013 greedy (registered)",
                 "family_cells": int(family.sum()), "bh_passing_cells": int(candidates.height),
                 "selected": [{k: c[k] for k in ("horizon", "source", "target", "feature", "spearman", "direction_excess",
                                                 "direction_p", "decile_mean_bps")} for c in chosen],
                 "steps": log}
    (out / "selection.json").write_text(json.dumps(selection, indent=2))
    (out / "manifest.json").write_text(json.dumps({"dev_window": ["2021-01-01", "2024-01-01"], "n_shifts": N_SHIFTS,
                                                   "min_shift_days": MIN_SHIFT_DAYS, "seed": SEED, "q_bh": Q_BH}, indent=2))
    print(json.dumps({k: v for k, v in selection.items() if k != "steps"}, indent=2, default=str))


if __name__ == "__main__":
    main()
