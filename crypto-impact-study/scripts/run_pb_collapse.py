"""Run the paper's P&B scale-collapse analysis on the corrected event blocks.

Same method as notebooks/PatzeltBouchaudScaling.ipynb (via crypto_impact.pb_collapse):
signed block volume / trailing 24 h volume, 60 equal-count bins per side, a
shared P&B master curve with free per-N scales, and log-log scaling exponents
for the fitted scales and the Hurst-style standard-deviation scalings.

Blocks come from scripts/build_pb_blocks.py (shared timestamp_direction_v1
events). The trailing-volume warm-up is measured from the start of the period,
as in the paper.

    python crypto-impact-study/scripts/run_pb_collapse.py --start 2025-01 --end 2026-05 --tag bridge_2025_01_2026_05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "crypto-impact-study" / "src"))
sys.path.insert(0, str(REPO / "crypto-impact-study" / "scripts"))
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from build_pb_blocks import N_VALUES, output_path  # noqa: E402
from crypto_impact.pb_collapse import (  # noqa: E402
    add_trailing_volume_normalisation, collapse_diagnostics, fit_pb_master_curve,
    independently_binned_signed_response, scaling_slope,
)
from crypto_market_data.months import iter_months  # noqa: E402


def load_blocks(root: Path, symbol: str, n: int, start: str, end: str) -> pd.DataFrame:
    paths = [output_path(root, symbol, n, m) for m in iter_months(start, end)]
    return pl.concat([pl.read_parquet(p) for p in paths]).to_pandas()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--n-starts", type=int, default=12)
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    args = parser.parse_args()
    if args.end > "2026-08":
        raise SystemExit("Months after 2026-08 are reserved for the alpha-search P-006 test.")
    out = REPO / "crypto-impact-study" / "reports" / "redo" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    sample_rows, curves, empirical = [], [], []
    for n in N_VALUES:
        blocks = add_trailing_volume_normalisation(load_blocks(args.shared_data_root, args.symbol, n, args.start, args.end))
        curve = independently_binned_signed_response(blocks, "pb_signed_volume", n_bins=args.bins)
        curve.insert(0, "n_events", n)
        curves.append(curve)
        sample_rows.append({"n_events": n, "blocks": len(blocks),
                            "utc_days": int((blocks["start_time_ms"] // 86_400_000).nunique()),
                            "median_duration_seconds": blocks["duration_seconds"].median(),
                            "valid_trailing_volume_blocks": int(blocks["pb_signed_volume"].notna().sum()),
                            "median_trailing_24h_volume": blocks["trailing_volume"].median(),
                            "median_abs_trailing_volume_fraction": blocks["pb_signed_volume"].abs().median()})
        empirical.append({"n_events": n, "trailing_volume_fraction_std": blocks["pb_signed_volume"].std(),
                          "signed_volume_std": blocks["signed_volume"].std(), "sign_sum_std": blocks["sign_sum"].std(),
                          "return_std": blocks["log_return"].std()})
        print(f"N={n}: {len(blocks):,} blocks", flush=True)
    curves = pd.concat(curves, ignore_index=True)
    fit = fit_pb_master_curve(curves, N_VALUES, n_starts=args.n_starts)
    scales = fit["scales"].merge(pd.DataFrame(empirical), on="n_events")
    exponents = pd.DataFrame([
        scaling_slope(scales, "fitted_q_scale", "xi: fitted response width"),
        scaling_slope(scales, "fitted_r_scale", "psi: fitted response height"),
        scaling_slope(scales, "trailing_volume_fraction_std", "H_q,24h: trailing-normalised signed volume"),
        scaling_slope(scales, "signed_volume_std", "H_q,raw: signed volume"),
        scaling_slope(scales, "sign_sum_std", "H_epsilon: order signs"),
        scaling_slope(scales, "return_std", "H_r: returns"),
    ])
    collapsed, diagnostics = collapse_diagnostics(curves, fit)
    master = {"alpha": fit["alpha"], "beta": fit["beta"], "robust_objective": float(fit["objective"]),
              "converged": bool(fit["result"].success), "period": [args.start, args.end], "symbol": args.symbol}
    pd.DataFrame(sample_rows).to_csv(out / "pb_block_sample_summary.csv", index=False)
    curves.to_csv(out / "pb_volume_response_curves.csv", index=False)
    scales.to_csv(out / "pb_scale_estimates.csv", index=False)
    exponents.to_csv(out / "pb_scaling_exponents.csv", index=False)
    collapsed.to_csv(out / "pb_collapsed_volume_curves.csv", index=False)
    diagnostics.to_csv(out / "pb_collapse_diagnostics.csv", index=False)
    (out / "pb_master_curve_parameters.json").write_text(json.dumps(master, indent=2))
    print(json.dumps(master, indent=2))
    print(exponents[["quantity", "robust_exponent", "ols_exponent"]].to_string(index=False))


if __name__ == "__main__":
    main()
