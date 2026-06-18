"""Command-line pipeline for robustness-focused impact-shape analysis."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import pandas as pd

from crypto_impact.binning import PRIMARY_BIN_COLUMNS, bin_by_abs_signed_imbalance, bin_by_participation_ratio
from crypto_impact.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from crypto_impact.data_download import download_agg_trades
from crypto_impact.data_load import load_agg_trade_files
from crypto_impact.impact import prepare_impact_observations
from crypto_impact.imbalance import make_audit_sample
from crypto_impact.plotting import plot_local_slopes, plot_shape_overlay
from crypto_impact.regression import RegressionFilters, fit_power_law, fit_to_summary_row
from crypto_impact.sampling import aggregate_time_bars, aggregate_volume_bars
from crypto_impact.shape import compute_local_loglog_slopes, compute_lowess_curve
from crypto_impact.trade_signing import add_signed_volume

LOGGER = logging.getLogger(__name__)
FitMode = Literal["none", "powerlaw", "both"]
OutputFormat = Literal["parquet", "csv", "both"]
SaveObservations = Literal["none", "parquet", "csv", "both"]


def _table_path(processed_dir: Path, run_name: str, suffix: str, output_format: OutputFormat) -> Path:
    """Return the primary path for a tabular output."""

    extension = "parquet" if output_format in {"parquet", "both"} else "csv"
    return processed_dir / f"{run_name}_{suffix}.{extension}"


def _write_table(df: pd.DataFrame, path: Path, output_format: OutputFormat, *, index: bool = False) -> None:
    """Write a table as Parquet, CSV, or both."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format in {"parquet", "both"}:
        parquet_path = path.with_suffix(".parquet")
        df.to_parquet(parquet_path, index=index, compression="zstd")
    if output_format in {"csv", "both"}:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=index)


def configure_logging() -> None:
    """Configure basic console logging."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def build_run_name(symbol: str, start_date: str, end_date: str, sampling: str, window_length: str | None, window_step: str | None, volume_bar_size: float | None) -> str:
    """Create a stable output prefix for a run."""

    if sampling == "volume":
        size = str(volume_bar_size).replace(".", "p")
        return f"{symbol}_{start_date}_{end_date}_volume_{size}btc"
    step = window_step or window_length
    return f"{symbol}_{start_date}_{end_date}_time_{window_length}_step_{step}"


def _fit_if_requested(binned: pd.DataFrame, filters: RegressionFilters, fit_mode: FitMode):
    """Fit power law only when requested and feasible."""

    if fit_mode == "none":
        return None
    try:
        return fit_power_law(binned, q_col="mean_x", impact_col="mean_impact", filters=filters)
    except ValueError:
        LOGGER.warning("Skipping power-law diagnostic; fewer than two bins survived filters")
        return None


def _write_report(
    path: Path,
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    sampling: str,
    window_length: str | None,
    window_step: str | None,
    volume_bar_size: float | None,
    n_observations: int,
    n_bins: int,
    lowess_frac: float,
    lowess_max_points: int,
    filters: RegressionFilters,
    fit_mode: FitMode,
) -> None:
    """Write a reproducibility report for the run."""

    text = f"""# Impact Robustness Run Report

- symbol: `{symbol}`
- date range: `{start_date}` to `{end_date}`
- sampling method: `{sampling}`
- window length: `{window_length}`
- window step: `{window_step}`
- volume bar size: `{volume_bar_size}`
- number of observations: `{n_observations}`
- number of bins: `{n_bins}`
- primary binning: `equal_count`
- diagnostic binning: `log`
- LOWESS frac: `{lowess_frac}`
- LOWESS max points: `{lowess_max_points}`
- fit mode: `{fit_mode}`
- filters: `{json.dumps(asdict(filters), sort_keys=True)}`

The main objective of this run is to inspect the shape of the impact function using equal-count bin averages, LOWESS smoothing, and adjacent-bin local slopes. Power-law fits, when requested, are secondary diagnostics for comparison with market-impact benchmarks such as the square-root law.
"""
    path.write_text(text, encoding="utf-8")


def append_powerlaw_summary(summary_path: Path, rows: list[dict[str, object]]) -> None:
    """Append fit rows to the cumulative power-law summary CSV."""

    if not rows:
        return
    new = pd.DataFrame(rows)
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined.to_csv(summary_path, index=False)


def run_pipeline(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    sampling: str,
    window_length: str | None,
    window_step: str | None,
    volume_bar_size: float | None,
    raw_dir: Path,
    processed_dir: Path,
    n_bins: int,
    fit_mode: FitMode,
    filters: RegressionFilters,
    lowess_frac: float,
    lowess_max_points: int,
    output_format: OutputFormat,
    save_observations: SaveObservations,
) -> dict[str, Path]:
    """Run one robustness-focused empirical impact analysis."""

    symbol = symbol.upper()
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    run_name = build_run_name(symbol, start_date, end_date, sampling, window_length, window_step, volume_bar_size)

    LOGGER.info("Starting run: %s", run_name)
    files = download_agg_trades(symbol, start_date, end_date, raw_dir)
    trades = add_signed_volume(load_agg_trade_files(files))

    if sampling == "time":
        if window_length is None:
            raise ValueError("window_length is required for time sampling.")
        bars = aggregate_time_bars(trades, window_length=window_length, window_step=window_step)
        audit_window = window_length
    elif sampling == "volume":
        if volume_bar_size is None:
            raise ValueError("volume_bar_size is required for volume sampling.")
        bars = aggregate_volume_bars(trades, volume_bar_size=volume_bar_size)
        audit_window = None
    else:
        raise ValueError("sampling must be 'time' or 'volume'.")

    observations = prepare_impact_observations(bars)
    analyses = {
        "abs_signed_imbalance": {
            "raw_x_col": "abs_signed_imbalance",
            "x_label": "Absolute signed imbalance |Q_T|",
            "equal": bin_by_abs_signed_imbalance(observations, n_bins=n_bins, method="equal_count"),
            "log": bin_by_abs_signed_imbalance(observations, n_bins=n_bins, method="log"),
        },
        "participation_ratio": {
            "raw_x_col": "participation_ratio",
            "x_label": "Participation ratio |Q_T| / V_T",
            "equal": bin_by_participation_ratio(observations, n_bins=n_bins, method="equal_count"),
            "log": bin_by_participation_ratio(observations, n_bins=n_bins, method="log"),
        },
    }

    paths: dict[str, Path] = {
        "audit": processed_dir / f"{run_name}_audit_sample.csv",
        "report": processed_dir / f"{run_name}_robustness_report.md",
    }
    if save_observations != "none":
        paths["observations"] = _table_path(processed_dir, run_name, "observations", save_observations)
        _write_table(observations, paths["observations"], save_observations, index=True)
    make_audit_sample(bars, audit_window).to_csv(paths["audit"], index=False)

    fit_rows: list[dict[str, object]] = []
    for analysis_name, payload in analyses.items():
        equal_bins = payload["equal"]
        log_bins = payload["log"]
        lowess_curve = compute_lowess_curve(observations, payload["raw_x_col"], frac=lowess_frac, max_points=lowess_max_points)
        slopes = compute_local_loglog_slopes(equal_bins, x_col="mean_x", impact_col="mean_impact")
        fit = _fit_if_requested(equal_bins, filters, fit_mode)

        paths[f"{analysis_name}_equal_count_bins"] = _table_path(processed_dir, run_name, f"{analysis_name}_equal_count_bins", output_format)
        paths[f"{analysis_name}_log_bins"] = _table_path(processed_dir, run_name, f"{analysis_name}_log_bins", output_format)
        paths[f"{analysis_name}_lowess"] = _table_path(processed_dir, run_name, f"{analysis_name}_lowess", output_format)
        paths[f"{analysis_name}_local_slope"] = _table_path(processed_dir, run_name, f"{analysis_name}_local_slope", output_format)
        paths[f"{analysis_name}_shape_plot"] = processed_dir / f"{run_name}_{analysis_name}_shape.png"
        paths[f"{analysis_name}_local_slope_plot"] = processed_dir / f"{run_name}_{analysis_name}_local_slope.png"

        _write_table(equal_bins[PRIMARY_BIN_COLUMNS], paths[f"{analysis_name}_equal_count_bins"], output_format)
        _write_table(log_bins, paths[f"{analysis_name}_log_bins"], output_format)
        _write_table(lowess_curve, paths[f"{analysis_name}_lowess"], output_format)
        _write_table(slopes, paths[f"{analysis_name}_local_slope"], output_format)
        plot_shape_overlay(
            observations,
            equal_bins,
            lowess_curve,
            paths[f"{analysis_name}_shape_plot"],
            raw_x_col=payload["raw_x_col"],
            x_label=payload["x_label"],
            title=f"{symbol} impact shape: {analysis_name}",
            fit=fit if fit_mode in {"powerlaw", "both"} else None,
        )
        plot_local_slopes(slopes, paths[f"{analysis_name}_local_slope_plot"], payload["x_label"], f"{symbol} local slopes: {analysis_name}")

        if fit is not None:
            fit_path = processed_dir / f"{run_name}_{analysis_name}_powerlaw_fit.json"
            fit_payload = fit_to_summary_row(
                fit,
                x_variable=analysis_name,
                sampling_method=sampling,
                window_length=window_length,
                window_step=window_step,
            )
            fit_path.write_text(json.dumps(fit_payload, indent=2, default=str), encoding="utf-8")
            paths[f"{analysis_name}_powerlaw_fit"] = fit_path
            fit_rows.append(fit_payload | {"run_name": run_name})

    append_powerlaw_summary(processed_dir / "powerlaw_fit_summary.csv", fit_rows)
    _write_report(
        paths["report"],
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        sampling=sampling,
        window_length=window_length,
        window_step=window_step,
        volume_bar_size=volume_bar_size,
        n_observations=len(observations),
        n_bins=n_bins,
        lowess_frac=lowess_frac,
        lowess_max_points=lowess_max_points,
        filters=filters,
        fit_mode=fit_mode,
    )

    for label, path in paths.items():
        LOGGER.info("Saved %s to %s", label, path)
    return paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Robustness-focused crypto impact-shape analysis.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--sampling", choices=["time", "volume"], default="time")
    parser.add_argument("--window-length", default=None, help="Time-bar length, e.g. 5min.")
    parser.add_argument("--window-step", default=None, help="Time-bar step, e.g. 1min. Defaults to window length.")
    parser.add_argument("--window", default=None, help="Backward-compatible alias for --window-length.")
    parser.add_argument("--volume-bar-size", type=float, default=None)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--n-bins", type=int, default=12)
    parser.add_argument("--fit", choices=["none", "powerlaw", "both"], default="both")
    parser.add_argument("--lowess-frac", type=float, default=0.3)
    parser.add_argument("--lowess-max-points", type=int, default=50_000)
    parser.add_argument("--min-abs-signed-imbalance", type=float, default=None)
    parser.add_argument("--min-participation-ratio", type=float, default=None)
    parser.add_argument("--min-mean-impact", type=float, default=None)
    parser.add_argument("--min-signed-impact", type=float, default=None, help="Backward-compatible alias for --min-mean-impact.")
    parser.add_argument("--min-obs-per-bin", type=int, default=None)
    parser.add_argument("--output-format", choices=["parquet", "csv", "both"], default="parquet", help="Format for large tabular outputs.")
    parser.add_argument("--save-observations", choices=["none", "parquet", "csv", "both"], default="parquet", help="Whether/how to save full observation tables.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    configure_logging()
    args = parse_args()
    window_length = args.window_length or args.window or "1min"
    min_mean_impact = args.min_mean_impact if args.min_mean_impact is not None else args.min_signed_impact
    filters = RegressionFilters(
        min_abs_signed_imbalance=args.min_abs_signed_imbalance,
        min_mean_impact=min_mean_impact,
        min_participation_ratio=args.min_participation_ratio,
        min_obs_per_bin=args.min_obs_per_bin,
    )
    run_pipeline(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        sampling=args.sampling,
        window_length=window_length if args.sampling == "time" else None,
        window_step=args.window_step or (window_length if args.sampling == "time" else None),
        volume_bar_size=args.volume_bar_size,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        n_bins=args.n_bins,
        fit_mode=args.fit,
        filters=filters,
        lowess_frac=args.lowess_frac,
        lowess_max_points=args.lowess_max_points,
        output_format=args.output_format,
        save_observations=args.save_observations,
    )


if __name__ == "__main__":
    main()
