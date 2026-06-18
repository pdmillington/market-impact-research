"""Command-line pipeline for the first market-impact analysis."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from crypto_impact.binning import bin_impact_observations
from crypto_impact.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, PipelineConfig
from crypto_impact.data_download import download_agg_trades
from crypto_impact.data_load import load_agg_trade_files
from crypto_impact.impact import prepare_impact_observations
from crypto_impact.imbalance import compute_window_imbalance, filter_nonzero_imbalance
from crypto_impact.plotting import plot_impact_power_law
from crypto_impact.regression import fit_power_law
from crypto_impact.trade_signing import add_signed_volume


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure basic console logging."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def run_pipeline(config: PipelineConfig) -> dict[str, Path]:
    """Run the complete download-to-plot impact estimation workflow."""

    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting run: %s", config.run_name)
    files = download_agg_trades(config.symbol, config.start_date, config.end_date, config.raw_dir)
    LOGGER.info("Loading %d raw files", len(files))
    trades = load_agg_trade_files(files)

    LOGGER.info("Signing trades and computing signed volume")
    signed_trades = add_signed_volume(trades)

    LOGGER.info("Resampling trades into %s windows", config.window)
    windows = compute_window_imbalance(signed_trades, config.window)
    windows = filter_nonzero_imbalance(windows)
    observations = prepare_impact_observations(windows)

    LOGGER.info("Binning observations and fitting power law")
    binned = bin_impact_observations(observations, n_bins=config.n_bins)
    fit = fit_power_law(binned)

    windows_path = config.processed_dir / f"{config.run_name}_windows.csv"
    binned_path = config.processed_dir / f"{config.run_name}_binned.csv"
    fit_path = config.processed_dir / f"{config.run_name}_fit.json"
    plot_path = config.processed_dir / f"{config.run_name}_impact.png"

    windows.to_csv(windows_path)
    binned.to_csv(binned_path, index=False)
    fit_path.write_text(json.dumps(fit.__dict__, indent=2), encoding="utf-8")
    plot_impact_power_law(binned, fit, plot_path)

    LOGGER.info("Saved windows to %s", windows_path)
    LOGGER.info("Saved binned impact estimates to %s", binned_path)
    LOGGER.info("Saved fit summary to %s", fit_path)
    LOGGER.info("Saved plot to %s", plot_path)

    return {
        "windows": windows_path,
        "binned": binned_path,
        "fit": fit_path,
        "plot": plot_path,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Estimate empirical crypto price impact.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance spot symbol.")
    parser.add_argument("--start-date", required=True, help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Inclusive end date, YYYY-MM-DD.")
    parser.add_argument("--window", default="1min", help="Pandas resampling window, e.g. 1min or 5min.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR, help="Directory for raw zip files.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory for processed outputs.",
    )
    parser.add_argument("--n-bins", type=int, default=12, help="Number of logarithmic bins.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    configure_logging()
    args = parse_args()
    config = PipelineConfig(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        window=args.window,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        n_bins=args.n_bins,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
