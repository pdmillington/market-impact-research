# Crypto Impact Study

Empirical market-impact research scaffold for public Binance BTCUSDT aggregate trade data, inspired by Tóth et al. (2011). The current focus is robustness testing: what is the shape of the impact function when public trade data are aggregated into time bars or volume bars?

## Scope

This is not a true metaorder study. Public Binance aggregate trade data does not include parent-order identifiers, trader identifiers, or complete order-book state. Fixed-window imbalance and volume-bar imbalance are proxies for directional trading pressure, not observed parent-order execution.

The first objective is deliberately modest: test whether a concave impact relation appears in public trade data, and whether absolute imbalance ratio is more informative than raw imbalance. Later extensions may include pseudo-metaorder detection, impact decay, order-book replenishment, resiliency, and latent-liquidity analysis.

## Repository Layout

```text
crypto-impact-study/
  data/
    raw/          # downloaded Binance zip files
    processed/    # derived tables, fit summaries, plots
  notebooks/      # exploratory notebooks
  src/crypto_impact/
  tests/
```

Raw data and processed outputs are kept separate. Paths are handled with `pathlib` and can be overridden from the command line.

## Install

Use Python 3.11 or newer. The Git repository root is `market-impact-research`, so install from that directory:

```bash
cd /Users/petermillington/Research/market-impact-research
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

After installation, this should work from the repository root:

```bash
python -m crypto_impact.pipeline --help
```

## Minimal Run

Example for BTCUSDT daily aggregate trades from January 1 through January 7, 2024, using 1-minute time bars:

```bash
python -m crypto_impact.pipeline   --symbol BTCUSDT   --start-date 2024-01-01   --end-date 2024-01-07   --sampling time   --window-length 1min   --window-step 1min
```

Rolling bars are controlled by separate length and step parameters. For example, 5-minute windows stepped every minute:

```bash
python -m crypto_impact.pipeline   --symbol BTCUSDT   --start-date 2024-01-01   --end-date 2024-01-07   --sampling time   --window-length 5min   --window-step 1min
```

Volume bars create one observation whenever cumulative traded volume reaches the requested threshold:

```bash
python -m crypto_impact.pipeline   --symbol BTCUSDT   --start-date 2024-01-01   --end-date 2024-01-07   --sampling volume   --volume-bar-size 100
```

## Methodology

Trade signing follows Binance aggregate trade semantics:

- `buyer_maker=True`: the buyer was passive, so the aggressor was seller initiated and `sign = -1`.
- `buyer_maker=False`: the aggressor was buyer initiated and `sign = +1`.

For trade `i`, the inferred aggressor sign is `epsilon_i`, and quantity is `q_i`. In each observation window or volume bar `T`:

```text
Q_T = sum(epsilon_i * q_i)
V_T = sum(q_i)
absolute_imbalance_ratio = |Q_T| / V_T
signed_impact = sign(Q_T) * log(P_end / P_start)
```

The saved observation table also includes absolute signed imbalance, gross volume, start/end prices, raw log return, absolute log return, and trade count. Audit CSV files contain the first 20 valid observations so the calculations can be checked by hand.

The analysis supports two explanatory variables:

```text
abs_signed_imbalance = |Q_T|
absolute_imbalance_ratio = |Q_T| / V_T
```

Equal-count bins are the primary methodology because each plotted point has comparable empirical support. Logarithmic bins remain available as diagnostic outputs.

For each equal-count bin the primary table reports:

```text
bin_id, mean_x, median_x, mean_impact, median_impact, impact_p10, impact_p25, impact_p75, impact_p90, n_obs
```

For absolute imbalance ratio, the main plot shows median signed impact with 10-90 and 25-75 percentile bands. This makes the distribution inside each bin visible instead of reducing each bin to a single mean.

The pipeline also creates an absolute-imbalance-ratio by gross-volume heatmap:

```text
x = absolute imbalance ratio
y = gross volume
colour = mean signed impact
```

The heatmap is intended to show whether gross volume changes signed impact even when absolute imbalance ratio is small.

## Monthly Parquet exploration

Monthly trade Parquet files use a different schema from the original Binance
aggregate-trade archives. A small exploratory command adapts that schema and
creates non-overlapping time bars, summary statistics, distribution plots, a
signed-imbalance versus return density plot, and UTC intraday profiles. For
example:

```bash
python -m crypto_impact.monthly_eda \
  crypto-impact-study/data/raw/BTCUSDT_monthly/BTCUSDT-trades-2025-01.parquet \
  --output-dir crypto-impact-study/data/processed/monthly_eda \
  --window 5min
```

The price measure is the start-to-end log return in basis points. Gross volume
and net signed volume are in BTC; the sign follows aggressor direction inferred
from `is_buyer_maker`. Distribution plots visibly label the central 99.5%
clipping used for the two signed, heavy-tailed measures.

Low-imbalance observations can be dominated by microstructure noise, including bid-offer bounce and price discreteness. The estimated exponent is sensitive to sampling, binning, and filtering choices, so percentile-band plots and heatmaps should be inspected before interpreting any regression.

## Power-Law Diagnostics

Power-law fits are retained as optional diagnostics for comparison with the market-impact literature and the square-root benchmark `delta = 0.5`:

```text
mean_impact = A * x^delta
log(mean_impact) = log(A) + delta * log(x)
```

Use:

```bash
--fit none
--fit powerlaw
--fit both
```

The current default is `--fit both`, but the fitted exponent should be interpreted cautiously. Filters are never selected automatically to improve fit quality; they must be supplied explicitly with options such as `--min-mean-impact`, `--min-obs-per-bin`, `--min-abs-signed-imbalance`, and `--min-absolute-imbalance-ratio`.

## Outputs

Large recurring analysis tables are written as compressed Parquet by default. This keeps processed runs smaller and faster to reload than CSV.

Typical outputs include:

```text
*_absolute_imbalance_ratio_equal_count_bins.parquet
*_absolute_imbalance_ratio_log_bins.parquet
*_imbalance_ratio_percentile_bands.png
*_imbalance_ratio_volume_heatmap.parquet
*_imbalance_ratio_volume_heatmap.png
*_local_slope.parquet
*_local_slope.png
*_audit_sample.csv
*_robustness_report.md
```

Use `--output-format csv` or `--output-format both` when CSV copies are needed for inspection or external tools. Audit samples remain CSV, reports remain Markdown, fit details remain JSON, and the cumulative power-law summary remains CSV. Full observation tables can be controlled separately with `--save-observations none|parquet|csv|both`.

## Tests

```bash
python -m pytest
```

The tests cover the trade-signing convention, window/bar aggregation calculations, equal-count bin summaries, heatmap summaries, local slopes, and power-law regression.

## Notes For Extension

The current design keeps core calculations in small functions so the study can be extended without rewriting the pipeline. Natural next modules include order-book snapshots, impact decay after high-imbalance observations, pseudo-metaorder construction, and liquidity replenishment metrics.
