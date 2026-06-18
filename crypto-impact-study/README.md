# Crypto Impact Study

Empirical market-impact research scaffold for public Binance BTCUSDT aggregate trade data. The first analysis estimates whether a concave relation appears between fixed-window signed volume imbalance and subsequent within-window price impact, inspired by Tóth et al. (2011).

## Scope

This is not a true metaorder study. Public Binance aggregate trade data does not include parent-order identifiers, trader identifiers, or complete order-book state. The fixed-window imbalance used here is only a proxy for directional trading pressure.

The first objective is deliberately modest: test whether a concave impact relation appears in public trade data. Later extensions may include pseudo-metaorder detection, impact decay, order-book replenishment, resiliency, and latent-liquidity analysis.

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

Use Python 3.11 or newer.

```bash
cd crypto-impact-study
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run The First Analysis

Example for BTCUSDT daily aggregate trades from January 1 through January 7, 2024, using 1-minute windows:

```bash
python -m crypto_impact.pipeline \
  --symbol BTCUSDT \
  --start-date 2024-01-01 \
  --end-date 2024-01-07 \
  --window 1min
```

Try 5-minute windows by changing `--window 5min`.

The pipeline will:

1. Download Binance public aggregate trade zip files from `data.binance.vision`.
2. Load compressed CSV files into pandas.
3. Infer trade sign from `buyer_maker`.
4. Compute signed volume as `sign * quantity`.
5. Resample trades into fixed windows.
6. Compute signed imbalance, absolute volume, start price, end price, log return, and signed impact.
7. Filter out zero-imbalance windows.
8. Bin observations by absolute imbalance using logarithmic bins.
9. Fit `impact = A * Q^delta` with log-log regression on binned observations.
10. Save processed CSV files, fit JSON, and a matplotlib plot.

Outputs are written to `data/processed/`, for example:

```text
BTCUSDT_2024-01-01_2024-01-07_1min_windows.csv
BTCUSDT_2024-01-01_2024-01-07_1min_binned.csv
BTCUSDT_2024-01-01_2024-01-07_1min_fit.json
BTCUSDT_2024-01-01_2024-01-07_1min_impact.png
```

## Methodology

Trade signing follows Binance aggregate trade semantics:

- `buyer_maker=True`: the buyer was passive, so the aggressor was seller initiated and `sign = -1`.
- `buyer_maker=False`: the aggressor was buyer initiated and `sign = +1`.

For each fixed time window, the pipeline computes:

```text
Q_T = sum(sign * quantity)
signed impact = sign(Q_T) * log(end_price / start_price)
```

Observations are binned by `abs(Q_T)` using logarithmic bins. Each bin reports mean absolute imbalance, mean signed impact, standard error, and observation count. The fitted power law is estimated from bins with positive mean signed impact.


## Extended Methodology

This repository implements fixed-window imbalance analysis, not true metaorder analysis. Public Binance aggregate trade data do not identify parent orders or traders, so the analysis treats short calendar-time windows as a proxy for directional order-flow pressure.

For trade `i`, the inferred aggressor sign is `epsilon_i`, and quantity is `q_i`. In each window `T`:

```text
Q_T = sum(epsilon_i * q_i)
V_T = sum(q_i)
participation_ratio = |Q_T| / V_T
signed_impact = sign(Q_T) * log(P_end / P_start)
```

The saved window table also includes absolute signed imbalance, gross volume, start/end prices, raw log return, absolute log return, and trade count. Audit CSV files contain the first 20 valid windows so the calculations can be checked by hand.

The analysis now supports two explanatory variables:

```text
abs_signed_imbalance = |Q_T|
participation_ratio = |Q_T| / V_T
```

Both can be summarized with logarithmic bins and equal-count bins. Regressions can be filtered by minimum absolute signed imbalance, minimum signed impact, minimum participation ratio, and minimum observations per bin.

Low-imbalance bins can be dominated by microstructure noise, including bid-offer bounce and price discreteness. Trade-only Roll-spread estimates or minimum-impact filters can be used to remove this noise floor before fitting a power law. The estimated exponent is sensitive to binning and filtering choices, so the raw/bin overlay plots should be inspected before interpreting any regression.

## Tests

```bash
pytest
```

The tests cover the trade-signing convention, fixed-window imbalance calculation, logarithmic bin summaries, and power-law regression.

## Notes For Extension

The current design keeps core calculations in small functions so the study can be extended without rewriting the pipeline. Natural next modules include order-book snapshots, impact decay after high-imbalance windows, pseudo-metaorder construction, and liquidity replenishment metrics.

## Robustness And Shape Analysis

The current research focus is the shape of the impact function, not automatic exponent optimisation. Equal-count bins are now the primary methodology because they give each point on the binned curve comparable empirical support. Logarithmic bins remain available as diagnostic outputs.

Supported sampling modes:

```bash
# Non-overlapping 5 minute bars
python -m crypto_impact.pipeline --symbol BTCUSDT --start-date 2024-01-01 --end-date 2024-01-07 --sampling time --window-length 5min --window-step 5min

# Rolling 5 minute bars stepped every minute
python -m crypto_impact.pipeline --symbol BTCUSDT --start-date 2024-01-01 --end-date 2024-01-07 --sampling time --window-length 5min --window-step 1min

# Volume bars, e.g. one observation when cumulative volume reaches 100 BTC
python -m crypto_impact.pipeline --symbol BTCUSDT --start-date 2024-01-01 --end-date 2024-01-07 --sampling volume --volume-bar-size 100
```

Each analysis is run for both `abs_signed_imbalance` and `participation_ratio`. The primary CSV is the equal-count bin file, with columns:

```text
bin_id, mean_x, median_x, mean_impact, median_impact, stderr, ci_lower, ci_upper, n_obs
```

The pipeline also saves LOWESS curves, local log-log slope diagnostics, shape plots, local-slope plots, audit samples, and a markdown robustness report describing sampling choices, LOWESS settings, filters, and observation counts.

Power-law fits are optional diagnostics for comparison with the market-impact literature and the square-root benchmark `delta = 0.5`. They are controlled by:

```bash
--fit none
--fit powerlaw
--fit both
```

The current default is `--fit both`, but the fitted exponent should be interpreted cautiously. It may depend on sampling method, binning, filters, and the fact that fixed-window imbalance is not a true metaorder measure. Filters are never selected automatically to improve fit quality; they must be supplied explicitly with options such as `--min-mean-impact`, `--min-obs-per-bin`, `--min-abs-signed-imbalance`, and `--min-participation-ratio`.

## Output Storage Policy

Large recurring analysis tables are written as compressed Parquet by default. This keeps processed runs smaller and faster to reload than CSV.

```bash
python -m crypto_impact.pipeline \
  --symbol BTCUSDT \
  --start-date 2024-01-01 \
  --end-date 2024-01-07 \
  --sampling time \
  --window-length 5min \
  --window-step 1min \
  --output-format parquet \
  --save-observations none
```

Use `--output-format csv` or `--output-format both` when CSV copies are needed for inspection or external tools. Audit samples remain CSV, reports remain Markdown, fit details remain JSON, and the cumulative power-law summary remains CSV. Full observation tables can be controlled separately with `--save-observations none|parquet|csv|both`.
