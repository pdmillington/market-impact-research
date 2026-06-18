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

## Tests

```bash
pytest
```

The tests cover the trade-signing convention, fixed-window imbalance calculation, logarithmic bin summaries, and power-law regression.

## Notes For Extension

The current design keeps core calculations in small functions so the study can be extended without rewriting the pipeline. Natural next modules include order-book snapshots, impact decay after high-imbalance windows, pseudo-metaorder construction, and liquidity replenishment metrics.
