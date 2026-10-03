# E-006 · Re-score of saved results under the revised objective

Objective (researcher, 2026-10-01): gross mean return per trade ≥
T(τ) = max(1.5, min(2·√(τ/5 min), 10)) bps. Target fees are 1 bp maker /
2 bps taker, so about a 4 bp round trip by any route.

Script: `scripts/run_rescore_e006.py`. It uses saved outputs only. Outputs
in `reports/e006_rescore/`: `cells.csv`, `books.csv`, `summary.csv`.

## Summary

A candidate needs ordinary-day gross ≥ T(τ), ordinary-day net > 0 and
t > 2.

| Source | Cells | Gross ≥ T(τ) | And net > 0 | Candidates |
|---|---:|---:|---:|---:|
| H-006 pretrend-reversal tails | 576 | 263 | 245 | **37** |
| H-007 stress reversion v1 | 36 | 19 | 17 | 4 |
| H-008 stress reversion v2 | 36 | 15 | 11 | 2 |
| E-002 seconds-scale buckets (2022) | 220 | 32 | 8 | 4 (tiny samples) |
| H-003 sell-side pretrend (2020–21) | 210 | 105 | 94 | 0 (no ordinary-day measure) |
| H-009 crowding tails | 96 | 65 | 65 | 0 |
| H-004, H-012 | 522 | 0 | 0 | 0 |

## The surviving effect: 5-minute pretrend reversal in the top 1% tail (BTC 2023 to 2026-05)

- Every one of the 37 H-006 candidates is the top 1% at 5–10 minutes, so
  it is one effect seen many ways.
- Example: `return_pretrend`, 1,000-event bars, 5 min:

  | Measure | Value |
  |---|---:|
  | Gross | 15.1 bps |
  | Ordinary days (excl. best 10) | 7.4 bps |
  | Day-block t | 14.4 |
  | Trades per day | 28 |

  By year: 2023 14.0, 2024 29.7, 2025 9.5, 2026 YTD 4.4 bps.
- Delayed entry (median 8–11 s) earns the same as immediate entry.
- The edge is front-loaded. Ordinary-day means across all top-1% cells:

  | Horizon | 5 min | 10 min | 15 min | 30 min |
  |---|---:|---:|---:|---:|
  | Ordinary-day mean | +4.3 | +2.0 | −2.0 | −3.7 bps |

- Weaknesses: about half the edge comes from the best 10 days, and 2026 is
  the weakest year.

## Stress reversion at its registered primary (100 episodes a year, trigger entry)

| | Hold | Gross | Ordinary | t | 2023 / 2024 / 2025 / 2026 |
|---|---|---:|---:|---:|---|
| v1 | 5 min | 9.8 | 4.9 | 2.9 | 6.7 / 32.6 / −6.6 / 9.6 |
| v2 | 5 min | 11.9 | 4.8 | 3.3 | 2.3 / 35.8 / 9.6 / 4.9 |

About 1 trade a day, heavily dependent on 2024.

## Books at the target fees

| Strategy | Gross | 2 bp/side | 1 bp/side |
|---|---|---|---|
| H-010 BTC (log-return P&L; failed C-010) | 20.4%/yr, Sharpe 0.98 | 18.0%, 0.87 | 19.2%, 0.93 |
| H-020 Bybit funding level | 18.8%, 0.49 | 13.2%, 0.34 | 16.0%, 0.41 |
| H-020 Binance funding level (used data) | 41.6%, 1.12 | 36.6%, 0.99 | 39.1%, 1.05 |

## Caveats

- **Re-scoring is not testing.** The H-006 survivors are the best of 576
  cells on data that has been examined repeatedly.
- **Confirmation needs data never used for these signals.** ETH and SOL
  5–15 minute returns have not been examined by any previous test.
