# H-019 · Crowding contrarian on a pooled alt-perp panel: results

Registered 2026-09-30 (see `HYPOTHESIS_LEDGER.md`: H-019, run notes,
outcome; origin E-004).
Runner: `scripts/run_crowding_panel_h019.py`. Data: `scripts/download_alt_panel.py`.
Outputs: `reports/crowding_panel_h019/`.

## P&L correction (v2, 2026-09-30)

- The figures below the verdict table are **v1**, which computed P&L with
  log returns. A linear perp earns notional × *simple* return.
- v2 (`src/alpha_search/perp_pnl.py`: fixed quantity between rebalances,
  funding on notional at settlement, costs from drifted notional) leaves the
  verdict unchanged:

| v2 (perp P&L) | Sharpe | Return/yr | NW t |
|---|---:|---:|---:|
| Net, 6 bps (gate) | −0.04 | −0.16% | −0.08 |
| Net, 3 bps | +0.13 | +0.56% | +0.28 |
| Gross | +0.29 | +1.27% | +0.64 |

- v2 conditions:
  - alpha +0.07%/yr (t 0.04);
  - breadth 46% of 39 instruments;
  - last 24 months Sharpe +0.14;
  - S1 t 0.31;
  - S2 net −28.5%/yr (t −1.80).
- v1 files are kept as `crowding_panel_h019/v1_log_returns_*`.

## Verdict: rejected

| Condition | Result | Pass |
|---|---|:-:|
| 1. Net Sharpe (6 bps) > 0, NW t > 2 | Sharpe −0.10, t −0.22 | ✗ |
| 2. Alpha vs equal-weighted members > 0, t > 2 | +0.69%/yr, t 0.35 | ✗ |
| 3. More than 55% of 12-month-plus instruments positive | 43.6% of 39 | ✗ |
| 4. Last 24 months Sharpe > 0 | +0.10 | ✓ |

## Panel

- 852 candidate USDT perps, including delisted ones.
- 182 symbols were members during the test; on average 29.9 were active.
- Test: 1,612 days, 2022-01 to 2026-05. Warm-up: 2021.

## Results

| | Sharpe | Return/yr | NW t | Last 24 months Sharpe |
|---|---:|---:|---:|---:|
| Net, 6 bps (gate) | −0.10 | −0.45% | −0.22 | +0.10 |
| Net, 3 bps | +0.05 | +0.23% | +0.11 | +0.28 |
| Gross | +0.20 | +0.91% | +0.44 | +0.46 |

Sharpe by year:

| | 2022 | 2023 | 2024 | 2025 | 2026 YTD |
|---|---:|---:|---:|---:|---:|
| Net, 6 bps | −0.23 | −0.30 | −0.63 | +1.03 | −0.03 |
| Gross | −0.04 | +0.13 | −0.38 | +1.40 | +0.43 |

Secondary:

- **S1 arbitrage capacity** (low- minus high-volume half, gross): +0.66%/yr,
  t 0.59; last 24 months t 0.78. Not supported.
- **S2 cross-sectional neutral** (6 bps): Sharpe −0.94, −31%/yr, t −1.95;
  last 24 months t −2.61. Fails.
- **S3 drop-one:** Sharpe −0.11 (drop funding), −0.22 (drop premium),
  −0.15 (drop return).

## Interpretation

- Time-series crowding contrarian at 24 hours has no edge on a broad alt
  panel. Even the gross Sharpe is 0.2, and the yearly pattern is noise
  rather than decay.
- Taken with E-004, the H-009/H-010 evidence on BTC (and ETH in 2023) looks
  like a 2021–2023 phenomenon, or instrument-specific luck, not a durable
  premium. The line is closed.

## Post-hoc diagnostic (after the verdict; a lead, not a result)

S2's loss is a cost effect, not a wrong ranking. Its decomposition is now
saved reproducibly: `results.json` → `S2_decomposition`, with the daily
series in `s2_decomposition_daily.parquet`.

| S2 leg, test period | v1 (log returns) | v2 (perp P&L) | v2 NW t |
|---|---:|---:|---:|
| Price, long | −57.6%/yr | −4.1%/yr | −0.13 |
| Price, short | +72.3%/yr | +24.0%/yr | 0.74 |
| Funding, long | | +9.8%/yr | 9.07 |
| Funding, short | | +4.7%/yr | 11.08 |
| Gross | +29.2%/yr | +34.4%/yr | 2.17 |
| Costs at 6 bps (turnover 2.87×/day) | | −62.9%/yr | |

- Log returns biased each leg by about ±50%/yr, because the held names have
  110–125% annual volatility. The two biases largely cancelled in the net,
  but the v1 leg split was badly wrong.
- The funding income is highly reliable. The price leg is not significant.
- The low-turnover follow-up is H-020 (rejected at Stage 1; see
  `carry-h020-results.md`).
