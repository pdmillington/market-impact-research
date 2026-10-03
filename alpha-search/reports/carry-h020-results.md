# H-020 · Low-turnover cross-sectional crowding carry: Stage 1 results

Registered 2026-09-30 (see `HYPOTHESIS_LEDGER.md`: H-020, run notes,
outcome).
Runner: `scripts/run_carry_h020.py`. Data: `scripts/download_bybit_panel.py`
and `shared-methodology/src/crypto_market_data/bybit.py`.
Outputs: `reports/carry_h020/`.
P&L is linear-perp accounting (`src/alpha_search/perp_pnl.py`).

## Verdict: rejected at Stage 1

- Neither signal passes.
- Stages 2 (locked 2026-06..08) and 3 (prospective) are not run; the locked
  months remain unread.

| Condition | C (primary) | Funding level (S1) |
|---|---|---|
| 1. Bybit net Sharpe > 0, NW t > 2 | −0.74, t −1.53 ✗ | 0.01, t 0.03 ✗ |
| 2. Positive ex best/worst 10 days, and more than 55% of months positive | −104.6%; 38% ✗ | +1.8%; 46% ✗ |
| 3. Last 24 months Sharpe > 0 | −1.29 ✗ | −0.02 ✗ |
| 4. Binance consistency, net Sharpe > 0 | −0.24 ✗ | 0.72 ✓ (used data; not evidence) |

## Results (test periods: Bybit 2022-06 to 2026-05, Binance 2022-01 to 2026-05)

| %/yr (Sharpe) | Bybit C | Bybit funding level | Binance C | Binance funding level |
|---|---:|---:|---:|---:|
| Gross | +20.5 (0.55) | +18.8 (0.49) | +35.3 (0.90) | +41.6 (1.12) |
| Net, 3 bps | −1.7 (−0.05) | +10.3 (0.27) | +12.9 (0.33) | +34.2 (0.92) |
| Net, gate | −27.6 (−0.74) | +0.5 (0.01) | −9.6 (−0.24) | +26.7 (0.72) |
| Price, long / short | +23.7 / −13.1 | +2.4 / −9.2 | −14.3 / +38.4 | −16.0 / +28.6 |
| Funding, long / short | +3.6 / +6.3 | +13.7 / +11.8 | +7.3 / +3.9 | +18.9 / +10.0 |
| Turnover per day | 2.03 | 0.77 | 2.05 | 0.68 |
| Holding days (mean / median) | 3.0 / 2 | 7.8 / 4 | 3.0 / 2 | 8.9 / 5 |

Gate Sharpe by year, Bybit:

| | 2022 H2 | 2023 | 2024 | 2025 | 2026 YTD |
|---|---:|---:|---:|---:|---:|
| C | 0.43 | 0.98 | −2.47 | −1.03 | −0.99 |
| Funding level | 0.99 | −0.41 | −0.46 | 0.56 | 0.02 |

Capacity is not binding: 95th-percentile daily participation is below 2% of a
name's traded value at a $1m book.

## Interpretation

- **C still churns.** Daily hysteresis only cut turnover from 2.9 to
  2.0 per day. The 24-hour return and premium components move a lot from day
  to day, so the costs remain fatal.
- **Funding is reliable income:** about +25%/yr on Bybit and +29%/yr on
  Binance on 2 units of gross, with NW t between 9 and 22 on every funding
  leg.
- **The price legs are noise.** With t < 1.1 throughout, their sign differs
  by venue and start date: Binance's test includes the H1 2022 crash, which
  paid the short book. A dollar-neutral long/short alt book carries about
  40%/yr of price volatility. That swamps the carry, giving a gross Sharpe
  of about 0.5 on the test venue.
- **Implication:** the edge is the funding itself, and the problem is
  unhedged alt price risk. Harvesting funding without that risk means
  pairing each perp with the same coin's spot (cash-and-carry). That is a
  different trade, arbitrage-like and limited by spot borrow, custody and
  capacity. It would need its own economic case and registration.
