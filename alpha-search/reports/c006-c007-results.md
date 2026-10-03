# C-006 and C-007 · Confirmation on ETHUSDT and SOLUSDT: results

Registered 2026-10-01 (see `HYPOTHESIS_LEDGER.md`: C-006 and C-007, run
notes, outcomes). Objective and fees: the revised objective (T(5 min) =
2 bps gross) and target fees (4 bp round trip).

## C-006 · 5-minute pretrend reversal (top 1% tail): CONFIRMED

| | ETH | SOL | BTC (reference) |
|---|---:|---:|---:|
| Trades per day | 28.4 | 18.3 | 27.6 |
| Gross per trade (simple) | 12.7 bps | 29.1 | 15.5 |
| Ordinary days (excl. best 10) | 5.6 | 11.0 | 7.5 |
| Day-block t | 13.5 | 12.9 | 14.3 |
| Months positive | 71% | 83% | 83% |
| 2023–24 → 2025–26 | 16.0 → 9.8 | 60.7 → 7.1 | 19.8 → 8.4 |
| Book Sharpe at 2 bp/side | 1.16 | 1.62 | 1.58 |

- **The runner is faithful:** it reproduces H-006's BTC cell exactly.
- **Executable prices:** on real quotes (2023-05 to 2024-03), taking the
  spread on both legs costs only 0.5–3.5 bps. Net of 2 bp fees: BTC +23.0,
  ETH +18.7, SOL +65.4 bps per trade in that window.
- **Caveats:**
  - about 60% of the edge comes from the best 10 days;
  - it is decaying, most on SOL;
  - trades cluster in stress bursts, with up to about 130 overlapping;
  - it likely overlaps with stress reversion.

## C-007 · Stress reversion: not confirmed

ETH passes, SOL fails. On SOL the result is entirely 2024. It is kept as
potentially additive for BTC and ETH.

## Next

- **Locked months** 2026-06 to 2026-08 (single use, can only reject).
- **Depth and slippage** at entry for realistic size: these trades happen
  in fast markets.
- **Fee route.**
- **Prospective capture.**

## C-006 locked months (2026-06 to 2026-08): not rejected, but weak per trade

The rule was fixed before reading: reject if the pooled day-weighted mean
gross is below 2 bps, or if fewer than 2 of 3 instruments are positive.

- **Verdict:** pooled 28.8 bps (t 7.5), 3 of 3 positive, so **not rejected**.
- **Per trade (P&L at a fixed size per trade):** BTC 0.2, ETH 4.9, SOL
  28.3 bps.
- **What happened:** most signal days were small winners, but the two
  large-move days (25 June, 19 August) were heavy losers on all three. SOL's
  result is mostly one day (22 August).
- **How to read it:** with 2–3 stress days in 92, this window cannot settle
  the question. Capital decisions need prospective data and a trade-weighted
  criterion registered in advance.
