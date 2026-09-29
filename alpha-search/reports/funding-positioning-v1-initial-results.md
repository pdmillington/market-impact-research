# Funding, basis and positioning v1: initial results

## Question

Do perpetual-futures crowding measures (funding, premium and perp–spot basis) or
positioning measures (open interest and long/short ratios) predict BTCUSDT perp
returns over hours, beyond past returns, and does any forecast clear costs once
funding carry is included?

## Data

New shared datasets, all consolidated with
`crypto-market-data download-and-parse-series` and backfilled from Binance daily
archives with `backfill-series`:

| Dataset | Rows | Remaining gaps |
|---|---:|---|
| Funding settlements, 2020-01 to 2026-08 | 7,305 | none; 8-hourly throughout |
| Premium index, 1-minute | 3.51M | 201 minutes |
| Mark price, 1-minute | 3.51M | 55 minutes |
| Index price, 1-minute | 3.51M | 60 minutes |
| Spot klines, 1-minute | 3.50M | 2,325 minutes of genuine exchange downtime |

Open interest and long/short ratios come from the Step 3 `metrics` table.
Binance's archive lacks the top-trader ratio for 87% of 2022 and the taker ratio
for 35%. Those ratio features are set to 0 (neutral) when missing.

## Design

- `configs/funding_positioning_v1.json`, `scripts/run_funding_positioning.py`,
  `notebooks/FundingPositioningWorkbench.ipynb`.
- Hourly decisions from February 2021 to May 2026, using only data available at
  each UTC hour. Metrics snapshots must be 5 to 15 minutes old.
- Targets: perp returns over 1, 4, 8 and 24 hours, entered one minute after the
  decision. Carry-inclusive edge subtracts the funding the position pays.
- The feature groups and eight nested models were fixed in the config before the
  run.
- Forecasts use rolling 12-month OLS with a horizon embargo, frozen for each test
  month from January 2023 to May 2026, the same protocol as Step 1.

## Result 1: crowding predicts reversal

At 24 hours, a high premium, basis or funding rate predicts lower returns:

| Feature | IC 2021–22 | IC 2023–26 | test t |
|---|---:|---:|---:|
| premium, 8h mean | −0.162 | −0.083 | −3.4 |
| premium, 1h mean | −0.147 | −0.077 | −3.6 |
| perp–spot basis, 1h | −0.146 | −0.074 | −3.5 |
| last funding rate | −0.143 | −0.073 | −3.5 |

The sign agrees in both periods and the effect strengthens with horizon. Out of
sample, adding funding or basis features to the price baseline raises the 24-hour
IC from 0.002 to 0.044 (funding; gain t 1.9) and 0.037 (basis; gain t 2.5,
better in 76% of months).

## Result 2: positioning ratios and settlement timing do not help

- The top-trader and all-account long/short z-scores reverse sign between
  periods; for example, top-trader goes from −0.086 to +0.037 at 24 hours.
  Adding the positioning group never improves out-of-sample IC.
- The 24-hour open-interest change is weakly contrarian in the test period only.
- Returns in the hour before and after the 8-hourly settlements show no reliable
  dependence on predicted funding (all |t| < 2).

## Result 3: the crowding effect is decaying fast

Mean 24-hour IC of `premium_8h` by calendar year:

| 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (Jan–May) |
|---:|---:|---:|---:|---:|---:|
| −0.147 | −0.177 | −0.136 | −0.095 | −0.043 | −0.019 |

The model-level results agree. From January 2023 to May 2025, the 24-hour
funding, basis and combined models had OOS IC of 0.044–0.057 (t 1.7–3.6). From
June 2025 to May 2026 they have 0.011, 0.025 and −0.031, none significant.

## Result 4: costs

The most robust tail is the combined model at 24 hours, top 20% of forecasts:

| | |
|---|---:|
| Carry-inclusive edge | +20.4 bps |
| Net of 10 bps taker round trip | +10.4 bps |
| Daily-block t (overlapping holds) | 2.6 |
| Best-10-day share | 69% |
| Edge on all other days | 6.5 bps |
| Edge by year, 2023 / 2024 / 2025 / 2026 | 35.5 / 13.6 / 3.3 / 15.5 bps |

It is far less dependent on stress days than Steps 1 and 3. On ordinary days it
covers a maker round trip (4 bps) but not a taker one. Passive execution is
realistic for a 24-hour signal. However, this is the best of 96
model/horizon/tail cells, the combined model's IC has been negative since mid-2025,
and the daily t-statistic is overstated by overlapping holds.

## Gate decision

**A real but fading effect.** Perp crowding (premium, basis and funding) is the
first signal in this project with the same sign across both periods, OOS value
beyond past returns, and tail edge not dominated by a few stress days. It has
weakened every year since 2022 and is near zero over the last twelve months.
A plausible cause is more arbitrage capital in basis trades since the spot ETFs,
but the data do not identify the cause.

Recommended next steps, in order:

1. **Implementable backtest (Step 2 cost layer).** A single position rebalanced
   hourly towards a clipped forecast, rather than overlapping per-signal trades,
   with maker and taker fills, funding carry, turnover and drawdown. This removes
   the overlapping-hold inflation.
2. **Decay-aware estimation.** Shorter or exponentially weighted training windows
   and a test for a structural break around 2024, specified before the holdout.
3. **Freeze one rule and pre-register the pass/fail threshold** before opening
   June–August 2026. Given the decay, the prior expectation for the holdout
   should be modest.

## Limitations

- Perp prices come from trade prints; entry at the next minute's close ignores
  queue position for maker orders.
- The feature set is fixed and small; nothing was tuned, but 96 tail cells were
  inspected.
- Premium, basis and funding are highly correlated (0.80–0.93), so the three
  groups are not independent confirmations.
