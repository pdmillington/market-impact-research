# Crowding backtest v1: initial results

## Question

Does the Step 4 forecast make money as an implementable strategy? This means one
netted position, maker or taker fills, funding carry and real turnover, rather
than overlapping per-signal trades.

## Design

- `configs/crowding_backtest_v1.json`, `scripts/run_crowding_backtest.py` and
  `notebooks/CrowdingBacktestWorkbench.ipynb`.
- Forecasts are the Step 4 nested OLS models, refitted monthly on the preceding
  12 or 6 months with a horizon embargo.
- Signal rules:
  - `tail`: ±1 when |forecast| is in the training top 20%;
  - `continuous`: forecast divided by that threshold, clipped to ±1.
- The position is the mean of the last H hourly signals (H = 8 or 24): staggered
  tranches, each held H hours, netted into one position.
- Hourly P&L = position × next-hour perp return − position × funding settled that
  hour − fee × |position change|. Fees per side: maker 2 bps (assumes passive
  fills), taker 5 bps.
- Test period January 2023 to May 2026. "Recent" means June 2025 to May 2026.
- **Primary variant, named before the run:** combined model (`all`), 24 hours,
  tail rule, 12-month window. The `baseline` and `baseline_positioning` models
  were added after the first run, for attribution only.

## Result 1: the primary variant

| | Primary | Buy and hold (paying funding) |
|---|---:|---:|
| Return, maker fills | 18.0%/yr | 36.5%/yr |
| Sharpe, maker | 0.87 | 0.78 |
| Return, taker fills | 14.4%/yr | — |
| Sharpe, taker | 0.69 | — |
| Maximum drawdown (maker) | −26% | −70% |
| Mean absolute position | 0.27 | 1.00 |
| Beta to BTC | 0.07 | 1.00 |
| Best and worst 10 days removed | 16.4%/yr | 39.8%/yr |
| Recent 12 months, maker | +10.0%/yr, Sharpe 0.52 | — |

The strategy is close to market-neutral. It is not driven by a handful of days:
removing the best and worst ten days changes little. By calendar year (maker)
it made 46.7%, 6.8%, −0.1% and, annualised for January–May, 19.6% in 2023–2026.

## Result 2: the edge comes from long/short ratios, not crowding

| Tail rule, 12-month window, maker | 8h | 24h |
|---|---:|---:|
| baseline (past price only) | 3.9%, Sharpe 0.19 | −1.2%, −0.06 |
| + funding | 7.5%, 0.36 | 5.8%, 0.30 |
| + basis | 3.4%, 0.16 | 2.0%, 0.11 |
| + positioning | 35.5%, 1.66 | 12.9%, 0.73 |
| all | 32.7%, 1.43 | 18.0%, 0.87 |

The crowding features that carried the Step 4 IC result (funding, premium and
basis) are worth only a few percent a year once implemented. Dropping one
positioning feature at a time shows that the top-trader and all-account
long/short ratio z-scores carry almost everything. Removing all three ratio
features cuts `baseline_positioning` from 35.5% to 2.4% a year at 8 hours and
from 12.9% to −6.6% at 24 hours.

These ratios had the opposite sign in 2021–22, and Binance's archive lacks the
top-trader ratio for 87% of 2022. The rolling fit learned their current sign from
2023 data alone. A strategy resting on a feature whose sign has already flipped
once is fragile.

## Result 3: recent performance is a coin flip

- The positioning-driven variants made 30–69% a year in 2023–2024. Over the last
  twelve months they lost 5% a year (8h) and 15% a year (24h).
- The primary variant stayed positive recently (+10%/yr), helped by funding
  features that recovered in early 2026.
- Across all 40 variants, 20 have positive recent Sharpe ratios.

## Decision

**Not ready to trade.** The backtest confirms a real, market-neutral effect over
2023–2024 that survives maker costs. But its main driver is sign-unstable, and
performance over the last twelve months is indistinguishable from zero across
the grid.

Recommended before any capital or holdout decision:

1. **Pre-register one rule and threshold now.** The natural choice is the primary
   variant: combined model, 24h, tail, 12-month window, maker execution. Pass if
   its June–August 2026 Sharpe is above 0 and net return exceeds the maker cost of
   its turnover. Opening the holdout is a one-shot test with only about 90 days,
   so it can only reject.
2. **Understand the ratio sign flip before relying on it.** Test whether the
   long/short ratios behave as contrarian or trend-following indicators, and
   whether the flip coincides with a known change (Binance's own definitions,
   2022 data gaps, or the post-ETF market).
3. **Execution realism.** Maker fills are assumed, not modelled. A queue and
   fill-probability model, or a mixed maker/taker assumption, is needed. Taker-only
   results (Sharpe 0.69 for the primary variant) are the conservative bound.

## Limitations

- 40 variants were run and the primary was named in advance, but the attribution
  models were added after seeing results.
- Prices are trade prints at one-minute resolution; no slippage beyond fees.
- Funding is charged at settlement on the position held during that hour.
