# Intraday flow v1 (H-012): initial results

## Registered question

Does persistent aggressive flow predict continuation at 5–30 minute horizons,
beyond the frozen return benchmark, and can passive entry overcome the taker
round trip? The full registration is H-012 in `HYPOTHESIS_LEDGER.md`; the
features are defined in `FEATURE_CATALOGUE.md` ("Trailing-window flow features").

- `configs/intraday_flow_v1.json`, `scripts/run_intraday_flow.py`.
- BTCUSDT, 560,448 decisions every 5 minutes (February 2021 to May 2026).
- Test months January 2023 to May 2026, all development data.
- Primary: W = 60 minutes, H = 15 minutes, benchmark + `z30(x_60)`.

## Result 1: flow adds nothing beyond recent returns

| Cell | ΔIC vs benchmark | t | Months ΔIC > 0 | Positive flow coefficient (refits) |
|---|---:|---:|---:|---:|
| W15 H5 / H15 / H30 | −0.002 / −0.002 / −0.003 | −2.7 / −2.2 / −2.7 | 39% / 37% / 44% | 0% / 10% / 20% |
| **W60 H15 (primary)** | **−0.002** | **−2.6** | **39%** | **51%** |
| W60 H5 / H30 | −0.001 / −0.004 | −2.7 / −3.6 | 41% / 29% | 27% / 49% |
| W240 H5 / H15 / H30 | −0.001 / −0.003 / −0.007 | −2.8 / −3.0 / −3.6 | 32% / 39% / 22% | 46% / 76% / 80% |
| Interaction (primary) | −0.003 | −2.4 | 39% | 34% |
| Scaling z_60 (primary) | −0.000 | −1.2 | 37% | 39% |
| Spaced (primary) | 0.000 | 0.7 | 49% | 61% |

The benchmark alone has an OOS IC of 0.042–0.046 (t 9–13), the return-reversal
effect from H-006. Flow is correlated with those recent returns, and adding it
only adds estimation noise. **H-012 is rejected under conditions 1 and 4.**

Coefficient signs are structured, but recorded rather than pursued: short-window
flow reverses, 240-minute flow continues. Pursuing that would be a new
hypothesis with its own registration and a new economic argument.

## Result 2: no intraday tail pays costs, passive or taker

Primary horizon 15 minutes, top 5% of forecasts, bps per trade:

| Model | Trades/day | Gross (mid to mid) | Gross, ordinary days | Passive net | Taker net |
|---|---:|---:|---:|---:|---:|
| Benchmark | 14.4 | +0.64 | +0.24 | −7.66 | −9.32 |
| Flow W60 H15 | 14.9 | +0.38 | −0.15 | −7.85 | −9.61 |

Across every cell and tail fraction, gross edges are 0.2–4.5 bps and about zero
on ordinary days. Passive entry saves 3 bps of fees but gives back about 1.3 bps
to adverse selection: fills cluster when price moves against the resting order.

## Execution model: a flaw in the registered fill rule

The registered passive order rests at the last trade price. When the last
aggressor was a buyer, that price is the ask, and a buy there would have
executed immediately as a taker. The simulation then counts the next sell at the
bid as a fill: median fill time 1 second, fill rate 86%. Passive results are
therefore **optimistic**, which strengthens the negative economic conclusion and
does not affect the statistical rejection.

Execution model v2, recorded in the ledger for future registrations, rests
passive orders at side-specific bid and ask proxies taken from event aggressor
signs.

## What this implies for intraday work on BTC

The best intraday BTC signal we have is return reversal, with IC about 0.045.
It is statistically strong but worth about 0.5 bps per trade on ordinary days,
against at least 7 bps of costs even with passive entry. An intraday strategy at
base-tier fees therefore needs either:

1. **A much larger edge per trade:** higher-volatility contracts (fees are fixed
   in bps, while edge scales with volatility), or an information source not yet
   used (order-book depth, spot–perp lead-lag);
2. **Passive execution on both sides** (about 4 bps round trip) with a credible
   fill model, which needs Binance `bookTicker` quotes (available 2023-05 to
   2024-04 only).
