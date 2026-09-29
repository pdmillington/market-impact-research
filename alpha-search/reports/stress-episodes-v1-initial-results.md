# Stress episodes v1 and v2: initial results

## Question

Step 1 found that the only tail edge clearing costs came from a handful of crash
and cascade days. Step 3 studies those days directly as discrete episodes. Is
there a reversion after extreme moves that survives realistic entry timing and
costs, and does open interest separate forced flow (reverting) from informed
flow (persistent)?

## Data and design

- **1-second grid** (`scripts/build_second_grid.py`, cached as
  `features/alpha/symbol=BTCUSDT/second_grid_v1`, January 2021 to May 2026).
  Each second is built from `timestamp_direction_v1` events and holds:
  - last, high and low price;
  - buy and sell volume and event counts;
  - deep events (endpoint sweep of at least 0.686 bps or at least five price
    levels) and summed sweep bps;
  - a spread proxy: the price gap between consecutive opposite-side events at
    most 250 ms apart, capped at 100 bps.
- **Open interest.** Binance USD-M daily `metrics` archives, 2020-09-01 to
  2026-09-28: 2,219 files, all checksum-verified. They are consolidated by the
  new shared command `crypto-market-data download-and-parse-metrics` into
  `canonical/metrics/symbol=BTCUSDT/metrics_5m.parquet`: 638,438 five-minute
  snapshots, 634 missing, and 75,255 duplicate rows removed. 473 zero-OI
  snapshots mark Binance outages (e.g. 2021-05-22) and are treated as missing.
- **Trigger.** $z$ is the largest aligned move over the trigger windows, in
  units of trailing-24h one-minute volatility scaled by $\sqrt{W/60}$. The
  threshold is calibrated on 2021–2022 to 50, 100 or 200 merged episodes a year
  and frozen for January 2023 to May 2026. The primary target is 100 a year.
  - v1 windows: 30, 60 and 300 seconds, with a 30-minute merge gap.
  - v2 windows: 15 and 60 minutes, with a 60-minute merge gap. v2 was specified
    after v1 failed, because v1's 30-second window produced 529 of 553 triggers,
    while Step 1's edge came from 15–60 minute moves.
- **Entry rules.** Enter at the trigger; after a delay of 1 s to 5 min; or at
  exhaustion (the trigger-direction extreme not extended for 30 or 60 seconds).
  Holding periods are 5, 15, 30 and 60 minutes.
- **Costs.** Net = gross − 10 bps taker round trip − the 30-second spread proxy
  at entry.
- **Inference.** Means with 95% intervals from resampling whole UTC days.
- `scripts/run_stress_episodes.py`, `configs/stress_episodes_v1.json` and
  `configs/stress_episodes_v2.json`, and
  `notebooks/StressEpisodeWorkbench.ipynb` (with an `EXPERIMENT` control).

## Result 1: extreme moves are mostly permanent

v1 test episodes (352 episodes on 280 days) follow an average 110 bps run-up over
ten minutes. Afterwards price retraces about 10 bps, mostly within seconds to a
few minutes, and then stays flat for the hour. v2 episodes (380 on 321 days,
median trigger move 124–178 bps by year) look the same.

The moves keep running after the trigger. For v1, the median further move in the
trigger direction is 31 bps within one minute and 68 bps within an hour. The
90th percentiles are 109 and 233 bps.

## Result 2: no entry rule clears costs

| Test episodes, 15-minute hold | v1 gross | v1 net | v2 gross | v2 net (95% CI) |
|---|---:|---:|---:|---:|
| Enter at trigger | 10.1 | −2.7 | 13.5 | +1.6 (−7.8, +11.3) |
| 10-second delay | 9.4 | −3.6 | 9.9 | −2.0 |
| 60-second delay | 4.1 | −6.3 | 6.1 | −4.3 |
| Exhaustion, 30 s quiet | 1.3 | −9.7 | 2.8 | −7.9 |
| 5-minute delay | 0.0 | −10.1 | 2.1 | −8.0 |

- Only immediate entry captures the small bounce. Waiting for exhaustion, the
  natural way to avoid the falling knife, is consistently worse.
- By year, v2 trigger-entry net edge with a 15-minute hold is −7.8 (2023),
  +33.5 (2024), −10.5 (2025) and −4.9 bps (2026). With the 30-second exhaustion
  rule it is −12.2, +11.4, −18.7 and −10.8 bps. Only 2024, the year of several
  large liquidation crashes, is positive.

## Result 3: open interest does not separate reverting from persistent episodes

Open interest fell during 271 of 352 v1 test episodes, so forced closing is
common. Yet the path for OI-falling episodes is almost identical to the path for
OI-rising ones. The real-time one-hour OI change and the ex-post label both fail
to identify a profitable subset. Episodes in the scheduled-release window
(:00/:30 UTC) and up-moves both do worse than average.

## A lead, not a result: liquidity withdrawal

Splitting by the spread proxy at the trigger gives the only large differences:

| v2, trigger entry, 15 min | Calibration net | Test net | Test hit rate |
|---|---:|---:|---:|
| Spread ratio (at trigger / trailing 24h) low | −1.6 | −12.3 | 50% |
| middle | −22.4 | +11.4 | 60% |
| high | −9.7 | +16.7 | 64% |

In the test period, high-spread episodes revert strongly. Their largest wins are
well-known liquidation crashes: 2024-03-05, 2024-01-09 and 2024-08-05. That fits
the theory that only overshoots with withdrawn liquidity revert.

It does not qualify as a finding:

- The calibration years show no such pattern. In v1 the high tercile loses
  49 bps net in 2021–22.
- It was found after inspecting 16 splits in two overlapping experiments.
- The raw spread proxy is in bps and is non-stationary because BTC's price
  roughly tripled with a fixed tick. The ratio form is post-hoc.

If pursued, it needs a pre-registered rule, frozen before the holdout is opened,
and ideally real quote data (Binance `bookTicker`) instead of the trade-based
proxy.

## Gate decision

**Step 3 fails as specified.** Neither fast spikes (v1) nor 15–60 minute moves
(v2) revert enough after the trigger to pay taker costs. Open interest does not
rescue it. The Step 1 stress-day edge is best understood as a small immediate
bounce plus a few very large crash reversals. It cannot be selected in advance
with the features tested here.

The data built here remains useful:

- the 1-second grid;
- the open-interest table, for positioning and funding-style signals in Step 4.

## Limitations

- Trade-price paths without quotes. The spread proxy is crude, and fills during
  stress would be worse than the trade prints suggest.
- Episodes are few, especially recently (36 v2 episodes in January–May 2026).
  Intervals are wide.
- The trigger threshold was chosen on 2021–2022, a higher-volatility period.
  Trailing-volatility scaling makes calm years such as 2023 produce more
  episodes.
