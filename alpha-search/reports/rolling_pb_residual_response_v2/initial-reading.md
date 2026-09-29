# Rolling scale-free residual response: initial reading

This is a provisional gross-predictability diagnostic, not a trading result. The
impact model and the conditional residual standardisation are both frozen before
each test month. June--August 2026 remain excluded.

## What changed from v1

- Future horizons are 30 seconds, 1, 2, 5, 10, 15, 30 and 60 minutes.
- Results are reported for pooled, buy and sell observations.
- `scaled` retains the response-scaled model error.
- `conditional_z` robustly centres and scales that error using the preceding three
  months, separately by side and training-frozen scaled-pressure quintile.
- The v1 output remains unchanged in `reports/rolling_pb_residual_ic_v1`.

## Initial observations using the 12-month impact calibration

- The 200-event clock represents the requested 30-second target reasonably closely:
  its median realised horizon is 35.7 seconds and the median monthly p90 is 51.4
  seconds.
- The sign changes systematically with aggregation scale. Mean monthly Spearman IC
  at 30 seconds is +0.00895 for 200 events and +0.00619 for 500 events, is near zero
  at 1,000 events, and becomes negative from 2,000 through 8,000 events.
- The 8,000-event conditional-z residual already has mean monthly Spearman IC
  -0.01563 at 30 seconds. It reaches -0.02374 at two minutes and is -0.02252 at 15
  minutes. Thus the reversal ordering appears quickly rather than emerging only at
  the intended holding horizon.
- At 8,000 events and 15 minutes, the mean monthly Spearman IC is negative for both
  buys (-0.02383) and sells (-0.02156). The winsorised Q5-minus-Q1 return spread is
  much larger for buys (-0.98 bps) than sells (-0.13 bps), so economic symmetry is
  not yet established despite similar rank correlations.
- Direct scaling and conditional standardisation tell a similar rank-correlation
  story at 8,000 events: mean monthly 15-minute Spearman IC is -0.02172 for `scaled`
  and -0.02252 for `conditional_z`.
- All five 8,000-event residual quintiles have negative average monthly median
  side-aligned returns at most horizons, but the response becomes progressively
  more negative toward the high-residual quintile. This means the next comparison
  must establish incremental information beyond ordinary post-bar reversal,
  pretrend, raw impact and imbalance.

## Required next falsification

Run the residual and the simple benchmarks on identical observations and months.
The important object is the incremental residual contribution, not the standalone
IC. Add calendar-block uncertainty before selecting an event scale or horizon, and
do not formulate a trading strategy until that comparison is complete.

