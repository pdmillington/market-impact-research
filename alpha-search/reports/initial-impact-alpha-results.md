# Initial impact-alpha sandbox results

These results use corrected `timestamp_direction_v1` events from September
2019 through December 2021. They are development evidence, not a final
out-of-sample claim.

## Split and targets

- Training: September 2019 through August 2020.
- Validation: September 2020 through June 2021.
- Later sandbox check: July through December 2021.
- Event-time reference: 4,000-event horizons `n+3` through `n+6`.
- Other event-bar sizes approximately match the same number of future events.
- Clock-time reference: five-minute horizons 3 through 6.

The event-time clock changed substantially across the sample. For 4,000-event
bars, the median duration of `n+3` fell from 44.0 minutes in training to 10.4
minutes in validation and was 11.8 minutes in the later sandbox. For `n+6`,
the corresponding medians were 89.8, 21.1 and 23.9 minutes. The event offset
therefore remains the primary definition; duration distributions must accompany
every estimate.

## Impact fits

Separate buy and sell logarithmic impact curves explain roughly 22--39% of
individual normalized contemporaneous impact, depending on bar construction.
Sell-side slopes exceed buy-side slopes for every event-bar size. The gap is
small for five-minute bars. This supports retaining side-specific models.

## Repeating candidate patterns

### Marginal impact

The clearest repeated event-time result occurs after buy-imbalance bars on
8,000-event bars. At its three-bar horizon, high marginal-impact observations
continued in the buy direction while low marginal-impact observations tended
to reverse. Pooled Q5 and Q1 means were +4.57 and -0.87 bps in validation,
then +2.31 and -1.39 bps in the later sandbox. An equal-day-weighted contrast
was 17.77 bps in validation (screening t-statistic 2.71) and 19.74 bps in the
later sandbox (t-statistic 2.27). The much larger equal-day estimate and the
fall in Q5 membership from 3,970 to 408 observations both show substantial
distribution and weighting sensitivity; a day-block bootstrap is required.

For sell bars, marginal impact is also informative in 2,000-event and
five-minute constructions. The repeated result is generally that low marginal
impact is followed by stronger reversal, while high marginal impact is followed
by less reversal or occasional continuation. This is not symmetric with the
8,000-event buy result.

### Impact residual

Five-minute sell bars show an overshoot/reversal pattern. At 30 minutes, the
highest impact-residual quintile had mean aligned future returns of -4.89 bps
in validation and -2.30 bps in the later sandbox. The Q5-minus-Q1 differences
in pooled means were -4.83 and -2.95 bps. The equal-day-weighted differences
were -6.30 bps (screening t-statistic -3.59) and -2.95 bps (t-statistic -1.79).

The 4,000-event sell-bar residual has the same qualitative later-sample
direction at longer horizons, but validation is weaker. Buy-side residual
patterns differ and are less stable.

## Interpretation

The first pass does not select a winning clock. Event bars give the strongest
repeated high-marginal-impact buy result; time bars give the cleanest
sell-side overshoot/reversal result. Pooling buys and sells would obscure both.

The next model should combine marginal impact and impact residual with trailing
activity, volatility, pre-trend and flow persistence. It must use delayed-entry
returns, frozen training transformations, and cost break-even reporting before
any pattern is called tradable.
