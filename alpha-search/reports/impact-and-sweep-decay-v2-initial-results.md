# Impact-function and sweep-decay v2: initial results

## Scope and sample discipline

The corrected dataset contains 84 validated months from September 2019 through
August 2026. The current development and validation calculations end in May
2026. June--August 2026 have not been used in curve fitting, model selection or
decay plots and remain a small alpha holdout.

The impact results are contemporaneous calibration diagnostics, not alpha
returns. The event-decay results are descriptive conditional responses based on
a deterministic one-in-500 event sample. Neither section includes execution
costs or a portfolio model.

## Short-bar duration

The full-history 200-event cache contains about 12.1 million bars through May
2026. Its median duration changed materially:

- 2019: about 1.31 minutes;
- 2020: about 0.52 minutes;
- 2021--22: about 0.14--0.16 minutes;
- 2023--26: about 0.20--0.24 minutes.

Thus a 200-event bar already represents roughly 8--14 seconds in the mature
market. Fifty- and 100-event bars would mainly move the study towards the
transaction-bounce region. The initial comparison therefore uses 200, 500 and
1,000-event bars and 30-second, one-minute, two-minute and five-minute bars.

## Impact-function comparison

Candidate curves were constrained through zero and fitted to equal-count
development-period bin means separately for buy- and sell-dominated bars.

Using relative imbalance, `abs(signed flow) / gross volume`, the fixed-event
power exponents are close to one:

| Construction | Sell exponent | Buy exponent |
|---|---:|---:|
| 200 events | 0.993 | 1.029 |
| 500 events | 0.975 | 0.952 |
| 1,000 events | 1.009 | 0.892 |

For the 200- and 500-event fits, the zero-origin logarithmic curvature reaches
the bottom of the diagnostic grid and is effectively linear. This is materially
different from the earlier affine-log interpretation with a free negative
intercept.

Using normalized net flow, `abs(signed flow) / preceding-24h gross volume`, the
fixed-event curves are clearly concave:

| Construction | Sell exponent | Buy exponent |
|---|---:|---:|
| 200 events | 0.687 | 0.804 |
| 500 events | 0.709 | 0.822 |
| 1,000 events | 0.724 | 0.798 |

The 30-second through five-minute clock bars are also generally concave, with
power exponents mostly between about 0.75 and 0.92. The curvature therefore
appears to combine relative imbalance with activity/scale rather than residing
in relative imbalance alone.

The current implication is to retain the two-dimensional object:

$
E[y_t\mid |Q_t^{signed}|/Q_t^{gross},\;
Q_t^{gross}/\overline V_{t,24h},\;\text{side}].
$

The next impact gate is out-of-period calibration of the two-dimensional
surface and comparison with a fixed master shape whose input and response
scales update through time.

### Logarithmic versus power selection

The initial power exponents above came from a conventional log--log fit, while
the logarithmic curve minimized response-space error in the bin means. A fair
model comparison now fits both two-parameter curves to the same equal-count
development bins using the same response-space squared loss.

With fits frozen through December 2022, the out-of-period result is not a
uniform victory for either family:

- over all imbalance values, the logarithmic curve usually has lower
  equal-bin calibration error, but the power curve often has slightly lower
  observation-weighted/bar-level error;
- in the upper tails during 2023, the power curve generally calibrates better;
- in 2024, the logarithmic curve has lower equal-bin error in all six
  bar-size/side cells above the frozen 90th and 95th percentile cutoffs;
- above the 95th percentile in 2024, the logarithmic curve's median bin-RMSE
  improvement is about 14%, and it also wins paired bar-level loss in four of
  six cells.

This supports the visual impression for the deepest 2024 tail but not as a
time-invariant result. The family used for residual construction should
therefore be chosen with rolling 12/24-month calibration and quarterly
walk-forward tests rather than a single frozen 2019--22 fit.

## Sweep-conditioned event decay

Event aggression is measured by side-aligned endpoint sweep:

$$
S_i=10^4\epsilon_i\log(P_i^{last}/P_i^{first}).
$$

The positive-sweep quartile boundaries, calibrated through December 2022 and
then frozen, are approximately 0.095, 0.300 and 0.686 basis points.

Across the 2.423 billion analysed events through May 2026:

| Class | Event share | Quantity share | Mean quantity (BTC) | Mean sweep (bps) |
|---|---:|---:|---:|---:|
| No endpoint sweep | 84.64% | 43.35% | 0.163 | 0.000 |
| Positive sweep Q1 | 5.29% | 7.16% | 0.431 | 0.042 |
| Positive sweep Q2 | 4.53% | 8.90% | 0.627 | 0.179 |
| Positive sweep Q3 | 2.98% | 10.59% | 1.133 | 0.458 |
| Positive sweep Q4 | 2.53% | 29.94% | 3.771 | 1.599 |
| Opposite endpoint move | 0.02% | 0.06% | 0.845 | -0.879 |

The strongest-sweep class is rare by count but represents nearly 30% of
quantity. Size control is therefore essential before attributing its path to
sweep independently of volume.

In the 2023--24 validation period, equal-weighting buy- and sell-aligned daily
paths gives the following Q4 response:

- immediate response: about 1.47 bps;
- 0.1 seconds: about 1.27 bps;
- 15 seconds: about 1.21 bps;
- one minute: about 1.06 bps;
- five minutes: about 0.93 bps;
- 30 minutes: about 0.33 bps.

The immediate response therefore begins reversing within 100 milliseconds.
The same qualitative pattern appears in both the 2019--22 development period
and the 2025--May 2026 later exposed period. Smaller positive-sweep classes
generally show initial continuation rather than immediate reversal.

This is not yet evidence of a resonant wave. The Q4 path is primarily a sharp
initial reversal followed by slower decay; isolated later sign changes are
small relative to day-block uncertainty and do not yet form a stable repeated
oscillation.

## Next identification step

The next decay model should test whether endpoint sweep adds information after
controlling for:

1. log event quantity or notional;
2. trailing volatility and event-arrival rate;
3. buy/sell side and calendar period;
4. lagged and subsequent signed order flow.

A local-projection or propagator specification should report total response,
post-event response and interval increments. Fill count and price-level count
remain available as secondary variables and should only be promoted if they
add information conditional on volume and realized endpoint sweep.
