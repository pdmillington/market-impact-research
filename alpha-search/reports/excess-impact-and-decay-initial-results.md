# Initial reversion, excess-impact, and event-decay results

## Scope

This is sandbox evidence, not a profitability backtest. The impact curve and all
quintile cut points are fitted using data before September 2020. Validation covers
September 2020 through June 2021; July through December 2021 is an already-exposed
later sandbox. Outcomes use the first completed event-bar endpoint at or after 5,
10, 15, 20 or 30 clock minutes. Fees, spread, slippage, overlapping positions and
position limits are not included.

## Step 1: reversion benchmarks

The pure top-quintile trailing-return contrarian benchmark is positive across all
three bar sizes. At 10 minutes its equal-weight UTC-day mean is approximately
5.6–7.1 bps in validation and 5.3–7.1 bps later. At 15 minutes it is approximately
5.1–6.4 bps in validation and 5.5–6.8 bps later.

Requiring the prior trend to align with current bar imbalance identifies a strong
sell-side subset. The plain sell-side aligned-pretrend rule averages approximately
9.7–10.5 bps at 10 minutes and 10.5–11.4 bps at 15 minutes in validation. The
buy-side equivalent is much weaker.

These are overlapping conditional returns, not independent trades or net returns.

## Step 2: excess-impact conditioning

For sell-dominated bars, requiring the impact residual to be in its
training-frozen top quintile increases the validation reversion return by roughly:

- 3.6–5.2 bps at 10 minutes across the three bar sizes; and
- 2.8–4.4 bps at 15 minutes.

All three 10-minute validation increments survive the screen's false-discovery
adjustment. Two of the three 15-minute increments do so at the 10% level. In the
later sandbox the sell-side increment remains positive across sizes, but is smaller
and statistically weak.

The buy-side result is not stable. The condition adds little in validation and is
flat or harmful in most later-sandbox comparisons. Excess impact therefore has
some mileage as a sell-side conditioner, but the current evidence does not support
a symmetric universal rule.

Training-frozen signal frequencies also fall sharply in the later sample. For the
plain aligned-pretrend rule they fall from roughly 16% per side in validation to
about 6.5–7.1%; for the doubly conditioned rule they fall from roughly 1.1–1.6% to
about 0.6%. This is direct evidence that calibration cannot be ignored if the
condition is retained.

## Execution classes

The corrected sample contains 694,748,436 reconstructed events. Exact exclusive
event classes are:

| Class | Event share | Quantity share |
|---|---:|---:|
| Single fill, single price | 65.64% | 19.64% |
| Multiple fills, single price | 14.25% | 15.18% |
| 2 price levels | 8.71% | 8.95% |
| 3–4 price levels | 6.09% | 10.91% |
| 5–9 price levels | 3.60% | 13.85% |
| 10+ price levels | 1.70% | 31.48% |

Deep multi-price events are rare by count but economically large by quantity. The
10+ level group alone represents nearly one third of reconstructed-event volume.

## Dense decay curves

The decay calculation uses a deterministic 1-in-200 sample of 3.47 million anchor
events while retaining the full event stream for future-price lookup. It reports
entry displacement, movement from first to last execution, total response from the
previous event price, and post-event response at horizons from 1 second to 30
minutes.

Raw buy and sell paths diverge strongly over longer horizons. That divergence may
reflect Bitcoin's directional drift, event selection and genuine asymmetry; a raw
pooled signed curve should therefore not be interpreted as pure impact decay.

As a first drift diagnostic, giving buy and sell daily paths equal weight produces
a clearer multi-price pattern:

- validation immediate response: about 0.73 bps;
- validation peak: about 0.88 bps around 15 seconds;
- validation 10-minute response: about 0.64 bps;
- validation 30-minute response: about 0.52 bps;
- later-sandbox immediate response: about 0.52 bps;
- later-sandbox peak: about 0.74 bps around 5 seconds;
- later-sandbox 30-minute response: about 0.35 bps.

Thus the average multi-price trajectory initially continues, then decays. The
dense incremental curves contain short sign changes, but the average evidence does
not yet establish a repeating wave or resonant frequency. Day-block uncertainty,
matched size/volatility controls and temporal replication are required before such
an interpretation.

## Current decision

The excess-impact hypothesis should not be discarded, but it should be narrowed to
a provisional sell-side condition. Before profitability modelling, the next gate
is temporal scaling and calibration: compare a frozen raw curve, a rolling raw
curve, a fixed master shape with rolling scales, and a slowly updating master
shape. The buy-side condition should not be rescued by post-hoc threshold searches;
if retained at all, it needs independent evidence from another prespecified
conditioner or later data.
