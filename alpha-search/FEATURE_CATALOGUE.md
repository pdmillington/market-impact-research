# Alpha feature catalogue

This is the human-readable reference for feature names accepted by the short-event-bar
alpha tools. The authoritative implementation is in
`src/alpha_search/screen_features.py`; impact features fitted separately within each
training fold are implemented in `src/alpha_search/impact_model.py`.

All returns and price ranges below are logarithmic and expressed in basis points.
The current bar is denoted by `t`. A feature described as **current-bar** is known only
when bar `t` has closed. A feature described as **strictly prior** excludes bar `t`.

## Primary reversion features

| Configuration name | Family | Timing | Definition and interpretation |
|---|---|---|---|
| `aligned_pretrend_1000_events` | Pretrend | Prior return + current-bar side | Return over the preceding approximately 1,000 events, multiplied by the sign of the current bar's imbalance. A high value means the preceding trend and current order-flow side agree. The price history excludes the current bar, but the aligned feature is known only when the current bar's side is known. |
| `aligned_pretrend_4000_events` | Pretrend | Prior return + current-bar side | The same calculation over approximately 4,000 preceding events. |
| `aligned_pretrend_12000_events` | Pretrend | Prior return + current-bar side | The same calculation over approximately 12,000 preceding events. This is the current primary benchmark. |

For an event history `H`, the construction is approximately

$$
\operatorname{sign}(Q_t)\,10{,}000\log\left(\frac{P_{t-1}}{P_{t-1-H}}\right).
$$

The code converts the requested event history to a whole number of earlier bars. Thus
12,000 events means 60 earlier 200-event bars, 24 earlier 500-event bars, or 12 earlier
1,000-event bars.

## Imbalance and volume

| Configuration name | Timing | Definition and interpretation |
|---|---|---|
| `log_abs_signed_imbalance` | Current-bar | Natural logarithm of the absolute signed BTC imbalance. |
| `absolute_imbalance_ratio` | Current-bar | Absolute signed BTC imbalance divided by the bar's gross BTC volume. |
| `normalised_imbalance` | Current-bar plus prior benchmark | Absolute signed BTC imbalance divided by gross market volume in the preceding 24 hours. This is the horizontal variable in the fitted impact curve. |
| `log_gross_volume` | Current-bar | Natural logarithm of the bar's gross BTC volume. |
| `bar_volume_fraction_24h` | Current-bar plus prior benchmark | Current bar volume divided by gross volume in the preceding 24 hours. |

The impact-surface workbench uses the equivalent, more descriptive names
`relative_imbalance` for `absolute_imbalance_ratio` and `relative_activity` for
`bar_volume_fraction_24h`. Their product is `normalised_imbalance`:

$$
x=\text{relative imbalance}\times\text{relative activity}.
$$

## Activity and volatility

| Configuration name | Timing | Definition and interpretation |
|---|---|---|
| `log_duration` | Current-bar | Natural logarithm of the seconds required to complete the fixed-event bar. |
| `relative_flow_rate` | Current-bar plus prior benchmark | Current BTC per second divided by the average BTC per second in the preceding 24 hours. |
| `normalised_range` | Current-bar plus prior benchmark | Current high-low range divided by realised volatility in the preceding 24 hours. |
| `log_trailing_volatility` | Strictly prior | Natural logarithm of preceding-24-hour realised volatility. |

The 24-hour benchmarks exclude the current bar.

## Impact and price response

Let the current signed impact be

$$
I_t=\operatorname{sign}(Q_t)\,10{,}000\log\left(\frac{P_{t,\mathrm{end}}}{P_{t,\mathrm{start}}}\right).
$$

| Configuration name | Timing | Definition and interpretation |
|---|---|---|
| `return_efficiency` | Current-bar | Signed current impact divided by the current high-low range. It measures directional movement relative to total intrabar movement. |
| `normalised_signed_impact` | Current-bar plus prior benchmark | Signed current impact divided by preceding-24-hour realised volatility. This is the vertical variable in the impact fit. |
| `expected_normalised_impact` | Training-fitted | Impact predicted by the side-specific logarithmic curve fitted only on the fold's training observations. |
| `impact_residual` | Training-fitted and current-bar | Observed normalised signed impact minus expected normalised impact. A high value means greater movement in the current imbalance direction than the fitted curve predicts. |
| `marginal_impact_bps_per_btc` | Training-fitted | Derivative of the fitted impact curve: estimated additional basis points for one more signed BTC at the current imbalance level. |

The buy and sell curves are fitted separately in every training fold:

$$
\widehat I_s(x)=a_s+b_s\log\left(1+\frac{x}{c}\right),
$$

where `s` is buy or sell, `x` is `normalised_imbalance`, and `c` is the training
sample's median normalised imbalance. The fitted parameters are frozen before the
following test period is evaluated.

The v1 surface experiment also estimates the zero-origin model

$$
\widehat I_s(z,a)=A_s z^{p_s}a^{q_s}.
$$

For this power family, `collapse_exponent_gamma` is $q_s/p_s$ and the candidate
collapsed coordinate is

$$
x^*=z a^{q_s/p_s}.
$$

`impact_residual` in the surface experiment always means observed impact minus a
specific named model's frozen prediction. Residuals from different model families
are therefore distinct features and must not be pooled without retaining the model
name.

## Order-flow persistence

| Configuration name | Timing | Definition and interpretation |
|---|---|---|
| `aligned_flow_persistence_1000_events` | Prior flow + current-bar side | Signed imbalance over approximately 1,000 preceding events, divided by their gross volume and aligned with the current side. The historical flow excludes the current bar, but the aligned feature is known only at the current bar's close. |
| `aligned_flow_persistence_4000_events` | Prior flow + current-bar side | The same calculation over approximately 4,000 preceding events. |
| `aligned_flow_persistence_12000_events` | Prior flow + current-bar side | The same calculation over approximately 12,000 preceding events. |

A positive value means recent net order flow has the same sign as the current bar.

## Reconstructed-event structure

| Configuration name | Timing | Definition and interpretation |
|---|---|---|
| `fills_per_event` | Current-bar | Original exchange fills divided by reconstructed events. High values identify bars containing more multi-fill reconstructed events. |
| `price_levels_per_event` | Current-bar | Sum of event price-level counts divided by reconstructed events. High values identify more sweeping across price levels. |
| `abs_event_sign_imbalance` | Current-bar | Absolute difference between buy- and sell-event counts divided by total event count. |

These are bar-level summaries. The separate dense-decay analysis retains explicit
single-fill/single-price, multi-fill/single-price and multi-price event categories.

## Using the catalogue in an experiment

The generic conditioner runner accepts one `primary_feature` and a list of
`conditioners`. The current strategy logic interprets the highest primary-feature
quantile as a contrarian entry condition and reports both the unconditioned result
and the result intersected with the highest conditioner quantile. It also saves the
complete primary-by-conditioner quantile surface, so useful low, middle, non-monotonic
or asymmetric regions are not hidden by the simple high-quantile summary.

Feature and quantile cut points are calculated separately for buy- and sell-dominated
bars using training observations only. Current-bar features assume a decision at the
bar close; they do not assume execution inside the bar.
