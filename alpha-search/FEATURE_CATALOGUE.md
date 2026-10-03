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

## Trailing-window flow features

These features are defined over a trailing **clock-time** window of length $W$
ending at the decision time $t$, rather than over one bar. They are computed from
reconstructed `timestamp_direction_v1` events via the 1-second grid
(`second_grid_v1`), which sums event quantities by aggressor side each second.
Because signed and gross volume are additive, the window totals are identical
whether they are summed from fills, events, event bars or time bars; the parser
verifies that events conserve fill volume exactly. What distinguishes the
features below is the window, the normalisation and the transformation, not the
bar construction.

Notation, for events $i$ with aggressor sign $\epsilon_i\in\{-1,+1\}$ and
quantity $q_i$ timestamped in $(t-W,\,t]$:

$$
Q_W(t)=\sum_{i}\epsilon_i q_i,\qquad V_W(t)=\sum_{i} q_i .
$$

The reference volume for a window of length $W$ is the typical gross volume of
such a window over the preceding 30 days, excluding the current window:

$$
V^{\mathrm{ref}}_W(t)=\frac{W}{30\,\mathrm{d}-W}\,\sum_{i\,\in\,(t-30\,\mathrm{d},\;t-W]} q_i .
$$

| Name | Definition | Nearest bar feature | Interpretation |
|---|---|---|---|
| `window_imbalance_ratio_W` | $z_W=Q_W/V_W$ | signed `absolute_imbalance_ratio` (`relative_imbalance`) | One-sidedness of the window's flow, in $[-1,1]$. Ignores whether the window was busy or quiet. |
| `window_relative_activity_W` | $a_W=V_W/V^{\mathrm{ref}}_W$ | `bar_volume_fraction_24h` (`relative_activity`) | Window volume relative to a typical window of the same length. |
| `window_normalised_imbalance_W` | $x_W=z_W\,a_W=Q_W/V^{\mathrm{ref}}_W$ | signed `normalised_imbalance` | Net demand in units of typical market capacity; the metaorder-size variable of the square-root-law literature. |
| `window_scaled_pressure_W` | $x^*_W=z_W\,a_W^{\gamma}$, $\gamma$ fixed at the canonical `pb_activity` value (about 0.55) | collapse coordinate $x^*=z a^{\gamma}$ | The impact research's price-relevant pressure, evaluated on the window. $\gamma$ is not re-fitted. |
| `window_sign_persistence_W_h` | $P_{W,h}=\frac{1}{n}\sum_{k=1}^{n}\mathbf 1\{\operatorname{sign}Q_h(t-(k-1)h)=\operatorname{sign}Q_W(t)\}$, $n=W/h$ | none (closest: `abs_event_sign_imbalance`) | Share of the $n$ sub-windows of length $h$ whose net flow has the same sign as the whole window. High values mean steady rather than bursty flow. |

Transformation used with these features:

| Name | Definition | Interpretation |
|---|---|---|
| `z30(f)` | $\dfrac{f(t)-\operatorname{mean}_{30\,\mathrm d}(f)}{\operatorname{sd}_{30\,\mathrm d}(f)}$ over the values of $f$ at earlier decision times in $[t-30\,\mathrm d,\,t)$ | Time-series standardisation that removes slow drifts in a feature's level and scale. Uses strictly earlier observations only. |

All window features are known at $t$ (strictly prior flow), and the decision is
assumed to execute after $t$ with the entry delay registered in the hypothesis.

## Execution-signature features (hourly; E-005 / H-013)

Built by `scripts/build_signature_features.py` from reconstructed events
(timestamp_direction_v1). For event $e$:

- aggressor sign $s_e$;
- notional $n_e=q_e\,\mathrm{vwap}_e$;
- price levels $L_e$;
- side-aligned sweep depth $d_e=s_e\,10^4\ln(p^{\mathrm{last}}_e/p^{\mathrm{first}}_e)\ge0$ bps.

Sums run over events in the window $[T-W,T)$, with $W\in\{8\,\mathrm h,24\,\mathrm h\}$.
Each feature is also stored as `z30` against its previous 720 hourly values
(at least 360).

| Name | Definition | Interpretation |
|---|---|---|
| `ml_share_W` | $\sum_{L_e\ge2}n_e\,/\,\sum n_e$ | Urgency: the share of aggressive notional that walked through more than one price level. Unsigned. |
| `ml_imbalance_W` | $\sum_{L_e\ge2}s_e n_e\,/\,\sum_{L_e\ge2}n_e$ | One-sidedness of urgent (multi-level) flow, in $[-1,1]$. |
| `sweep_pressure_W` | $\sum_{L_e\ge2}s_e d_e\,/\,N_W = \sum_{L_e\ge2}10^4\ln(p^{\mathrm{last}}_e/p^{\mathrm{first}}_e)\,/\,N_W$, with $N_W$ the number of events | Signed depth taken per event: H-014's side-aligned sweep depth $d_e\ge0$, signed by the aggressor and aggregated. |
| `absorption_imbalance_W` | $(R^{-}_W-R^{+}_W)/(R^{-}_W+R^{+}_W)$, where $R^{\mp}_W$ counts level runs by sellers (−) or buyers (+) that first reach $\ge3$ hits in the high absorption tercile inside the window (frozen E-003 cuts); 0 if there are no runs | Passive interest. Positive means sellers repeatedly hit a bid that held (a large passive buyer). |
| `burst_skew_W` | $(N^{(10)}_W-N^{(1)}_W)/N_W$, with $N^{(k)}$ the events whose signed 5 s burst intensity is in decile $k$ (frozen E-002 cuts) | One-sided self-excitation. Event-count share stands in for time share. |
| `large_imbalance_W` | $\sum_{n_e\ge\$1\mathrm m}s_e n_e\,/\,\sum n_e$ | Large-trader footprint. |
| `flow_imbalance_W` | $\sum s_e n_e\,/\,\sum n_e$ | Plain signed flow (control; the `window_imbalance_ratio_W` analogue in notional). |
| `own_return_W` | $\ln(P_T/P_{T-W})$, $P$ = last event price of the hour | Conditioning variable for E-005 (with `flow_imbalance_W`). |

All are known at $T$. The locked months (2026-06 onward) are not built.

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
