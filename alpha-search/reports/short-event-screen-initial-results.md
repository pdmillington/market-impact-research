# Short-event-bar feature screen: initial sandbox results

## Design

- Corrected `timestamp_direction_v1` events only.
- Event bars: 200, 500 and 1,000 events.
- Targets hold future event count constant:
  - 12,000 events: `n+60`, `n+24`, `n+12` respectively;
  - 16,000 events: `n+80`, `n+32`, `n+16` respectively.
- Training: through August 2020.
- Validation: September 2020 through June 2021.
- Later sandbox: July through December 2021. This is already exposed and is
  not a final holdout.
- Feature quintile cutoffs are estimated on training only, separately for buy-
  and sell-dominated bars.
- The target is future return aligned with the current imbalance side.
- Inference uses paired UTC-day Q5-minus-Q1 differences. Validation p-values
  use a normal approximation and Benjamini--Hochberg correction over the full
  276-cell search.
- A stricter ranking collapses correlated bar-size cells into cross-scale
  hypotheses. This is the primary interpretation table.

The 12,000-event target had a median duration of about 10.4 minutes in
validation and 11.8 minutes in the later sandbox. The 16,000-event target had a
median duration of about 13.9 and 15.9 minutes respectively. Training durations
were much longer because market activity was lower.

## Preregistered feature families

1. **Imbalance:** absolute signed BTC, within-bar imbalance ratio and imbalance
   divided by trailing 24-hour volume.
2. **Volume:** gross BTC volume and current-bar volume as a fraction of trailing
   24-hour volume.
3. **Activity:** bar duration and flow rate relative to the trailing 24-hour
   average rate.
4. **Volatility:** bar range divided by trailing realised volatility and the
   trailing volatility level itself.
5. **Impact:** return efficiency and contemporaneous normalized signed impact.
6. **Microstructure:** fills per reconstructed event, price levels per event
   and order-sign imbalance by event count.
7. **Pre-trend:** price return over the preceding 1,000, 4,000 or 12,000 events,
   aligned with the current bar's imbalance side. The current bar is excluded.
8. **Flow persistence:** signed flow over those same strictly prior event
   histories, aligned with the current side.
9. **Fitted impact:** training-fitted expected impact, impact residual and
   marginal impact in basis points per additional BTC.

## Main result

The dominant cross-scale relationship was mean reversion after a prior price
trend aligned with the current imbalance. For the preceding 12,000-event
pre-trend, Q5-minus-Q1 aligned future return was negative for both sides, both
targets, all three bar sizes and both out-of-training periods.

| Side | Future events | Validation median spread | Later median spread |
|---|---:|---:|---:|
| Sell | 16,000 | -9.33 bps | -8.59 bps |
| Sell | 12,000 | -8.23 bps | -7.49 bps |
| Buy | 16,000 | -6.99 bps | -6.59 bps |
| Buy | 12,000 | -6.05 bps | -6.14 bps |

Negative means that bars in the highest aligned-pretrend quintile subsequently
had lower returns in the current imbalance direction than bars in the lowest
quintile. This is a relative conditional spread, not a strategy profit.

Other repeated cross-scale relationships included:

- sell-side reversal following high normalized bar range;
- sell-side reversal in high relative-flow-rate bars;
- reversal following persistent prior flow, particularly on the sell side;
- sell-side reversal following high contemporaneous normalized signed impact;
- price-level intensity and imbalance magnitude effects.

The fitted impact residual itself was weak and unstable. Marginal impact was
more promising on sell-dominated bars at the 16,000-event horizon, but it did
not pass the strict cross-scale rule as consistently as pre-trend, flow
persistence or normalized signed impact.

## Interpretation limits

- The features and targets are highly correlated across bar sizes and horizons;
  individual significant cells are not independent discoveries.
- The later sandbox has already been examined.
- Q5-minus-Q1 spreads do not specify a position, entry price, exit price,
  turnover or capacity.
- Overlapping 10--15 minute targets make individual-bar standard errors
  inappropriate; day blocks are an improvement but not the final inference.
- Costs, spread, slippage and delayed entry are not yet deducted.
- The next notebook is for visual diagnosis and rule formulation, not for
  further unconstrained feature mining.
