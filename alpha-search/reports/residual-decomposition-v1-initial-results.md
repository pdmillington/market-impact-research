# Residual decomposition v1: initial results

## Question

The scale-free P&B residual showed small, consistent reversal ICs at larger event
scales (about −0.022 at 8,000 events and 15 minutes). The signed residual is

$$
X_t = R_t - P_t,
$$

where $R_t$ is the bar return over trailing 24-hour volatility and
$P_t=s_t\widehat y_t$ is the frozen `pb_activity` prediction. Because $X$ is a
fixed 1:−1 combination of $R$ and $P$, this experiment asks:

1. Is the 1:−1 restriction supported?
2. Does $P$ add out-of-sample information once $R$, clock-time pretrend and a
   naive linear flow term $q_t=s_t|Q_t|/V_{24h}$ are present?
3. Does any resulting forecast clear realistic costs, and is that edge steady?

## Design

- `configs/residual_decomposition_v1.json`, `scripts/run_residual_decomposition.py`
  and `notebooks/ResidualDecompositionWorkbench.ipynb`.
- Event bars of 200 to 8,000 events. $P$ uses the existing 12-month rolling
  `pb_activity` fits from `rolling_pb_residual_response_v2`, each calibrated
  before its month.
- Pretrend: clock-time returns over 15 and 60 minutes ending at the bar start,
  over trailing volatility. In signed-price terms this is the earlier aligned
  pretrend.
- Targets: signed 5, 10, 15 and 30-minute returns on the 200-event price clock.
  Delayed targets enter at the first 200-event endpoint at least one second after
  the decision. The realised median entry delay is about 8–11 seconds.
- Forecast regressions are fitted on the preceding 12 months and frozen for each
  test month. Training targets that would end inside the test month are
  embargoed. Tests run from January 2023 to May 2026 (41 months).
- June–August 2026 are not used, including as target prices.
- Costs are base-tier assumptions: 10 bps taker and 4 bps maker round trip.

## Result 1: the flow component also reverses, so the residual discards information

Mean monthly Spearman IC at 15 minutes, 2023–May 2026:

| Events | $R$ | $P$ | $X=R-P$ | $q$ | pre15 | pre60 |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | −0.006 | −0.012 | +0.002 | −0.013 | −0.070 | −0.068 |
| 1,000 | −0.022 | −0.028 | −0.002 | −0.028 | −0.067 | −0.066 |
| 4,000 | −0.039 | −0.041 | −0.010 | −0.042 | −0.059 | −0.060 |
| 8,000 | −0.052 | −0.049 | −0.021 | −0.049 | −0.056 | −0.060 |

The residual was built on the assumption that flow-implied impact is the
"permanent" part and the excess is transient. The data say otherwise. The
flow-predicted move $P$ reverses at least as strongly as the realised move $R$.
Subtracting one from the other cancels most of the signal.

In within-month regressions, $R$ and $P$ enter with the same sign at 2,000
events and above, and $P$ is far larger at small scales. The restriction
$b_R+b_P=0$ is rejected at every scale and horizon (t between −5.2 and −12.3).

## Result 2: the impact model adds nothing once return, pretrend and flow are present

Mean monthly OOS IC (15 minutes):

| Events | Residual | Bar return | Return + pretrend | Full impact + flow |
|---:|---:|---:|---:|---:|
| 200 | 0.002 | 0.004 | 0.045 | 0.047 |
| 1,000 | −0.002 | 0.022 | 0.049 | 0.050 |
| 8,000 | 0.018 | 0.052 | 0.055 | 0.053 |

- The residual is worse than the plain bar return at every scale. At 500–8,000
  events and 10–15 minutes, the paired monthly difference has t between −7.6
  and −16.7.
- Adding $P$ to return and pretrend changes OOS IC by −0.001 to +0.003
  (|t| ≤ 1.8).
- Adding $P$ to return, pretrend and linear flow $q$ makes the forecast slightly
  *worse* at most scales (t between −0.6 and −4.0). The nonlinear impact surface
  does not beat a linear flow term for prediction.
- Pretrend carries most of the information: OOS IC of about 0.04 on its own,
  roughly independent of bar size. The bar return adds to it mainly at large
  bar sizes.

## Result 3: tails exceed costs only because of a few stress days

On the face of it, the top 1% of return-plus-pretrend forecasts clear taker costs.
For example, pretrend alone at 1,000 events and 10 minutes has a 20.4 bps gross
edge, daily-block t of 9.6, and 73% of months positive.

Those statistics hide the concentration:

| Tail (top 1%) | Mean edge | Best-10-day share | Edge on other days | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pretrend, 1,000 ev, 10 min | 20.4 | 93% | 1.6 | 20.9 | 37.5 | 12.8 | 3.5 |
| pretrend, 8,000 ev, 5 min | 21.2 | 73% | 7.0 | 18.7 | 32.0 | 16.7 | 12.7 |
| full + flow, 8,000 ev, 10 min | 17.3 | 72% | 5.4 | 7.6 | 30.5 | 13.1 | 17.1 |
| pretrend, 8,000 ev, 30 min | 27.5 | 141% | −13.4 | 8.9 | 49.7 | 45.9 | 2.7 |

Across all 576 tail cells (sizes × horizons × models × tail fractions), none has
an edge above the 10 bps taker round trip once its best ten days are removed. The
maximum is 8.0 bps. In several cells the remaining days lose money in aggregate.
Monthly trade counts vary widely: some months have two or three trades at
50–250 bps, while calmer months have 50–80 trades near zero. The top-5% tails
earn at most 3.0 bps on ordinary days (median 0.5 bps), below even the maker
round trip.

The residual itself never clears costs. Its best tail is 4.2 bps gross.

## Gate decision

**Step 1 fails for the impact-based factor.**

- The P&B residual should be retired as an alpha factor. It is dominated by its
  own components, and the impact surface adds no out-of-sample information
  beyond bar return, pretrend and linear flow.
- The impact model is still useful as a cost and execution model (Step 2) and
  for the academic work.

**What does survive is an event-driven reversal effect.** After extreme moves,
price reverts by tens of bps over 5–30 minutes. The effect is concentrated in a
small number of crash and cascade days, has shrunk in 2025–2026, and is not a
steady intraday signal. Economically this is exactly the Step 3 hypothesis:
forced flow overshoots and reverts. It should be studied as an explicit event
study of stress episodes, not as a continuous factor. Two things need direct
attention there:

- slippage and spread widening on exactly those days;
- the losses on the other side of the trade when a move keeps going.

## Limitations

- Tail trades overlap heavily. Trades-per-day counts are signal occurrences,
  not independent positions.
- Costs exclude own-size impact and stress-day spread widening, both of which
  would reduce the stress-day edge that the tails depend on.
- The one-second-minimum entry uses a 200-event price clock, so the realised
  delay is coarser (median 8–11 seconds). Immediate and delayed edges are almost
  identical, which argues against a microstructure artefact.
- Twelve-month training windows start in 2022, so the first test months use
  impact fits and regressions trained on the 2022 bear market.
