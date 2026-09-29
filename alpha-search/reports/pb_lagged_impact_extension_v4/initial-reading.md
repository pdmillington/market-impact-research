# Frozen-P&B previous-bar extension: initial reading

## Exactly what is frozen, and when

For each chronological sample and fold, the experiment first loads the existing
`pb_activity` fit from `pb_surface_scaling_v1`. That fit was calibrated only on
the fold's training observations, using equal-weight relative-imbalance by
activity cell means across all event scales and both sides.

Before any lag regression is run, **every P&B parameter is frozen**: the shared
shape parameters, pressure and response scales, event-count exponents, activity
exponent, normal activity scale, and buy/sell response levels. The P&B model is
not jointly refitted after the lag variables are introduced.

The frozen P&B prediction is then evaluated on the same fold's training bars.
Only the incremental linear coefficients are calibrated against those training
residuals. Each training calendar month receives equal total weight. The lag
coefficients and the training-only median duration are then frozen and applied
unchanged from the fold's `test_start` through `test_end`. June--August 2026 are
not used.

The `calibration_ledger.csv` table records these dates and observation counts for
every sample, fold and event scale.

## Dynamic variable

Let $A_t$ be the side-aligned impact predicted by the frozen P&B surface. The
previous-bar impulse expressed in the current bar's direction is

$$
H_{t-1}=s_ts_{t-1}A_{t-1}.
$$

It is positive for same-sign consecutive bars and negative for opposite-sign
bars. The simplest dynamic model is

$$
y_t=A_t+c_{s_t}+\kappa H_{t-1}+\varepsilon_t,
$$

where $c_{s_t}$ is a training-fitted buy or sell residual mean. A negative
$\kappa$ means that prior impact contributes a counter-move during the current
bar; a positive value would mean continuation.

## Chronological result

The simple lag coefficient is negative in every one of the 48 calibrations: two
training-history definitions, four folds and six event scales. With full-history
training, its mean value moves from about -0.26 at 200 events to -0.13 at 8,000
events. The post-2021 estimates are also negative throughout and are most negative
around 500--1,000 events.

Relative to the untouched frozen P&B prediction, `lag_impact` improves mean OOS
monthly RMSE by approximately:

| Event scale | Full history | Post-2021 |
|---:|---:|---:|
| 200 | 1.12% | 1.43% |
| 500 | 2.31% | 2.59% |
| 1,000 | 2.94% | 3.20% |
| 2,000 | 2.82% | 3.29% |
| 4,000 | 2.44% | 2.62% |
| 8,000 | 1.53% | 1.62% |

Every individual fold improves at every event scale. Averaged across scales and
folds, RMSE falls from 0.031665 to 0.030969 with full-history training and from
0.031615 to 0.030845 post-2021. Mean monthly residual/lag-impulse correlation
falls from about -0.22 to approximately zero, showing that the fitted term is
capturing the structure it targets rather than merely shifting the mean.

## Does previous activity help?

Only marginally. The activity interaction changes aggregate RMSE from 0.030969
to 0.030962 with full-history training and from 0.030845 to 0.030840 post-2021.
Its design condition numbers are much larger than those of the simple lag model,
and the main and interaction coefficients compensate for one another. The
duration interaction is similarly unconvincing.

The parsimonious reading is therefore:

1. there is robust evidence that the previous bar contributes a counterreaction
   to the current bar return;
2. the effect is largest at intermediate short event scales and weakens at long
   scales;
3. adding previous activity does not yet produce a stable structural improvement;
4. the next physical extension should be a small distributed-lag/propagator test,
   rather than another flexible static surface.

