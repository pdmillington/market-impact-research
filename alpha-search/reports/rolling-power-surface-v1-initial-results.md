# Rolling power surface v1: initial reading

## Design

The power surface is fitted at every calendar-month boundary using rolling 6-, 12-,
18-, 24- and 36-month windows plus expanding history. Every fit is evaluated
separately in its first, second and third month of use. Operational cadence summaries
simulate monthly, two-monthly and three-monthly recalibration using non-overlapping
origins aligned to January 2022. June--August 2026 remain absent and untouched.

## Training-window comparison

For next-month event-bar calibration, recent history is materially better than an
expanding fit. The lowest mean RMSE is obtained from six months at all three event
scales. The lowest median is six months for 200 and 500 events and twelve months for
1,000 events.

The table below uses the common October 2022--May 2026 comparison sample, so every
window is evaluated on exactly the same calendar months and sides.

| Construction | Best median window | Median RMSE | Six-month mean RMSE | Expanding mean RMSE |
|---|---:|---:|---:|---:|
| 200 events | 6 months | 0.000939 | 0.000955 | 0.001274 |
| 500 events | 6 months | 0.001567 | 0.001653 | 0.002077 |
| 1,000 events | 12 months | 0.002568 | 0.002731 | 0.003194 |

Longer 36-month windows sometimes have a lower 90th-percentile error despite a
higher typical error. This is a genuine responsiveness-versus-tail-stability
trade-off and argues against choosing a window from the mean alone.

## Calibration ageing and cadence

For a six-month training window, the median cell RMSE from calibration month one to
month three changes approximately as follows:

- 200 events: 0.000939 to 0.001022;
- 500 events: 0.001567 to 0.001671;
- 1,000 events: 0.002704 to 0.002706.

Monthly recalibration has the lowest mean error. For a fixed six-month training
window, moving to two-monthly recalibration raises mean error about 2--3%; moving to
three-monthly raises it about 3--5%. Two-monthly recalibration therefore remains a
credible operational compromise. At 1,000 events, a 36-month window becomes narrowly
best by mean RMSE under two- and three-month cadences, highlighting the greater
stability benefit of long histories at the coarser event scale.

## Collapse-exponent drift

The rolling estimates confirm that the decline in the expanding-fold collapse
exponent is not solely an artefact of cumulative averaging. The two-side mean gamma
from January 2022 to May 2026 changes as follows:

| Construction | Window | January 2022 | May 2026 |
|---|---:|---:|---:|
| 200 events | 6 months | 0.303 | 0.140 |
| 200 events | 12 months | 0.328 | 0.168 |
| 500 events | 6 months | 0.341 | 0.208 |
| 500 events | 12 months | 0.377 | 0.222 |
| 1,000 events | 6 months | 0.417 | 0.280 |
| 1,000 events | 12 months | 0.462 | 0.284 |

The rolling path is not assumed to be monotonic month by month, but its secular
decline is clear. A fixed gamma near one-half is therefore too slow-moving for a
live residual model, particularly at 200 events.

## Buy--sell symmetry

The formal five-fold comparison evaluates:

1. separate side shape and level, $A_s z^{p_s}a^{q_s}$;
2. common shape with separate levels, $A_s z^p a^q$;
3. common shape and level, $A z^p a^q$.

For event bars, imposing a common shape does not worsen median out-of-period
calibration. Relative to fully separate shapes, the common-shape/separate-level
model is tied at 200 events and improves median RMSE by approximately 0.6% at 500
events and 0.3% at 1,000 events. A completely pooled level is also very close, but
allowing side-specific amplitudes is the more conservative restriction.

The preferred operational specification is therefore provisionally

$$
\widehat I_s(z,a)=A_s z^p a^q,
$$

with a common collapse exponent and side-specific levels. This reduces rolling
parameter noise while preserving any small buy--sell level difference.

## Provisional decision

The evidence currently favours a 6--12 month training window and monthly or
two-monthly recalibration. This is not frozen yet. The next refinement should add
calendar-block uncertainty bands and test whether residual-alpha conclusions remain
stable under the common-shape specification before selecting one window and cadence
for the untouched holdout.

The next registered analysis is therefore to reconstruct out-of-sample residuals
from the rolling common-shape power surface, compare 6- and 12-month histories under
monthly and two-monthly recalibration, and repeat the 5/10/15/30-minute alpha screen.
Fixed-reference impact paths and calendar-block uncertainty bands for $p$, $q$ and
$\gamma$ will be produced before a calibration rule is frozen.
