# P&B scale-free impact v1: initial reading

## Design

The experiment uses corrected `timestamp_direction_v1` events and continuous,
non-overlapping bars containing 200, 500, 1,000, 2,000, 4,000 or 8,000 events.
June--August 2026 remain untouched. Every model is fitted to equally weighted
relative-imbalance by relative-activity training-cell means.

All P&B candidates impose

$$
\mathcal Q_N=\mathcal Q_{1000}(N/1000)^\xi,
\qquad
\mathcal R_{N,s}=\mathcal R_{1000,s}(N/1000)^\psi.
$$

The original model uses $u=za/\mathcal Q_N$ and central exponent $p=1$. The
activity-adjusted model replaces this with $u=za^\gamma/\mathcal Q_N$. A third
model also frees the central exponent. The flexible benchmark fits the existing
power surface independently at every event count and side.

Two expanding training samples are compared on identical tests from 2023 through
May 2026: full history beginning September 2019, and a sensitivity beginning
January 2021.

## Cross-scale result

On the full descriptive sample, the original P&B coordinate has cell RMSE
0.004183. Activity adjustment lowers this to 0.001367. The fitted activity
exponent is 0.589, the pressure-scale exponent is 0.413 and the response-scale
exponent is 0.632. Buy and sell response levels at 1,000 events are 0.01591 and
0.01527 respectively.

Freeing the central exponent changes full-sample RMSE only from 0.001367 to
0.001366. On the post-2021 sample it changes 0.00152755 to 0.00152753 and returns
$p=1.002$. The extra parameter is not justified by this run; the activity-adjusted
model with $p=1$ is the preferred scaling specification.

## Chronological comparison

Mean out-of-period cell RMSE across the four common test periods is:

| Training sample | Original P&B | Activity-adjusted P&B | Free central exponent | Independent surfaces |
|---|---:|---:|---:|---:|
| Full history | 0.006037 | 0.004342 | 0.004343 | 0.004299 |
| Post-2021 | 0.005763 | 0.004035 | 0.004037 | 0.003959 |

Activity adjustment improves mean OOS RMSE by about 28% relative to the original
P&B coordinate. More importantly, the constrained scale-free model is only about
1.0% worse than independently fitted surfaces under full-history training and
1.9% worse under post-2021 training. This is strong evidence that the event-count
dimension can be compressed into a common scaling law over 200--8,000 events.

## Early-period sensitivity

The early market is visibly different. At 1,000 events the median duration falls
from about 442 seconds in 2019 and 165 seconds in 2020 to 49 seconds in 2021 and
44 seconds in 2022. Median relative activity changes at the same time.

Excluding 2019--2020 improves mean activity-adjusted OOS RMSE from 0.004342 to
0.004035, approximately 7.1%, over identical 2023--2026 tests. Full history is
slightly better for the 2023 test, but post-2021 training is better thereafter.
The early observations should remain in structural descriptions but should not be
assumed useful for an operational calibration window.

## Important limitation

The aggregation-scale result is much stronger than the calendar-time result.
Out-of-period residuals retain a substantial negative relationship with log
activity, and errors rise in the later test periods. Independent surfaces reduce
but do not remove this effect. The likely next problem is calibration drift or a
missing calendar-time liquidity state, not failure of event-count scaling alone.

The next test should roll the activity-adjusted scale-free model through time with
6- and 12-month histories. Only after choosing the calibration rule should its
scaled residual be tested at 5, 10, 15 and 30 minutes and finally evaluated on the
untouched June--August 2026 holdout.

## Rolling residual correlation update

The activity-adjusted model was subsequently calibrated at every month boundary
from January 2022 through May 2026 using trailing 6- and 12-month histories. The
rolling activity exponent is substantially more stable than the earlier per-scale
surface estimates:

| Window | Mean $\gamma$ | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| 6 months | 0.545 | 0.067 | 0.417 | 0.644 |
| 12 months | 0.565 | 0.058 | 0.435 | 0.653 |

For a 12-month calibration and 15-minute future return, mean monthly Spearman
correlations between the scaled residual and the side-aligned future return are:

| Events per bar | Mean monthly Spearman IC | Monthly sign consistency |
|---:|---:|---:|
| 200 | +0.0020 | 77% positive |
| 500 | +0.0003 | approximately balanced |
| 1,000 | -0.0024 | 62% negative |
| 2,000 | -0.0068 | 77% negative |
| 4,000 | -0.0118 | 85% negative |
| 8,000 | -0.0217 | 89% negative |

At 8,000 events and 15 minutes, pooled Pearson IC is -0.0091. The mean monthly
median side-aligned future returns from the lowest to highest residual quintile are
approximately -0.34, -0.43, -1.34, -1.59 and -2.24 basis points. The ordering is
consistent with increasingly strong mean reversion after unusually high impact.
Six- and twelve-month calibrations give very similar results.

These are gross predictability diagnostics. The quoted monthly t-statistics in the
result table are descriptive and do not yet correct for the tested scale/horizon
family. The next gate is comparison with raw impact, imbalance and pretrend on the
same test months, followed by calendar-block inference and a costed strategy.
