# Impact surface and residual alpha v1: initial reading

## Scope

The experiment fits separate buy and sell impact models for 200-, 500- and
1,000-event bars and 1-, 2- and 5-minute bars. Five expanding chronological folds
test calendar 2022, 2023, 2024, 2025 and January--May 2026. June--August 2026 remain
excluded from the cache and untouched.

Each training sample defines a 12 by 6 equal-count grid in relative imbalance and
relative activity. Candidate models are fitted with equal weight on populated cell
means. The same frozen model and residual-quintile cuts are then applied to the next
test period. Future returns begin at the completed bar endpoint and use the first
completed endpoint at or after 5, 10, 15 or 30 minutes.

## Surface result

The two-factor surface materially improves event-bar calibration. Relative to the
previous zero-origin log curve in normalised net flow, the best model's median
out-of-period equal-cell RMSE is lower by approximately:

- 44.8% for 200-event bars;
- 44.0% for 500-event bars;
- 39.4% for 1,000-event bars.

The power surface wins all ten fold-by-side comparisons at 200 events. At 500
events, the power surface wins four and the log-imbalance/activity surface wins six.
At 1,000 events, the log-imbalance/activity surface wins nine of ten.

For clock-time bars, the old one-dimensional log curve in normalised net flow remains
competitive: it has the lowest median test-cell RMSE at one and two minutes, while
the log-activity surface is only narrowly better at five minutes. The event-clock and
wall-clock results should therefore not be forced into one specification.

## Candidate scale collapse

For event bars, the unrestricted power surface produces median collapse exponents
$\gamma=q/p$ of approximately:

- 0.48 at 200 events;
- 0.55 at 500 events;
- 0.59 at 1,000 events.

Across individual event-bar fold/side fits, the range is about 0.40--0.68. The
expanding estimates drift downward through time; for example the two-side mean for
1,000-event bars declines from approximately 0.67 in fold 1 to 0.52 in fold 5.
Thus $z\sqrt a$ is a useful event-bar approximation but not yet a stable universal
law.

Clock-time power surfaces are different: median $\gamma$ is about 1.12, 1.16 and
1.27 at one, two and five minutes. Moreover, the power surface is not normally the
best clock-time model. This is positive evidence that bar construction changes the
economic meaning of activity, not merely the number of observations.

## First residual-alpha screen

The surface residual has only a very small linear information coefficient, but its
sign differs by construction:

- 200-event residuals do not show stable 10--15 minute reversal;
- 1,000-event residuals show modest reversal in seven of ten fold/side cells;
- clock-time residuals show negative within-day IC in all ten fold/side cells at
  both 10 and 15 minutes for the one-, two- and five-minute constructions.

For the power-surface residual, high-residual-quintile contrarian returns average
roughly 0.1--0.7 basis points gross at 10--15 minutes, depending on construction.
The effect is most consistent for two-minute bars and is also visible at five
minutes. This is too small to interpret as tradeable before fees, spread, slippage,
latency, turnover and overlapping positions are modelled.

The residual effect is nonlinear: top-quintile contrasts are clearer than the
unconditional linear IC. Multiple-testing-adjusted significance is mixed, and the
results are an initial hypothesis screen rather than a frozen strategy.

## Physical interpretation

The power surface

$$
\frac{\Delta p}{\sigma}=A z^p a^q
$$

is consistent with net aggressive pressure divided by effective liquidity. When
$p$ is near one, it implies effective liquidity proportional to
$V_{24h}a^{1-q}$. An event-bar value of $q$ near one-half corresponds approximately
to effective liquidity growing with the geometric mean of current-bar and daily
volume. The market absorbs more flow in active conditions, but available depth and
replenishment do not grow one-for-one with aggressive volume.

The next physical test should introduce first-to-last event-price sweep as a proxy
for the initial local liquidity shock, then estimate how the shock decays. If the
surface residual predicts reversal after controlling for sweep and pretrend, it is
more plausibly temporary overshoot. Continuation would instead point toward delayed
or permanent informational impact.

## Next gate

Before profitability modelling:

1. inspect the surface and collapse plots manually across all folds and sides;
2. compare power and log-activity residuals rather than selecting on one fit;
3. add pretrend and volatility controls to the residual screen;
4. connect event sweep to immediate residual size and the subsequent decay kernel;
5. freeze a small candidate rule before opening June--August 2026.
