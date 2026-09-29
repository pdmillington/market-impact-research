# Alpha search

This sub-project investigates predictive patterns separately from the academic
reaction-function paper while sharing the same source data, canonical fill schema,
event reconstruction, and bar-building methodology.

Research outputs here should be assessed using chronological train/validation/test
splits, walk-forward evaluation, and realistic fees, spread, slippage, latency, and
turnover assumptions. Explanatory results from the academic work can generate alpha
hypotheses; alpha discoveries can generate robustness questions for the paper, but
the reporting and validation standards remain separate.

## Project terminology

The project uses the following vocabulary consistently:

- **Feature**: any measured input, whether or not it proves predictive. Examples
  include imbalance, pretrend, excess impact and flow rate.
- **Candidate alpha factor**: a feature, or transformation of features, deliberately
  tested for association with future returns. In this single-asset setting,
  **time-series predictor** is often more precise than the broader industry term
  **factor**.
- **Primary alpha factor**: the main predictive hypothesis in an experiment. The
  initial benchmark is aligned pretrend followed by a contrarian position.
- **Conditioning factor**: a variable used to determine when the primary factor is
  stronger, weaker or changes sign. Excess impact and reconstructed-event structure
  are examples.
- **Signal**: the actionable output after choosing direction, threshold and possibly
  position size.
- **Strategy**: the complete trading rule, including signal combination, entry,
  exit, position sizing and costs.
- **Risk factor**: a systematic exposure such as broad market direction, momentum,
  volatility or liquidity. A risk factor is not automatically alpha.
- **Information coefficient (IC)**: the correlation between a factor observed at
  time `t` and a future return over horizon `h`. Pearson IC measures linear
  association; Spearman or rank IC measures monotonic association and is less
  sensitive to outliers.
- **Factor correlation**: correlation among candidate factors, used to identify
  redundancy or potentially distinct information.
- **Conditional IC**: factor--future-return correlation evaluated within states or
  bins of a conditioning factor.

The project distinguishes contemporaneous relationships from prediction. For
example, current imbalance versus the price movement within the same bar describes
impact; a feature known at bar close versus the return beginning after bar close is
a predictive test. Correlation alone is not evidence of a profitable strategy:
stability across chronological periods, overlapping observations, turnover and
realistic trading costs must also be considered.

The code generally retains the neutral term `feature`, because a measured input
should only be called an alpha factor after predictive testing. Configuration fields
named `primary_feature` and `conditioners` correspond respectively to the primary
alpha factor and conditioning factors in research discussion.

## Initial research track

The first alpha track compares impact-conditioned predictors on two clocks built
from the same corrected `timestamp_direction_v1` events:

- continuous fixed-event bars, initially 2,000, 4,000 and 8,000 events; and
- non-overlapping five-minute bars.

Event-bar targets are bar offsets such as `n+3`, `n+4` and `n+5`. Their realized
clock-time distributions are always reported. The primary cross-size comparison
matches total future event count to 4,000-event offsets 3--6: offsets 6--12 for
2,000-event bars and the nearest integer offsets for 8,000-event bars. A separate
sensitivity analysis may select offsets by training-period duration. Five-minute
targets use offsets 3--6.

The initial feature family combines signed imbalance, participation, activity,
volatility, pre-trends, flow persistence, fitted impact, impact residuals and
separate buy/sell marginal-impact estimates.

## Current gated roadmap

The completed corrected dataset covers September 2019 through August 2026. The
current first research block resolves the contemporaneous impact function and
short-horizon decay before returning to strategy profitability. June--August
2026 are excluded from model development by default and retained as a small
unopened alpha holdout.

1. Establish simple trailing-return and order-flow-aligned reversion benchmarks
   on 200, 500 and 1,000-event bars. As part of this foundation, estimate dense
   individual-event response paths and separate single-fill/single-price,
   multi-fill/single-price and multi-price executions. The decay work is placed
   here because continuation versus reversal determines the sensible alpha
   direction and holding horizon.
2. Test whether training-fitted excess impact adds information to the reversion
   benchmark. The first test uses side-specific, training-frozen residual
   quintiles and exact 5, 10, 15, 20 and 30-minute outcomes.
3. If the condition adds stable information, investigate temporal scale collapse,
   rolling calibration and standardized real-time residuals. Otherwise test a
   small preregistered set of alternative conditioners.
4. Freeze an entry/exit rule and model profitability with overlapping positions,
   fees, spread, slippage, latency and turnover.
5. Only then add market-state conditioning and nonlinear/mutual-information
   diagnostics.

The immediate next task within step 3 is now fixed: reconstruct impact residuals
from the rolling common-shape power surface

$$
\widehat I_{s,t}=A_{s,t}z_t^{p_t}a_t^{q_t},
$$

compare 6- and 12-month training histories with monthly and two-monthly
recalibration, and retest 5/10/15/30-minute residual alpha. Before selecting the
operational calibration rule, add calendar-block uncertainty bands for $p$, $q$ and
$\gamma=q/p$ and compare fitted impact at fixed $(z,a)$ reference points. This step
must precede profitability modelling because a static residual now demonstrably
mixes genuine excess impact with calibration drift.

## Impact and decay research v2

The revised impact work treats relative imbalance and relative activity as
separate axes:

$$
z_t=Q_t^{signed}/Q_t^{gross},\qquad
a_t=Q_t^{gross}/\overline V_{t,24h}.
$$

Zero-origin logarithmic and power curves are compared with flexible monotonic
and two-dimensional benchmarks. Buy and sell branches remain separate, and
the preferred curve is selected by next-period calibration and temporal
stability rather than individual-bar in-sample R-squared. Short event bars and
clock-time bars are compared explicitly; their realized duration distributions
are always reported.

For individual-event decay, the primary aggression measure is now the
side-aligned first-to-last execution-price displacement:

$$
S_i=10^4\epsilon_i\log(P_i^{last}/P_i^{first}).
$$

Positive sweeps are classified using training-frozen quartile boundaries.
No-sweep and opposite-endpoint events remain explicit. Fill count and distinct
price-level count are retained only as secondary diagnostics because quantity
already captures the main size dimension and multiplicity does not itself show
price aggression. The old `event_decay_v1` results remain unchanged for
reproducibility; revised results belong to `reports/event_decay_sweep_v2`.

The active research notebooks are deliberately limited to:

- `notebooks/PBScaleFreeImpactWorkbench.ipynb`, the canonical impact workbench;
  it covers the original and activity-adjusted Patzelt--Bouchaud collapses across
  200--8,000-event bars, the 2019--2020 inclusion sensitivity, rolling calibration
  and the first out-of-sample residual-correlation screen;
- `notebooks/EventSweepDecayWorkbenchV2.ipynb`, the separate event-level decay
  workbench for dense total, post-event and incremental response paths conditioned
  on realized execution sweep.

The activity-adjusted scale-free model is the default impact specification for new
alpha tests and for conditioning other factors. Its rolling, past-only residual is
the canonical `excess_impact` factor. Earlier one-dimensional curves, independently
fitted power surfaces and multiscale residual constructions are retained as
benchmarks in `notebooks/archive`, but should not define new conclusions unless the
result is reproduced with the scale-free specification. See `notebooks/README.md`
for the complete notebook map.

The surface experiment is configured by
`configs/impact_surface_residual_alpha_v1.json` and run with
`scripts/run_impact_surface_residual_alpha.py`. It compares relative-imbalance-only,
normalised-net-flow and two-factor models on five expanding chronological folds.
All surface cuts, parameters and residual-quintile boundaries are frozen using the
past before the following test period. Compact results live in
`reports/impact_surface_residual_alpha_v1`; the underlying bar caches remain below
`shared-data/features/alpha`.

The selected power surface is stress-tested by
`configs/rolling_power_surface_v1.json` and
`scripts/run_rolling_power_surface.py`. That experiment refits every month using
6-, 12-, 18-, 24- and 36-month histories plus expanding history, then measures the
first, second and third month of calibration age. The companion
`scripts/run_power_surface_symmetry.py` compares separate buy/sell shapes, a common
shape with separate side levels, and a completely pooled surface. Results are in
`reports/rolling_power_surface_v1` and are displayed in the archived
`notebooks/archive/ImpactSurfaceAndResidualAlpha.ipynb` notebook.

The cross-event-count scaling experiment is configured by
`configs/pb_surface_scaling_v1.json` and run with
`scripts/run_pb_surface_scaling.py`. It imposes power-law horizontal and vertical
scales in event count and compares the original P&B coordinate $za$, the
activity-adjusted coordinate $za^\gamma$, and a generalised central exponent.
The same 2023--2026 tests compare training from September 2019 with training from
January 2021. Larger 2,000-, 4,000- and 8,000-event caches are derived exactly
from the aligned 1,000-event cache by
`scripts/build_coarse_impact_research_bars.py`. Compact results live in
`reports/pb_surface_scaling_v1`; the initial reading is in
`reports/pb-scale-free-impact-v1-initial-results.md`.

The residual/activity concern is tested by the nested extension in
`configs/pb_activity_vertical_extension_v2.json`, with compact results in
`reports/pb_activity_vertical_extension_v2`. It allows a separate vertical
activity exponent while centring activity at its training-fitted normal level for
each event scale. The additional term produces only a small out-of-sample RMSE
gain, does not materially reduce residual/log-activity correlation and is weakly
identified against the original horizontal activity exponent. It is retained as a
documented falsification, not promoted over the parsimonious `pb_activity` model.

The remaining pattern is exposed directly by
`configs/pb_residual_activity_diagnostics_v1.json` and
`scripts/run_pb_residual_activity_diagnostics.py`. Compact cell, bar-decile,
pressure-by-activity and monthly stability tables live in
`reports/pb_residual_activity_diagnostics_v1` and are plotted in the canonical
scale-free workbench. All diagnostic bins are training-frozen. The initial result
shows a monotonic bar-level relationship that strengthens with pressure and in the
later test folds, so neutralisation has deliberately not yet been imposed.

The interpretable activity-dependent-concavity hypothesis is tested in
`configs/pb_activity_beta_extension_v3.json`, with results in
`reports/pb_activity_beta_extension_v3`. It allows the P&B beta parameter to vary
logistically with relative activity and adds observed-versus-predicted conditional
curves to the canonical workbench. The unrestricted model produces only a small,
inconsistent OOS RMSE gain, does not solve full-history residual/activity
correlation and develops compensating, economically implausible parameters in the
post-2021 sample. It is therefore documented but not promoted.

The same notebook displays the active residual-response experiment configured by
`configs/rolling_pb_residual_response_v2.json` and run with
`scripts/run_rolling_pb_residual_ic.py`. The preferred activity-adjusted model is
calibrated monthly on trailing 6- and 12-month histories. Same-bar residuals are
compared with side-aligned future returns from 30 seconds through 60 minutes using
the 200-event bar endpoints as a common high-resolution price clock. Results are
reported pooled and separately for buy and sell bars. A past-only conditional
z-score uses the preceding three months within side and scaled-pressure quintile;
the direct response-scaled residual remains available as a robustness view. This
is not the older overlapping construction in which a long impact-measurement
window is evaluated at every shorter decision bar. Compact results live in
`reports/rolling_pb_residual_response_v2`; the v1 tables remain unchanged in
`reports/rolling_pb_residual_ic_v1`.

The notebook index is now consolidated. Superseded exploratory notebooks are
archived and labelled, not deleted, so the research trail and old results remain
reproducible. Historical notebook builders also write to `notebooks/archive` so
they cannot silently repopulate the active directory.

Generated alpha features and bar caches belong below
`shared-data/features/alpha`; no new parser or independent trade store should be
created.

The original starting notebook is archived at
`notebooks/archive/ImpactAlphaEventVsTime.ipynb`. It defaults to planning mode so opening
the notebook does not start a full scan accidentally; set `RUN_BUILD = True` only
when a deliberate feature build is intended.

The earlier graphical research workbench is archived at
`notebooks/archive/ImpactAlphaWorkbench.ipynb`. Its first control cell exposes bar size,
horizon, fit family, pooled versus separate buy/sell curves, raw-bar versus
binned-mean fitting, and the alpha feature. It reports data coverage and
distributions, prints the exact fitted equations, shows the earlier
Patzelt--Bouchaud scaling estimates, and keeps contemporaneous impact separate
from future-return tests. Notebook-only plotting dependencies are listed in
`requirements-notebook.txt`.

The shorter-bar systematic screen uses 200, 500 and 1,000-event bars with
matched 12,000 and 16,000 future-event targets. Its caches are versioned as
`short_event_screen_v1`, its result tables are in
`reports/short_event_screen_v1`, and the manual visual companion is
`notebooks/archive/ShortEventFeatureWorkbench.ipynb`. The initial interpretation is in
`reports/short-event-screen-initial-results.md`.

The archived Step 1/2 manual workbench is
`notebooks/archive/ExcessImpactAndDecayWorkbench.ipynb`. It shows the fitted impact
equations and training bin means, residual drift, plain and conditioned reversion,
two-dimensional conditioning surfaces, monthly stability, and dense event-level
decay by fill and price-level multiplicity. Compact predictive tables live under
`reports/excess_impact_v1`; decay tables live under `reports/event_decay_v1`.
The initial reading is recorded in
`reports/excess-impact-and-decay-initial-results.md`.

The archived multiscale residual experiment measures excess impact over the current
decision bar, 1,000 events or 4,000 events while retaining 200-, 500- and
1,000-event decision clocks and exact 10/15-minute outcomes. Its manual visual
companion is `notebooks/archive/MultiscaleImpactResidualWorkbench.ipynb`. This is kept
separate from the pretrend-conditioner workbench because excess impact is the
primary candidate factor rather than a condition on a pretrend rule.
Impact curves are calibrated on globally aligned, non-overlapping blocks at each
measurement scale, then applied as rolling factors at every decision-bar close.

## Reusable tools for the extended dataset

The current conclusions are provisional; the priority is reproducible testing on
the longer history. `TOOLING.md` documents the portable SSD workflow, data-root
audit, experiment manifests, rolling or expanding calendar folds, generic
conditioner screen, notebook factory and detailed event-decay runner. The tools
derive folds from the data actually present and record a portable fingerprint, so
results can be tied to an exact dataset even after `shared-data` moves drives.

`FEATURE_CATALOGUE.md` is the human-readable reference for configuration feature
names, formulas, timing and interpretation. The authoritative implementation remains
in `src/alpha_search/screen_features.py` and `src/alpha_search/impact_model.py`.
