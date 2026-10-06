# Thesis figures and tables: full-data redo (October 2026)

This folder holds the impact chapter's figures and tables, reproduced on the
corrected, extended BTCUSDT sample. It is ready to copy into
`thesis/market impact/`. The file names match the chapter's existing figure
names, so the `\includegraphics` lines need no change. The two exceptions are
08 and 09 (see the table below).

**Sample.** January 2021 to August 2026, BTCUSDT. Events are the shared
`timestamp_direction_v1` events: one event per contiguous same-sign run at a
timestamp. The original inputs had one row per timestamp, carrying the first
fill's sign; about 1% of timestamps were mis-signed as a result. Data from
2026-09 onward is not used (reserved for the alpha-search test).

**How each set was produced:**

- **Figures 01–25:** the paper notebooks run unchanged on the corrected event
  files (`scripts/export_paper_events.py`), using
  `scripts/run_paper_notebooks.py`. Only the data and output paths are
  changed; the per-cell log is in `run_log.json`. One cell is skipped:
  BarConstructionRobustness cell 3, an out-of-order zero-range diagnostic
  that produces nothing used later.
- **Figures 26–28 and the `activity_surface_*` tables:**
  `scripts/make_surface_figures.py`. It reads the outputs of
  `scripts/run_activity_surface.py` and `scripts/diagnose_activity_surface.py`.

## Chapter figures

| File | Producer | Note for caption / text |
|---|---|---|
| 01_log_abs_return_vs_imbalance_hexbin | PowerLawImpact | |
| 02_binned_signed_impact_curve | PowerLawImpact | |
| 02a_impact_curve | PowerLawImpact | |
| 04_normalised_signed_impact_curve | PowerLawImpact | |
| 05_normalised_impact_model_comparison | PowerLawImpact | The logarithmic form still tracks the binned means best |
| 05b_model_standardised_residuals | PowerLawImpact | |
| 06_monthly_model_stability | PowerLawImpact | See caveat 1 |
| 08_event_bar_fitted_curves | BarConstructionRobustness | |
| 09_time_bar_fitted_curves | BarConstructionRobustness | |
| 10_regime_response_curves_grid | RegimeResponseCurves | |
| 18_state_dependent_effective_liquidity | StateDependentEffectiveLiquidity | See caveat 2 |
| 21_impact_decay_lead_lag_4000_event_trajectories | ImpactDecayAndLeadLag | |
| 22_pb_volume_impact_scaling_collapse | PatzeltBouchaudScaling | Pooled ξ = 0.825, ψ = 0.575, α = 1.54, β = 0.77 (`pb_scaling_exponents.csv`, `pb_master_curve_parameters.csv`) |
| 24_pb_transaction_price_pinning | PatzeltBouchaudScaling | |
| 25_pb_pinning_by_activity | PatzeltBouchaudScaling | |
| **26_activity_surface_collapse** | make_surface_figures | **New.** See caveat 3 |
| **27_activity_surface_gamma** | make_surface_figures | **New.** γ by N and by calendar year |
| **28_activity_surface_oos** | make_surface_figures | **New.** Out-of-sample weighted RMSE, 2023–2026 |

The notebooks also wrote figures the chapter does not currently use. They
include 02_binned_log_log, 03, the 05 linear-axis version, 06/07/08 regime
curves, 09 trajectories, 16, 19, 22b and 23. The 08/09 regime-curve files
from PowerLawImpact (`08_duration_regime_…`, `09_normalised_impact_…`) are
not the chapter's 08/09, which come from BarConstructionRobustness.

## Activity surface (new leading model) — headline numbers

These come from the `activity_surface_*` tables in `tables/`.

**Model.** E[r/σ₂₄ | z, a, N, s] = s · R_N · F(z a^γ / Q_N), with
z = |Q|/V and a = V/V₂₄.

**Fit.** Common γ = 0.362, with month-block bootstrap 95% interval
[0.332, 0.391].

**Full-sample weighted RMSE:**

| Model | Weighted RMSE |
|---|---:|
| Activity surface | 4.72 |
| P&B (γ = 1) | 20.65 |
| Imbalance ratio (γ = 0) | 24.11 |

**Out of sample**, the surface beats P&B in every test year from 2023 to 2026.

**γ varies along both axes:**

- it rises with N, from 0.33 at N = 100 to 0.65 at 16,000;
- it falls over time, from 0.49 in 2023 to 0.21 in 2026.

The full discussion is in `../README.md`.

## Caveats that belong in the write-up

1. **Figure 06.** 68 monthly tick labels overlap; the figure may need a
   sparser axis for print.
   - The free-power exponent drifts down, from about 0.70–0.75 (2021–23) to
     about 0.60–0.65 (2025–26).
   - The log-model WRMSE improvement is positive in almost all months. The
     exception is 2022-11 (FTX), at −82%.
2. **Figure 18, five-minute panel.** The fitted threshold model is unchanged
   until activity is about 3× its median (activity threshold multiple 2.99).
   Every activity quintile therefore sits below the threshold, and the
   liquidity ratio is flat at 1 for all of them.
   - For event bars, the threshold model hits a bound on the imbalance
     threshold.
   - In 5-fold cross-validation, the threshold model does not beat the
     smooth state model (`state_dependent_liquidity_cross_validation_summary.csv`).
   - The paper's threshold claim is therefore weaker on the full sample.
3. **Figure 26.** At N = 100 no cell reaches the master-curve bend. Only
   R_N/Q_N is identified there, so the N = 100 points' position along the
   curve is arbitrary.
   - Weighted RMSE, not scaled RMSE, is the comparison metric.
   - Identified-N exponents for the surface: ξ = 0.14, ψ = 0.58
     (`activity_surface_scaling_exponents.csv`).
4. **Raw returns.** Measured on raw rather than volatility-normalised
   returns, γ < 0, because relative activity proxies the previous day's
   volatility (`activity_surface_robustness.csv`). The volatility-normalised
   response is primary.
5. **Displacement.** The paper's "high-activity bars sit above the curve"
   displacement is reduced, not removed (`activity_surface_displacement.csv`).
6. **α differs slightly between estimators.** The notebook's pooled α (1.54)
   differs from the standalone script's (1.49, `../pb_collapse_2021_01_2026_08/`), owing to
   optimiser starts. ξ, ψ and β agree.
7. **Figures 08 and 09: the tail.** At every event-bar size and every
   time-bar duration, the logarithmic form tracks the body of the curve. In
   the top one or two imbalance bins, however, it falls below the binned
   means, and the free power form is closer there. Any text claiming the
   logarithmic form wins at every scale should be re-checked against the
   notebook's fit tables before it is kept.
