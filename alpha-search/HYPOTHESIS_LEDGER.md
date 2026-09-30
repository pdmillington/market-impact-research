# Hypothesis ledger

This ledger records every predictive hypothesis tested in `alpha-search`,
including failures, so that the number of attempted specifications is never
understated. It is the reference for multiple-testing corrections, deflated
Sharpe ratios and reality checks.

## Rules

1. **Register before running.** A new entry records the economic hypothesis,
   exact feature definition, expected direction, primary bar/clock, primary
   horizon, transformation, training/validation/test periods, selection criterion,
   cost assumptions and rejection conditions *before* any result is seen.
2. **Nothing is deleted.** Failed hypotheses stay, with their specification
   counts. A rejected hypothesis is revived only with a new economic argument,
   recorded as a new entry that links to the old one.
3. **One primary specification per hypothesis.** Other bar sizes, horizons and
   thresholds are robustness checks, reported as a complete grid.
4. **Incremental value.** A conditioner must improve the frozen benchmark
   (bar return plus 15- and 60-minute pretrend for intraday work; past returns
   for hourly work). A significant IC for the combined model is not enough.
5. **Testing budget.** By default: one primary transformation, one primary
   interaction, one primary horizon, three robustness horizons and three adjacent
   bar sizes.
6. **Evaluation protocol.** Nested walk-forward (fit → validate → freeze → test),
   labels purged and embargoed across boundaries, day- or month-block inference,
   and a spaced-observation check when targets overlap.
7. **Simple first.** Training-frozen quantiles, linear or monotone effects and
   prespecified interactions come before flexible models. Flexible models must
   beat them.
8. **Stability over peaks.** Expected sign in most test months, several regimes,
   adjacent horizons and bar sizes, no dependence on a handful of days, and
   survival under the frozen cost assumptions.
9. **Confirmation is on data not used for development**: see the sample register.
10. **Status values**: `proposed` → `registered` → `rejected` |
    `survives-development` → `confirmed` | `retired`.

Workflow: economic hypothesis → exploratory visualisation → frozen
specification → walk-forward validation → replication → single locked test.

## Sample register

| Data | Status | Notes |
|---|---|---|
| BTCUSDT 2019-09 to 2022-12 | **Development** | Calibration and training in every experiment. |
| BTCUSDT 2023-01 to 2026-05 | **Development** | Inspected repeatedly by H-004 to H-010. It is no longer a clean test set. |
| BTCUSDT 2026-06 to 2026-08 | **Locked, single use** | Raw archives are on disk; no analysis has read them. Reserved for one final, frozen candidate. About 90 days, so it can reject but not confirm. |
| BTCUSDT 2026-09 onward | **Prospective** | Accumulates as a genuine forward sample. |
| ETHUSDT, SOLUSDT (all periods) | **Untouched confirmation set** | Built 2026-09-30: trades, canonical fills, events, 1-second grid (to 2026-05), price series and metrics (from 2021-12). Only data-quality checks have read them; no return analysis. Used to replicate hypotheses frozen on BTC. Correlated with BTC, so strong but not fully independent evidence. |

## Frozen economic assumptions

Unless an entry registers otherwise (see `trading-access` below):

- **Fees per side:** taker 5 bps, maker 2 bps (Binance base tier). Taker costs
  are the primary decision basis. Maker fills are not modelled, so maker results
  are an optimistic bound.
- **Prices:** trade prints; no order-book data.
- **Entry:** intraday work enters at the first 200-event endpoint at least one
  second after the decision; hourly work enters one minute after the decision.
- **Funding:** positions held across a settlement pay or receive the settled rate.
- **Horizons:** following the 2026-09-29 decision, new work targets horizons of
  hours to days. Short horizons (1–15 minutes) are pursued only as a maker problem
  with order-book data.

`trading-access`: the researcher can classify as a MiFID professional client and
would trade at Binance's base fee tier.

## Specification count to date

Approximate counts of evaluated cells (a cell is one combination of model or
feature, bar size, horizon, side, threshold or variant that produced a reported
statistic). These are the denominators for later corrections.

| Entry | Experiment | Cells evaluated |
|---|---|---:|
| H-001 | `initial-impact-alpha-results` (2019–2021 sandbox) | not enumerated; roughly hundreds |
| H-002 | `short_event_screen_v1` | 276 (84 cross-scale) |
| H-003 | `excess_impact_v1` | 1,500 grid + 120 incremental |
| H-004 | `impact_surface_residual_alpha_v1` | 1,200 IC + 3,600 strategy |
| H-005 | `rolling_pb_residual_ic_v1`, `rolling_pb_residual_response_v2` | 48 + 576 |
| H-006 | `residual_decomposition_v1` | 192 OOS model cells + 576 tail cells |
| H-007 | `stress_episodes_v1` | 3 rates × 9 rules × 4 holds, plus 16 splits |
| H-008 | `stress_episodes_v2` | as H-007, plus 2 post-hoc spread splits |
| H-009 | `funding_positioning_v1` | 68 univariate + 32 OOS + 96 tail + 6 settlement |
| H-010 | `crowding_backtest_v1` | 40 variants + 10 attribution variants + 16 drop-one |
| | **Total (enumerated rows)** | **about 9,000 cells, plus H-001** |

Impact-model selection experiments (`pb_surface_scaling_v1`, the v2–v4
extensions, `rolling_power_surface_v1`) are explanatory calibration work, not
return prediction, and are not counted. `pb_lagged_impact_extension_v4` fed the
lagged residual used in H-005 and H-006.

## Entries

### H-001 · Marginal impact and impact residual (initial sandbox)
- **Status:** retired (superseded by H-004 to H-006).
- **Hypothesis:** high marginal impact continues and excess impact reverses.
- **Data:** training September 2019 to August 2020; validation September 2020 to
  June 2021; later sandbox July to December 2021.
- **Outcome:** mixed and asymmetric. There was an 8,000-event buy-side
  marginal-impact contrast and a 5-minute sell-side overshoot. Both were sensitive
  to weighting, and costs were not deducted.
- **Report:** `reports/initial-impact-alpha-results.md`.

### H-002 · Short-event-bar feature screen
- **Status:** retired as a screen; its main finding carried into H-006.
- **Hypothesis:** preregistered families (imbalance, volume, activity,
  volatility, impact, microstructure, pretrend, flow persistence, fitted impact)
  predict 12,000- to 16,000-event returns.
- **Outcome:** aligned 12,000-event pretrend reversal was the dominant
  cross-scale effect. The fitted impact residual was weak and unstable.
  Benjamini–Hochberg over 276 cells.
- **Report:** `reports/short-event-screen-initial-results.md`.

### H-003 · Excess impact as a conditioner of pretrend reversal
- **Status:** rejected (sell-side effect not stable in later data).
- **Outcome:** sell-side validation increment of 3–5 bps survived the
  false-discovery screen but weakened in the later sandbox. The buy side was
  unstable.
- **Report:** `reports/excess-impact-and-decay-initial-results.md`.

### H-004 · Two-dimensional impact-surface residual
- **Status:** rejected.
- **Outcome:** gross top-quintile contrarian return of 0.1–0.7 bps at 10–15
  minutes. The sign depended on construction.
- **Report:** `reports/impact-surface-residual-alpha-v1-initial-results.md`.

### H-005 · Scale-free P&B residual (rolling)
- **Status:** rejected via H-006.
- **Outcome:** mean monthly IC −0.022 at 8,000 events and 15 minutes; the sign
  changed with bar size.
- **Report:** `reports/pb-scale-free-impact-v1-initial-results.md`.

### H-006 · Residual decomposition (Step 1)
- **Status:** rejected. The residual is retired as an alpha factor.
- **Registered question:** does $X=R-P$ add information beyond $R$, $P$,
  pretrend and linear flow, and does any tail clear costs?
- **Outcome:** the flow-predicted component $P$ also reverses, so $X$ discards
  information. $P$ adds no OOS information beyond return, pretrend and flow.
  Tail edge above taker costs comes from 10 stress days; no tail variant covers
  taker costs on ordinary days (maximum 8.0 bps across all variants).
- **Development evidence carried forward:** 15- and 60-minute pretrend reversal
  (OOS IC about 0.04–0.06), which is the intraday benchmark in rule 4.
- **Report:** `reports/residual-decomposition-v1-initial-results.md`.

### H-007 · Stress-episode reversion, fast triggers (Step 3 v1)
- **Status:** rejected.
- **Outcome:** after about 110 bps moves only about 10 bps retraces. The best
  rule (enter at the trigger, hold 15 minutes) nets −2.7 bps. Open interest does
  not separate forced from informed flow.
- **Report:** `reports/stress-episodes-v1-initial-results.md`.

### H-008 · Stress-episode reversion, 15- and 60-minute triggers (Step 3 v2)
- **Status:** rejected. Specified after H-007 failed, motivated by H-006.
- **Outcome:** the best rule nets +1.6 bps with an interval spanning zero; only
  2024 is positive.
- **Post-hoc lead** (not a result): high spread-proxy episodes revert in 2023–26
  but not in 2021–22. Recorded as proposed entry H-011.
- **Report:** `reports/stress-episodes-v1-initial-results.md`.

### H-009 · Perp crowding: funding, premium and basis (Step 4)
- **Status:** survives development, economically insufficient alone.
- **Registered design:** fixed feature groups and eight nested models, 12-month
  rolling OLS, hourly decisions, 1–24 hour targets, carry-inclusive tails.
- **Outcome:** 24-hour contrarian IC about −0.08 in the test period, with the
  same sign in 2021–22 (about −0.15). It adds OOS value beyond past returns. The
  effect decays every year (premium IC −0.18 in 2022 to −0.02 in 2026) and is
  about zero over the last 12 months. Long/short ratios flip sign between
  periods. There is no settlement-timing effect.
- **Report:** `reports/funding-positioning-v1-initial-results.md`.

### H-010 · Implementable crowding and positioning backtest
- **Status:** survives development, fragile; not forward-tested (researcher
  decision, 2026-09-29).
- **Registered primary:** combined model, 24 hours, tail rule, 12-month window.
- **Outcome:** 18%/yr with maker fills (Sharpe 0.87) and 14%/yr with taker fills
  (Sharpe 0.69), beta 0.07. Attribution shows the edge comes from the long/short
  ratio z-scores, which flipped sign after 2022. Last 12 months: 20 of 40
  variants positive.
- **Open questions before any revival:** the cause of the ratio sign flip, a
  maker fill model, and replication on ETH/SOL.
- **Report:** `reports/crowding-backtest-v1-initial-results.md`.

### H-011 · Liquidity-withdrawal reversal (proposed)
- **Status:** proposed; parked. It is a short-horizon effect, so it waits for
  order-book data under the frozen assumptions.
- **Origin:** post-hoc lead from H-008. It must be tested on data not used to
  find it (ETH/SOL, or prospective BTC) with a pre-registered, stationary spread
  measure.

## Template for new entries

```markdown
### H-0NN · Short name
- **Status:** registered (YYYY-MM-DD)
- **Economic hypothesis:** mechanism and why it should persist.
- **Feature definition:** exact formula, units, look-back, publication lag.
- **Expected direction:** sign of the effect on the primary target.
- **Primary specification:** bar or clock; horizon; transformation; interaction.
- **Robustness grid:** adjacent horizons and bar sizes (the testing budget).
- **Benchmark:** frozen model the hypothesis must improve on.
- **Periods:** training / validation / test; purge and embargo.
- **Statistics:** primary (e.g. mean monthly OOS ΔIC, month-block t) and
  economic (net return under the frozen cost assumptions, stress-day concentration).
- **Rejection conditions:** stated before running.
- **Confirmation plan:** ETH/SOL replication; locked BTC sample.
- **Cells to be evaluated:** count added to the table above.
- **Outcome:** filled in after the run; link to report.
```
