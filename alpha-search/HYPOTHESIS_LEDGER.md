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
| BTCUSDT `bookTicker` 2023-05 to 2023-07 | **Order-book exploration** (decision 2026-09-30) | Used for E-004 (order-book features) and for mid-proxy validation. Trade-based hypotheses tested on 2023–26 include these months; that overlap is limited to order-book exploration and is noted. |
| BTCUSDT `bookTicker` 2023-08 to 2024-04 | **Order-book test** | Reserved for registered order-book hypotheses. |
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
- **Horizons:** on 2026-09-30 the researcher chose to pursue intraday horizons
  (5–30 minutes) with **passive entry**, the explicit aim being to overcome the
  taker round trip. The passive fill model is part of each registration (see
  H-012). Hours-to-days work remains open (H-012d).

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
| H-012 | `intraday_flow_v1` (rejected) | 12 cells + 4 benchmark cells, × 3 tail fractions × 2 execution models |
| E-001 to E-003, V-001 | exploration and validation | logged in the exploration log (descriptive slices, not tests) |
| H-014 to H-016 | `seconds_signals_test_v1` (proxy run complete; true-mid pending) | 8 + 7 + 7 = 22 cells |
| H-017 | quoting simulation (registered) | 12-combination calibration grid + 2 strategies × 2 instruments + 3 sensitivities |
| C-010 | H-010 confirmation on ETH and SOL (failed; H-010 retired) | 2 symbols × 1 frozen primary |
| H-018 | execution simulation (registered) | 12-combination calibration grid per instrument + 4 strategies × 3 instruments + 4 sensitivities |
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
- **Status:** survives development on BTC; **failed confirmation (C-010, 2026-09-30)** on ETH and SOL. Retired.
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

### H-012 · Persistent aggressive flow continues (intraday)
- **Status:** registered (2026-09-30).
- **Economic hypothesis:** large orders are executed as metaorders split over
  time (Lillo–Farmer). Impact is concave and partly transient, so a window of
  one-sided aggressive flow signals demand that is not yet complete and not
  fully priced. The competing hypothesis (exhaustion or overreaction) predicts
  reversal.
- **Feature definitions** (see "Trailing-window flow features" in
  `FEATURE_CATALOGUE.md`; BTCUSDT perp, from `second_grid_v1`):
  - primary variable $x_W$ = `window_normalised_imbalance_W` $=Q_W/V^{\mathrm{ref}}_W$;
  - primary transformation `z30(x_W)`;
  - persistence $P_W$ = `window_sign_persistence_W_h` with $h$ = 1 minute,
    used only in the interaction;
  - scaling robustness: `z30(z_W)` with $z_W$ = `window_imbalance_ratio_W`.
- **Expected direction:** continuation (positive coefficient on `z30(x_W)`).
- **Decision clock:** every 5 minutes on UTC boundaries, BTCUSDT, February 2021
  to May 2026.
- **Primary specification:**
  - window W = 60 minutes; target horizon H = 15 minutes;
  - OLS on benchmark + `z30(x_60)`;
  - 12-month rolling fit with an embargo of H + 5 minutes;
  - winsorisation bounds from training only.
- **Benchmark (frozen, from H-006):**
  - return over the last 5 minutes;
  - returns over the last 15 and 60 minutes;
  - all divided by trailing-24h realised volatility (1-minute returns);
  - plus that volatility.
- **Primary interaction (one):** + `z30(x_60)` × $P_{60}$, primary cell only.
- **Robustness grid:** W ∈ {15, 60, 240 min} × H ∈ {5, 15, 30 min} (9 cells,
  including the primary); the scaling robustness at the primary; a spaced
  check at the primary using non-overlapping decisions every H.
- **Periods:** test months January 2023 to May 2026. These are development data,
  so results rank specifications rather than confirm them.
- **Execution model (frozen, primary): passive entry.**
  - At t, a limit order is placed at the last trade price L on the forecast
    side.
  - It fills only if a later trade prints strictly through L (below for a
    buy, above for a sell) within 60 seconds. Otherwise there is no trade.
  - Exit is taker at fill time + H, at the last trade price.
  - Cost: 2 bps entry + 5 bps exit = 7 bps round trip.
  - The trade-through rule avoids assumptions about queue position and
    includes adverse selection, because passive fills occur when price moves
    against the order.
  - Reported alongside, not selected on: a taker-entry version (entry at the
    last trade price one second after t; 10 bps round trip) and the fill rate.
- **Signal rule:** trade when |forecast| is in the training top 5%. The 1% and
  20% tails are robustness checks. Trades are evaluated individually and may
  overlap.
- **Statistics:**
  - Primary: mean monthly OOS ΔIC (Spearman) against the benchmark at the
    primary cell, month-block t, and the share of months with ΔIC > 0.
  - Economic: mean net edge per filled trade under the passive model, **on
    ordinary days** (excluding the best 10 days), plus the best-10-day share,
    the fill rate, trades per day and results by year.
- **Rejection conditions (any one):**
  1. Mean ΔIC ≤ 0, or ΔIC > 0 in 60% of months or fewer.
  2. The flow coefficient is negative in most monthly refits. That would be
     reversal, a different hypothesis; it is recorded, not flipped.
  3. Ordinary-day net edge per filled trade under the passive model ≤ 0 at the
     primary cell.
  4. ΔIC is negative in more than 3 of the 8 robustness cells.
- **Confirmation plan:** re-run the frozen specification on ETHUSDT and SOLUSDT
  (ΔIC > 0 with a positive flow coefficient in both), then a single BTC holdout
  test (June–August 2026) on the economic statistic.
- **Cells to be evaluated:** 9 + 1 interaction + 1 scaling + 1 spaced = 12, each
  with 3 tail fractions and 2 execution models reported.
- **Outcome (2026-09-30): rejected** under conditions 1 and 4.
  - Adding flow lowers OOS IC in all nine W × H cells: ΔIC −0.001 to −0.007,
    t −2.2 to −3.6, positive in 22–44% of months. The interaction and scaling
    checks are also negative; the spaced check is flat.
  - Recorded, not acted on: flow coefficients reverse at W = 15 min (negative
    in 80–100% of refits), are split at W = 60, and continue at W = 240
    (positive in 76–80%).
  - Economics: tails earn 0.2–4.5 bps gross mid-to-mid and about 0 bps on
    ordinary days, including the benchmark. Passive net −7 to −8 bps, taker
    net −9 to −10 bps. Passive entry costs about 1.3 bps of adverse selection
    even under the (optimistic) registered fill model.
  - **Fill-model flaw found:** the registered limit price (the last trade price)
    is at the ask when the last aggressor was a buyer, so a "passive" buy there
    would really be taker. The median simulated fill came after 1 second, with
    an 86% fill rate. The flaw makes passive results optimistic and does not
    affect the statistical rejection. The corrected model (execution model v2,
    below) applies to future registrations.
  - Report: `reports/intraday-flow-v1-initial-results.md`.

#### Execution model v2 (for registrations after H-012)
- A passive buy rests at the **bid proxy**: the price of the most recent
  seller-initiated event before t. A passive sell rests at the **ask proxy**:
  the price of the most recent buyer-initiated event.
- The order fills only if a later trade prints strictly through that level
  within the timeout. Otherwise there is no trade.
- Everything else as H-012: maker 2 bps entry, taker 5 bps exit.
- Implemented from event data (aggressor sign and last price), not the
  1-second grid, which does not keep side-specific prices.

### H-013 · Cross-asset flow spillover (proposed)
- **Status:** proposed; to be registered after H-012.
- **Economic hypothesis:** hedgers and informed traders use the most liquid
  correlated instrument, so flow in one coin predicts continuation in
  correlated coins (cross-impact; Benzaquen et al. 2017, Pasquariello & Vega
  2015).
- **Design constraint:** develop with BTC returns as the only target, using a
  market-wide flow factor built from BTC, ETH and SOL flow. Confirm by
  predicting ETH and SOL returns. ETH and SOL flow will have been seen, but
  not their returns (partial contamination, stated in advance).

### H-012d · Persistent aggressive flow, daily horizon (proposed)
- **Status:** proposed; lower priority.
- **Specification:** as H-012, but on the hourly Step 4 panel with W ∈
  {8, 24, 72 h} and H ∈ {8, 24, 72 h}, primary W = H = 24 h, the benchmark from
  the common slow-horizon protocol, and a taker-cost netted backtest.

### Common seconds-scale protocol (registered 2026-09-30; applies to H-014 to H-016)
- **Data:** BTCUSDT reconstructed events. Test months January 2023 to May 2026,
  **excluding May–July 2023** (the order-book exploration window, in which
  V-001 measured sweep continuation): 38 months.
- **Mid:** the trade-based proxy is primary (E-002 definition). Its known bias
  (V-001: 1–9 level continuation overstated by about 15–20%) is stated in
  advance. A true-mid re-run on August 2023 to April 2024 (`bookTicker`) is a
  robustness cell.
- **Unit:** each aggressive event. Means use all events; monthly Spearman ICs
  use a fixed-seed sample of 2 million events per month.
- **Horizons:** τ = 5 s primary; 1, 15 and 60 s robustness.
- **Frozen cuts** from exploration, not re-estimated: activity terciles
  [0.8234, 1.5731] (E-002); absorption terciles [0.2609, 1.6753] (E-003);
  burst deciles from E-002.
- **Inference:** month blocks: mean across months, t-statistic and the share
  of months with the expected sign.
- Main effects and interactions are judged separately.
- **No economic gate on BTC** (0.04 bps spread against a 2 bps maker fee).
  Reported diagnostic: *protective value*, the mean maker markout of the fills
  that would be avoided by pulling the at-risk quote for 5 s after a flagged
  event.
- **Confirmation:** the frozen specifications on ETHUSDT and SOLUSDT events
  2023–26, then the wider-spread alts.

### H-014 · Sweep toxicity
- **Status:** registered (2026-09-30).
- **Economic hypothesis:** an aggressive order that walks several price levels
  reveals urgency (information or forced flow) and a thin book. The mid
  continues in its direction and the liquidity providers it hit are adversely
  selected. In high-activity markets the book refills faster and sweeps carry
  less information, so toxicity is lower.
- **Features:** `event_price_levels` L; aggressor sign s; `event_notional`
  (USD) for size control; activity state (5-minute event rate ÷ 24-hour
  rate); resiliency (E-003 definition).
- **Primary (main effect):** monthly Spearman IC of s·L with the post-event
  mid change at τ = 5 s. Expected positive.
- **Primary interaction:** for events with L ≥ 2, maker markout in the high
  activity tercile minus the low tercile, per month. Expected positive.
- **Secondary registered cell:** the same difference by tercile of resiliency
  orthogonalised to activity. The residual is from a regression of resiliency
  on log activity, fitted on 2022 and frozen; tercile cuts from 2022
  residuals. Expected positive.
- **Robustness:** τ ∈ {1, 15, 60} s for the main effect; a size-controlled IC
  (s·L within each of the 5 notional classes, averaged); the true-mid re-run.
- **Rejection conditions:**
  1. Main effect: mean IC ≤ 0, or positive in 80% of months or fewer.
  2. Main effect: IC not positive at 2 or more of the 3 robustness horizons.
  3. Interaction: mean difference ≤ 0, or positive in 60% of months or fewer.
  4. Secondary cell: judged by condition 3 on its own; failing it means
     resiliency adds nothing beyond activity.
- **Cells:** 8.
- **Outcome (2026-09-30, trade-based mid; 38 test months): main effect and
  activity interaction pass; secondary cell rejected.**
  - Main effect: IC of s·L at τ = 5 s is 0.191 (t 31.6), positive in 38 of 38
    months. It is positive at every robustness horizon (τ = 1 s 0.34, 15 s
    0.11, 60 s 0.06). Size-controlled IC 0.245.
  - Activity interaction: +0.10 bps (t 4.6), positive in 71% of months, so it
    passes. **It is decaying:** yearly means 0.24 (2023), 0.15 (2024), 0.01
    (2025), −0.04 (2026).
  - Secondary (resiliency orthogonalised to activity): −0.12 bps, expected
    sign in only 20% of 35 valid months, so **rejected**. Resiliency adds
    nothing beyond activity, and the residual goes the wrong way. In three
    months a frozen-cut tercile was empty, because resiliency's level
    drifted from a 2022 median of about −0.35 to about −0.70 in 2025.
  - Protective value (diagnostic): same-side fills within 5 s after a 5+ level
    sweep have maker markout −0.33 bps, against −0.09 bps otherwise.
  - True-mid robustness: pending (`bookTicker` download in progress).
  - Report: `reports/seconds-signals-test-v1-results.md`.

  - **Confirmation (ETHUSDT, SOLUSDT; frozen BTC specification, same 38
    months): replicated.** Same verdicts on all four conditions.
    - Main-effect IC at 5 s: ETH 0.183, SOL 0.170, positive in 38 of 38
      months for both.
    - Activity interaction: ETH +0.14 bps (97% of months), SOL +0.26 bps
      (95%). It decays on ETH, as on BTC, but not on SOL.
    - Secondary cell fails on both.
    - Protective value: ETH −0.34 against −0.17 bps (2 times as toxic).
      **SOL reverses: −0.15 against −0.24 bps.** SOL's tick is about 1 bp
      against about 0.02 bp for BTC, so a price-level count is not comparable
      across instruments; sweep depth in bps is the scale-free measure for
      future work.
### H-015 · Level-depletion break
- **Status:** registered (2026-09-30).
- **Economic hypothesis:** repeated same-side executions at an unchanged price
  deplete the visible queue. The more hits, and the more volume absorbed
  relative to the level's typical depth, the more likely the level breaks:
  the mid continues in the aggressor direction and the remaining makers are
  adversely selected. (E-003 contradicted the "iceberg holds" alternative.)
- **Features** (single-level events): `level_hits` h, the run length of
  same-side single-level events at an unchanged price; `absorption_ratio` A,
  run quantity ÷ typical same-side depth per level (trailing 30-minute
  geometric mean of quantity ÷ levels over same-side sweeps).
- **Primary (main effect):** monthly Spearman IC of s·h with the post-event mid
  change at τ = 5 s. Expected positive.
- **Primary interaction:** for runs with h ≥ 3, continuation in the high
  absorption tercile minus the low tercile, per month. Expected positive.
- **Robustness:** τ ∈ {1, 15, 60} s; maker markout in place of continuation
  (expected negative in s·h); the true-mid re-run.
- **Rejection conditions:**
  1. Main effect: mean IC ≤ 0, or positive in 80% of months or fewer.
  2. Main effect: IC not positive at 2 or more of the 3 robustness horizons.
  3. Interaction: mean difference ≤ 0, or positive in 60% of months or fewer.
- **Cells:** 7.
- **Outcome (2026-09-30, trade-based mid; 38 test months): main effect and
  absorption interaction pass.**
  - Main effect: IC of s·h at τ = 5 s is 0.105 (t 44), positive in 38 of 38
    months; positive at every robustness horizon. IC of hits with maker
    markout −0.024, expected sign in 95% of months.
  - Absorption interaction: +0.25 bps (t 11.5), positive in 95% of months.
    **It is strengthening:** yearly means 0.14 (2023), 0.24, 0.28, 0.40 (2026).
  - Protective value (diagnostic): after 6+ hit runs, markout −0.26 against
    −0.23 bps otherwise, a small difference.
  - True-mid robustness: pending.

  - **Confirmation: replicated** on ETH and SOL (all three conditions).
    - Main-effect IC at 5 s: ETH 0.108, SOL 0.150 (38 of 38 months).
    - Absorption interaction: ETH +0.36 bps (100% of months), SOL +1.00 bps
      (100%).
### H-016 · Burst inverted U
- **Status:** registered (2026-09-30).
- **Economic hypothesis:** aggressive order flow is self-exciting, so moderate
  one-sided bursts continue over seconds. Extreme bursts signal exhaustion and
  reverse over tens of seconds.
- **Feature:** `burst_signed_intensity_5s` = (buy − sell events) over the
  preceding 5 s (current event excluded) ÷ (trailing-24-hour event rate ×
  5 s). Deciles frozen from E-002.
- **Primary A (continuation):** deciles 2–4 and 7–9; the mid change from just
  before the event, signed by the burst direction, at τ = 5 s. Expected
  positive.
- **Primary B (exhaustion):** deciles 1 and 10; the same at τ = 60 s.
  Expected negative.
- **Primary interaction:** primary A in the low activity tercile minus the high
  tercile. Expected positive.
- **Robustness:** primary A at τ ∈ {1, 15} s; primary B at τ = 15 s; the
  true-mid re-run.
- **Rejection conditions:**
  1. Primary A: mean ≤ 0, or positive in 70% of months or fewer.
  2. Primary B: mean ≥ 0, or negative in 60% of months or fewer.
  3. Interaction: mean ≤ 0, or positive in 60% of months or fewer.
  The inverted U is rejected if either primary fails.
- **Cells:** 7. The decile grouping was chosen from E-002's shape and is frozen
  here as the guard against re-fitting.
- **Outcome (2026-09-30, trade-based mid; 38 test months): inverted U
  rejected (primary B fails); primary A continuation and the activity
  interaction pass.**
  - Primary A (moderate bursts, τ = 5 s): +0.17 bps (t 33), 38 of 38 months.
  - Primary B (extreme bursts, τ = 60 s): **+0.23 bps**, the wrong sign;
    negative in only 26% of months. Extreme bursts *continue* in 2023–26,
    unlike 2022, where they reversed.
  - Activity interaction: +0.15 bps (t 21), 38 of 38 months.
  - Under the registered rule the inverted U is rejected. The surviving
    finding is monotone burst continuation, stronger in quiet markets. It is
    recorded, not re-registered under a new name.

### H-017 · Signal-protected passive quoting (economic test)
- **Status:** registered (2026-09-30).
- **Economic hypothesis:** a small passive quoter earns the spread but is
  adversely selected. The confirmed seconds-scale signals identify moments
  when resting quotes are about to be picked off, and pulling the at-risk side
  then improves net P&L after maker fees. On contracts where one tick is large
  relative to fees, the protected quoter can be profitable.
- **Instruments:** SOLUSDT primary; ETHUSDT control (tick ≈ 0.03 bps). Wider-
  spread alts only if SOL shows signal value.
- **Data:** `bookTicker` level-1 quotes plus reconstructed events.
  - Calibration: 2023-05-16 to 2023-07-31.
  - Test: 2023-08-01 to **2024-03-31**. Binance's archive ends on 2024-04-01
    around 07:00 UTC.
- **Simulation (frozen):**
  - *Size:* the contract's minimum-notional unit, assumed not to affect the
    market.
  - *Quoting:* one order at the best bid and one at the best ask; when a best
    price moves, cancel and re-quote at the back of the new queue.
  - *Latency:* 100 ms from market event to order action (primary).
  - *Conservative queue:* on joining, we are behind all displayed quantity at
    that price; only executions at our price reduce the queue ahead;
    cancellations are assumed to be behind us. We fill when executed quantity
    at our price exceeds the queue ahead, or when price trades through our
    level.
  - *Inventory:* limit ±5 units; at a limit, quote only the reducing side; no
    taker flattening. Marked to mid; funding charged across settlements.
  - *Fees:* 2 bps maker per fill.
- **Strategies:**
  - S0 naive: continuous two-sided quoting within inventory limits.
  - S1 protected: pull the at-risk side for T seconds after:
    - (a) a sweep of depth ≥ D bps (scale-free), pulling the threatened side;
    - (b) a level run of ≥ k hits in the high absorption tercile at our price;
    - (c) burst intensity in deciles 7–10 or 1–4, pulling the side against
      the burst.
  - S1 grid: D ∈ {2, 5, 10} bps × T ∈ {2, 5} s × k ∈ {3, 6}. Chosen by
    calibration net P&L per day and frozen before any test month is simulated.
- **Metrics:** net P&L per day, decomposed into spread, adverse selection
  (markouts at 1/5/60 s), fees and inventory; fills per day; time quoting;
  daily Sharpe; maximum drawdown; P&L excluding the best and worst 10 days;
  and results by month with the tick in bps.
- **Pass conditions (SOL):**
  1. S1 − S0 net P&L per day > 0, daily-block t > 2, positive in more than 60%
     of test months.
  2. S1 net P&L per day > 0, daily-block t > 2, positive in more than 60% of
     test months, and positive excluding the best and worst 10 days.
  3. Reported, not a gate: the break-even tick in bps.
- **ETH control:** condition 2 expected to fail; condition 1 reported.
- **Sensitivities (not selected on):** latency 250 ms; an optimistic queue
  model (cancellations spread across the queue); inventory limit ±2.
- **Limitations:** unknown true queue position; no market impact of our
  orders; level 1 only; a short year in which SOL's rally makes regime and
  tick size move together.
- **Confirmation:** time-based, on prospectively captured live quotes (capture
  to be built).
- **Outcome:** pending.

### C-010 · Confirmation of H-010 (crowding and positioning) on ETHUSDT and SOLUSDT
- **Status:** registered (2026-09-30), before any ETH or SOL hourly return is
  examined.
- **Frozen specification:** the H-010 primary, unchanged: combined model (`all`
  feature groups), 24-hour horizon, tail rule (top 20%), 12-month rolling OLS,
  hourly decisions, staggered 24 tranches, carry included. The same feature
  definitions, including zero-imputation of missing ratio features.
- **Data:** each symbol's own panel (perp prices from its 1-second grid; its own
  funding, premium, spot and metrics). Metrics start 2021-12, so the first
  training window is 2022. Test months 2023-01 to 2026-05, as for BTC.
- **Contamination note:** ETH and SOL *seconds-scale* returns were used to
  confirm H-014 to H-016. Hourly and daily returns have not been examined.
- **Pass conditions (each symbol):**
  1. Taker-cost net Sharpe > 0 over the test period.
  2. Beta-adjusted alpha (daily P&L regressed on the symbol's own daily
     return) > 0.
  3. Reported, not gates: the maker-cost result, yearly results, the last 12
     months, and the drop-one attribution of the long/short ratio features.
- **H-010 is confirmed** only if both symbols pass conditions 1 and 2. The
  locked BTC holdout (June–August 2026) is considered only after that.
- **Outcome (2026-09-30): H-010 not confirmed.**

  | | BTC (reference) | ETH | SOL |
  |---|---:|---:|---:|
  | Taker net return | 14.4%/yr | −2.0%/yr | +1.3%/yr |
  | Taker Sharpe (condition 1) | 0.69 | **−0.08 ✗** | **+0.04 ✓** |
  | Beta-adjusted taker alpha (condition 2) | 11.4%/yr, t 1.0 | **−3.2%/yr ✗** | **−0.5%/yr ✗** |
  | Maker Sharpe | 0.87 | 0.05 | 0.14 |
  | Taker, last 12 months | +6.1%/yr | −27.7%/yr | +5.0%/yr |

  - ETH fails both conditions; SOL fails condition 2. H-010 is **not
    confirmed**, so the locked BTC holdout will not be spent on it. Even on
    BTC its beta-adjusted alpha is t ≈ 1.0.
  - Attribution (reported): on ETH, dropping the long/short ratio features
    makes results worse, as on BTC. On SOL, the `baseline_positioning` model
    earns Sharpe ≈ 0.9 at taker costs, but survives removal of the ratios
    (0.73). This is a post-hoc observation from a non-primary model. It is
    recorded, not a finding, and would need its own registration and new
    data.
  - Data note: SOL's taker long/short ratio contains zeros (log = −∞). As
    registered, non-finite ratio features are set to 0, the same treatment as
    missing values on BTC.
  - Configs: `configs/funding_positioning_v1_{ETHUSDT,SOLUSDT}.json`,
    `configs/crowding_confirmation_c010_{ETHUSDT,SOLUSDT}.json`; outputs in
    `reports/crowding_confirmation_c010_*`.

### H-018 · Signal-timed execution (implementation shortfall)
- **Status:** registered (2026-09-30).
- **Economic hypothesis:** at base-tier fees, crossing costs about 5 bps plus
  half the spread per child order. Passive execution saves fees but suffers
  adverse selection and timing risk. The confirmed seconds-scale signals
  (H-014, H-015, H-016 moderate continuation) predict short-term direction and
  should tell the executor when to cross and when to wait or pull, lowering
  mean shortfall.
- **Parent orders (synthetic, alpha-independent, frozen seed):** one per 2-hour
  block at a uniformly random time, direction from a fair coin (seed
  recorded). Size $10,000 notional primary ($50,000 robustness). Window
  W = 30 minutes primary (60 robustness); the remainder is crossed at the
  deadline.
- **Benchmark price:** the true quote mid at arrival t₀.
- **Data:** `bookTicker` quotes plus reconstructed events for BTCUSDT, ETHUSDT
  and SOLUSDT. Calibration 2023-05-16 to 2023-07-31; test 2023-08-01 to
  2024-03-31.
- **Fill and latency model (frozen, shared with H-017):**
  - 100 ms latency.
  - Conservative queue: we join behind all displayed quantity and only
    executions ahead reduce it; we fill on executions beyond our queue position
    or on a trade-through.
  - Taker fills at the touch while level-1 quantity covers the size; any
    excess fills one tick worse (reported).
  - Fees: maker 2 bps, taker 5 bps.
- **Strategies:**
  - E0 immediate taker.
  - E1 passive at our best price, re-pegged, with a taker fallback at the
    deadline.
  - E2 signal-timed: E1 plus
    - *run-away → cross now:* a sweep ≥ D bps in our direction; a level run of
      ≥ k hits on the opposite best; a burst in our direction in deciles 7–9 or
      2–4;
    - *adverse → pull T s, re-post at the new best:* a sweep ≥ D bps against
      us.
    - Grid D ∈ {2, 5, 10} bps × T ∈ {2, 5} s × k ∈ {3, 6}, chosen per
      instrument by calibration mean IS and frozen.
  - E3 (secondary): E2, with the passive start delayed by up to W/2 when the
    frozen H-012 return-reversal benchmark forecasts a favourable move.
- **Metrics:** mean IS (bps) and its decomposition (spread, fees, drift);
  day-clustered standard errors; IS standard deviation, 95th and 99th
  percentiles; share completed passively; time to completion; by month.
- **Pass conditions:**
  1. IS(E0) − IS(E1) > 0, day-block t > 2, on at least 2 of 3 instruments.
  2. IS(E1) − IS(E2) > 0, day-block t > 2, positive in more than 60% of months,
     on at least 2 of 3 instruments, and not significantly negative (t < −2)
     on any.
  3. E2's 95th-percentile IS no more than 1 bp worse than E1's (reported with
     condition 2).
  4. E3 − E2 is judged by condition 2's rule (secondary).
- **Sensitivities (not selected on):** latency 250 ms; the optimistic queue
  model; $50k size; W = 60 minutes.
- **Limitations:** synthetic random orders (real alpha-driven orders often trade
  with momentum, which raises adverse selection); level-1 data only; a short
  year.
- **Economic translation (reported):** IS saving × a strategy's turnover.
- **Build note:** the shared fill model is validated on H-017/H-018
  calibration months (fill rates and queue times plausible) before either
  registered test simulates a test month.
- **Outcome:** pending.

## Exploration log

Exploratory analysis generates hypotheses; it tests nothing. Each entry records
what was sliced, so that any hypothesis it inspires is registered with its
origin known and the slices counted.

  - **Confirmation: same pattern** on ETH and SOL. Moderate bursts continue
    (ETH +0.19, SOL +0.29 bps at 5 s, 38 of 38 months). Extreme bursts also
    continue (60 s: ETH +0.22, SOL +0.53 bps), so the inverted U is rejected
    on all three instruments. The activity interaction passes on both.
### E-001 · H-012 diagnostics (2026-09-30)
- **Script:** `scripts/explore_intraday_flow.py`; outputs in
  `reports/intraday_flow_v1/exploration/`; notebook
  `notebooks/IntradayFlowExploration.ipynb`.
- **Slices examined:**
  - univariate ICs of 13 features × 3 horizons × 2 periods;
  - flow–return correlations;
  - ICs of flow orthogonalised to the benchmark (3 windows × 3 horizons × 2
    periods);
  - decile shapes (3 windows × raw/residual × 3 horizons);
  - 3 × 3 return–flow divergence grids;
  - volatility-tercile and year splits;
  - adverse selection by passive fill delay.
- **Observations (development data, not results):**
  - Flow is mostly a proxy for the same-window return (Spearman 0.74–0.78),
    and on its own is a reversal signal (IC −0.02 to −0.05).
  - Flow orthogonalised to returns has a small continuation IC at 15–60 minute
    windows: +0.009 to +0.011, t ≈ 4 in 2023–26, t ≈ 1 in 2021–22. At
    240-minute windows it reverses. The residual continuation appears in low
    volatility and reverses in high volatility.
  - All decile and divergence-cell means are below 1 bps at 15 minutes.
  - Passive fills are followed by adverse mid moves of 0.5–1.4 bps, worse the
    later the fill. Orders that never fill would have earned about +4.5 bps.

### E-002 · Seconds-scale signals, BTC 2022 only (started 2026-09-30)
- **Decisions (researcher, 2026-09-30):**
  - Exploration is restricted to BTCUSDT **2022**, so that 2023–May 2026 stays
    fresh for registered seconds-scale tests.
  - Primary horizon τ = 5 s; robustness τ ∈ {1, 15, 60} s.
  - Three pre-specified conditioning states (activity, liquidity, volatility),
    with at most one pre-registered interaction per signal.
  - Download the BTC `bookTicker` year (2023-05 to 2024-04) for level-1
    queue imbalance and mid validation. That period lies inside the test
    window, so it is used only for registered tests and for proxy
    validation, not for exploration.
- **Families examined:**
  - S1 sweeps (multi-price fills), with realised depth;
  - S2 aggressive bursts;
  - S3 level depletion;
  - S4 flip-gap spread.
- **Targets:**
  - signed change in the mid proxy (mean of the last buyer and last seller
    trade prices) over τ, aligned with the aggressor or burst direction;
  - **maker markout**: profit to the liquidity providers who were hit,
    −sign × log(mid(t+τ) / VWAP), in bps.
- **Script:** `scripts/explore_seconds_signals.py`; outputs in
  `reports/seconds_signals_e002/`; notebook
  `notebooks/SecondsSignalsExploration.ipynb`. State tercile cuts are from
  January 2022.
- **Observations (2022, 523 million events; development data, not results):**
  - *Every aggressive event is followed by continuation* of the mid proxy:
    about 0.2 bps for single-level events and about 0.8 bps for 2+ levels at
    5 s, largely persistent to 60 s. Rank IC of signed levels at τ = 1/5/15/60 s
    is 0.20 / 0.10 / 0.06 / 0.03, positive in 12 of 12 months.
  - *Maker markout* (profit to the liquidity providers who were hit) is
    negative: −0.09 bps single-level, −0.42 to −0.49 bps for 2–9 levels.
    10+ level sweeps are less adverse (−0.17 bps at 5 s, +0.02 bps at 60 s),
    consistent with partial overshoot. For a given size, more levels means
    worse markouts, and larger notional is worse at every level count
    (above $1m: −1.0 to −1.6 bps).
  - *Activity is the strongest conditioner:* sweeps in the high-activity
    tercile are much less adverse for makers (2 levels: −0.24 vs −0.65 bps;
    10+ levels: +0.08 vs −0.45 bps). Volatility shows the same pattern, more
    weakly. The realised-depth liquidity state goes the other way from
    expectation (thicker books, worse markouts) and is unexplained.
  - *Bursts:* inverted U. Moderate bursts continue (+0.13 to +0.16 bps at
    5 s); the extreme deciles reverse at 60 s (top decile −0.78 bps). Stronger
    continuation in low activity.
  - *Level depletion:* repeated same-side hits at an unchanged price predict
    a *break*, not a hold. Continuation rises monotonically from 0.16 bps
    (first hit) to 0.66 bps (11+ hits), and maker markout falls to −0.55 bps.
  - *Spread proxy:* a wide effective spread doubles subsequent absolute moves
    (4.3 vs 1.9 bps at 5 s) but *improves* maker markout (−0.07 vs −0.18 bps),
    since wider spreads compensate liquidity providers.
  - *Scale:* every maker markout lies between −1.6 and +0.1 bps, against a
    2 bps maker fee and a 0.01–0.02 bps BTC spread. As expected, the signals
    are statistically strong and protective, but BTC quoting at base-tier fees
    cannot be economic.
  - *Measurement caveat:* the mid proxy may be stale within a second. Filtering
    to fresh (updated within 1 s) observations barely changes the ICs, but
    validation against `bookTicker` mids is required before registration.

### E-003 · Passive interest from trades: absorption, resiliency, asymmetry (BTC 2022, started 2026-09-30)
- **Motivation (researcher):** unfilled passive orders are informative
  (E-001). Distinguish market-making liquidity from genuine directional passive
  interest, preferring evidence that is costly to fake (executions) over
  displayed size (which can be spoofed).
- **Decision:** E-003 runs before H-014 to H-016 are registered, so that
  H-015's pre-specified interaction is chosen on evidence.
- **Features** (trades only; definitions in `FEATURE_CATALOGUE.md` once
  confirmed):
  - **A1 absorption ratio:** cumulative quantity executed in the current run of
    same-side single-level events at an unchanged price, divided by the typical
    depth per level on that side. Typical depth is the trailing-30-minute
    geometric mean of quantity ÷ levels over same-side sweeps.
  - **A2 resiliency state:** trailing-30-minute mean, over sweeps whose
    outcome is already known, of the fraction of each sweep's mid-proxy impact
    that reverted within 5 s.
  - **A3 absorption asymmetry:** over the trailing 5 minutes, (sell volume
    absorbed at an unchanged bid − buy volume absorbed at an unchanged ask) ÷
    total volume. Positive means passive buyers are absorbing selling.
- **Targets:** as E-002.
- **Script:** `scripts/explore_absorption.py`; outputs in
  `reports/absorption_e003/`; section E-003 of
  `notebooks/SecondsSignalsExploration.ipynb`. Tercile cuts are from
  January 2022.
- **Observations (2022; development data, not results):**
  - *A1 absorption goes the opposite way from the "iceberg holds" idea.*
    Within each hit class, more volume executed at an unchanged price (relative
    to typical depth) means **more** continuation and worse maker markouts. At
    11+ hits: continuation +0.75 vs +0.12 bps, markout −0.63 vs −0.09 bps
    (high vs low absorption tercile, τ = 5 s). The effect persists to 60 s.
    Rank IC with continuation is +0.02 at 5 s (11 of 12 months). Executed
    volume at a level measures depletion of the visible queue, not the
    strength of a hidden passive order. Separating genuine passive interest
    needs order-book data (E-004).
  - *A2 resiliency is a strong sweep conditioner.* When recent sweeps have
    reverted more (high tercile), new sweeps continue less (0.64 vs 0.90 bps)
    and are far less toxic for makers (2 levels −0.22 vs −0.62 bps; 10+
    levels +0.16 vs −0.42 bps). Rank IC is negative in 11 of 12 months. It is
    comparable in size to the activity conditioner from E-002; the correlation
    between the two has not yet been measured.
  - *A3 absorption asymmetry mostly mirrors aggressive flow* (Spearman −0.74
    with 5-minute aggressive imbalance). Its positive IC (+0.02 to +0.03, 12
    of 12 months) is largely the known flow reversal, and the decile pattern
    is not monotone. Treated as having no independent evidence.
- **Implication for registration:** H-015 (level-depletion break) takes
  absorption (A1) as its pre-specified interaction, with a **positive** sign:
  more absorption, stronger break. For H-014 (sweep toxicity) the
  one-interaction choice is between activity (E-002) and resiliency (A2);
  the choice is the researcher's.

### V-001 · Mid-proxy validation against bookTicker (2023-05-16 to 2023-05-31)
- **Type:** measurement validation, inside the order-book exploration window.
- **Script:** `scripts/validate_mid_proxy.py`; outputs in
  `reports/mid_proxy_validation/`.
- **Data:** 13.7 million events; 160 million quote updates. Median quoted
  spread 0.037 bps; 82% of updates at one tick.
- **Findings:**
  - *Level:* median absolute error of the proxy mid against the true mid is
    0.00 bps, 90th percentile 0.21 bps (0.30 bps when one side is more than
    1 s stale).
  - *Responses:* per-event continuation measured with the proxy and with the
    true mid has rank correlation 0.945 at 1 s, rising to 0.988 at 60 s.
    Rank IC of signed levels: proxy 0.18 against true 0.16 at 5 s.
  - *Bias:* for 1–9 level events the proxy overstates continuation by about
    15–20% (for example 2 levels at 5 s: 0.83 against 0.69 bps). For 10+ level
    sweeps it **understates** continuation (0.96 against 1.32 bps). Maker
    markouts agree within about 0.03 bps for 1–9 levels; for 10+ levels the
    proxy makes them about 0.1 bps too negative.
  - *Resiliency:* proxy and true-mid versions correlate 0.96. But resiliency
    correlates **0.77–0.80 with activity** and 0.74–0.78 with volatility;
    activity and volatility correlate 0.72.
- **Conclusion:** the qualitative E-002 and E-003 findings stand. Proxy-based
  magnitudes are approximate, with modest overstatement of 1–9 level
  continuation. Registered tests should use true quote mids wherever
  `bookTicker` exists, and the proxy (with this bias noted) elsewhere.
  Resiliency is mostly an activity state.

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
