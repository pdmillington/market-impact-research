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

## Exploration log

Exploratory analysis generates hypotheses; it tests nothing. Each entry records
what was sliced, so that any hypothesis it inspires is registered with its
origin known and the slices counted.

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
