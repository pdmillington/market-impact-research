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
| BTCUSDT 2026-06 to 2026-08 | **Used (C-006 locked test, 2026-10-01)** | Spent on the C-006 primary only (BTC, ETH and SOL event bars). Not rejected under the frozen day-weighted rule; per-trade BTC 0.2, ETH 4.9, SOL 28.3 bps. |
| BTC, ETH, SOL and the alt basket, 2021-01 to 2023-12, at 8–24 h | **E-005 screening window** (2026-10-01) | Hourly execution-signature features vs forward 8 h/24 h returns. These returns were already used by H-009/H-010/C-010/E-004; the features are new. |
| BTC, ETH, SOL and the alt basket, 2024-01 to 2026-05, at 8–24 h | **H-013 test** (registered 2026-10-01) | Returns already seen by earlier hypotheses (not with these features). Genuine out-of-sample evidence comes only from the locked months and the prospective period. |
| ETHUSDT, SOLUSDT 5–15 minute returns, 2023-01 to 2026-05 | **C-006/C-007 test** (registered 2026-10-01; used by C-007 and C-006, 2026-10-01) | Not examined by any previous test at this horizon. Calibration and feature years are 2021–2022. |
| BTC, ETH, SOL 2026-09 onward | **Prospective: P-006** (registered 2026-10-01) | Reserved for the trade-weighted C-006 test. Data-quality checks only until each scheduled look (12 and 24 months). |
| BTCUSDT 2026-09 onward (other uses) | **Prospective** | Accumulates as a genuine forward sample. |
| BTCUSDT `bookTicker` 2023-05 to 2023-07 | **Order-book exploration** (decision 2026-09-30) | Used for E-004 (order-book features) and for mid-proxy validation. Trade-based hypotheses tested on 2023–26 include these months; that overlap is limited to order-book exploration and is noted. |
| BTCUSDT `bookTicker` 2023-08 to 2024-04 | **Order-book test** | Reserved for registered order-book hypotheses. |
| ETHUSDT, SOLUSDT (all periods) | **Used for confirmation** | Built 2026-09-30. Seconds-scale returns used by the H-014 to H-016 confirmations; hourly/daily returns used by C-010 and E-004; `bookTicker` 2023-05 to 2024-03 used by H-017/H-018. No longer untouched. |
| Alt USDT perps (all except BTC, ETH, SOL), 2020-01 to 2026-05 | **Used (H-019 test, 2026-09-30)** | Used once by the registered H-019 test and the post-hoc S2 diagnostic. No longer fresh. |
| Alt USDT perps, 2026-06 to 2026-08 | **Locked, single use** | Not read. H-020 failed Stage 1, so it remains available for a future candidate. |
| Bybit USDT perps, 2021-05 to 2026-05 | **Used (H-020 Stage 1, 2026-09-30)** | Used once by the registered Stage 1 test. |
| Bybit USDT perps, 2026-06 to 2026-08 | **Locked, single use** | Downloaded, not read. Available for a future candidate. |
| Binance and Bybit, 2026-10 onward | **Prospective** | H-020 Stage 3 (six months). |

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

**Revised screening objective (researcher decision, 2026-10-01; a deliberate
departure from the net-of-cost gates above, recorded as such):**

- **Gross return threshold by horizon τ** (mean simple return per unit
  notional of the trades actually taken, mid to mid):

  T(τ) = max(1.5, min(2 × √(τ / 5 min), 10)) bps.

  Examples:

  | τ | ≤ 3 min | 5 min | 10 min | 15 min | 30 min | 60 min | ≥ 2 h 5 min |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | T(τ), bps | 1.5 | 2.0 | 2.83 | 3.46 | 4.90 | 6.93 | 10 |

  Any idea whose gross return is above T(τ) at some horizon stays in the
  mix.
- **Rationale (researcher):** fees can be reduced for BTC (route to be
  specified), so the net-of-base-tier-fee gate is relaxed to a gross
  threshold for screening.
- **Up to the 2-hour cap**, the threshold is roughly a constant-IC line: at
  1 sd of signal, E[r] ≈ IC × σ(τ), and σ(τ) ∝ √τ. For BTC (σ(5 min) ≈ 15
  bps at about 50% annual volatility), 2 bps per 5 min ≈ IC 0.13, or about
  0.1 for the top-decile mean. For ETH/SOL the same bps corresponds to a
  lower IC.
- **Beyond 2 h the 10 bps cap is looser** than the constant-IC line (at
  24 h, 10 bps ≈ IC 0.04 for BTC). Long-horizon ideas are therefore also
  judged on t-statistic and Sharpe, not on mean return per trade alone.
- **Unchanged:**
  - register before testing;
  - nothing deleted;
  - specification counts;
  - stability tests (months, best/worst-day removal, tail concentration);
  - **strategies that are potentially additive to others are kept even when
    they fail alone** (researcher, 2026-10-01).
- **Costs:** net results at base-tier fees are still reported alongside
  gross, so the fee assumption stays visible.
- **Target-fee scenario (researcher estimate, 2026-10-01; not verified):**
  - maker 1 bp and taker 2 bps per side, achievable for BTC by a route the
    researcher has in mind;
  - Binance base-tier fees are to be ignored for now;
  - net results are reported at these fees, plus E1 drift (about 1 bp per
    side) for passive execution;
  - whether the same fees apply to ETH, SOL and alts is not established.

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
| H-014 to H-016 | `seconds_signals_test_v1` (complete, including true-mid and ETH/SOL confirmation) | 8 + 7 + 7 = 22 cells |
| H-017 | quoting simulation (registered) | 12-combination calibration grid + 2 strategies × 2 instruments + 3 sensitivities |
| C-010 | H-010 confirmation on ETH and SOL (failed; H-010 retired) | 2 symbols × 1 frozen primary |
| H-018 | execution simulation (registered) | 12-combination calibration grid per instrument + 4 strategies × 3 instruments + 4 sensitivities |
| H-019 | `crowding_panel_h019` (rejected) | 4 primary conditions + 2 tested secondaries + 2 descriptive (6 cells) + 3 cost levels + 1 post-hoc S2 decomposition |
| H-020 | `carry_h020` Stage 1 (rejected) | 4 Stage 1 conditions × 2 signals (C primary, funding level secondary) + descriptive S2–S5 |
| E-005 | MI screen (registered protocol) | 7 features × 4 targets × 2 horizons × own/cross/market pairs (BH family), plus a separate crowding side table |
| H-013 | execution-signature spillover (registered) | up to 3 selected cells × 4 conditions |
| E-006 | re-score of saved results (descriptive) | 1,696 saved cells re-scored; no new tests |
| C-006/C-007 | ETH/SOL confirmation (registered) | C-006: 1 primary × 2 instruments × 3 conditions; C-007: 2 co-primary rates (Holm) × 2 instruments × 3 conditions |
| P-006 | prospective C-006 test (registered) | 1 primary statistic, 2 looks (O'Brien–Fleming-type) |
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

### H-013 · Cross-asset execution-signature spillover at 8–24 hours
- **Status:** registered (2026-10-01); **not run, because E-005 selected no cell (2026-10-01)**. Registered together with exploration E-005, before any signature feature at 8–24 h was computed. Selection from E-005 follows the frozen rule below. Search horizon moved to 8–24 h by the researcher (2026-10-01).
- **Why this is a new argument (ledger rule 2)**

- **What failed before:**
  - At seconds, the signatures predict direction robustly (H-014 to H-016,
    38/38 months) but are too small to pay fees.
  - At 8–24 hours, crowding features failed (H-009/H-010/H-019/H-020).
- **The untested question:** whether *persistent* execution behaviour over
  hours carries information about the next day. Examples are urgency (sweeps
  through levels), passive absorption (large resting interest soaking up
  aggression) and burst one-sidedness.
- **Economic argument:**
  - An informed or constrained trader who must complete a large order within
    hours leaves a footprint. They are urgent when time is short (sweeps) and
    passive when patient (absorption).
  - Kyle-type models predict continuation while the order is incomplete.
  - Liquidity-provision models predict reversal once urgent demand is
    absorbed.
  - The sign is not known a priori, which is why a non-directional screen
    comes first.
- **Cross-asset angle (H-013's original idea):** hedgers and informed traders
  express views in the most liquid instrument, so a signature in BTC (or ETH)
  may lead SOL and alts.
- **Features (per instrument i; from reconstructed events; trailing window W = 8 h primary, 24 h secondary)**

All features use the frozen timestamp_direction_v1 events. Each is z-scored
against its own trailing 30 days of hourly values; nothing is fitted.

| Name | Definition (over the window W) | Signature |
|---|---|---|
| `ml_share` | multi-level aggressive notional ÷ total aggressive notional (unsigned) | urgency level |
| `ml_imbalance` | (buy − sell multi-level notional) ÷ multi-level notional | one-sided urgency |
| `sweep_pressure` | Σ s × sweep_bps over multi-level events ÷ event count | signed depth taken |
| `absorption_imbalance` | (sell-side − buy-side high-absorption level runs) ÷ all such runs; frozen E-003 tercile cuts | passive interest: sellers hit a bid that holds (a bullish passive buyer), and vice versa |
| `burst_skew` | (time in burst decile 10 − time in decile 1) ÷ W; frozen E-002 cuts | one-sided self-excitation |
| `large_imbalance` | signed notional of events ≥ $1 m ÷ total notional | large-trader footprint |
| `flow_imbalance` | plain signed notional ÷ total (benchmark; the H-012 family) | control |

- That is 6 signatures plus 1 control.
- The market composite M is the mean of each feature's z across BTC, ETH and
  SOL.
- **Targets**

- Forward simple returns (linear-perp accounting) over **8 h (primary)** and
  **24 h**, from the decision hour.
- Instruments: BTC, ETH and SOL perps, plus the equal-weighted Binance alt
  basket (H-019 membership, simple returns).
- Own-asset and cross-asset pairs: features of i → returns of j, and M →
  each j.
- **E-005 (the screen that feeds the selection):** registered in the exploration log as E-005.
- **H-013 · Cross-asset execution-signature spillover at 8–24 hours (registered after E-005, with this frozen rule)**

- **Selection rule (frozen now; applied mechanically to E-005's table):**
  - keep (feature, source → target, horizon) cells whose *conditional*
    directional MI passes BH at 0.10;
  - choose **up to 3 cells by greedy forward selection:**
    1. The first cell has the largest excess I(X; sign r | Z).
    2. Each later cell maximises excess I(X; sign r | Z, chosen features
       for that target).
    3. It is added only if that increment is significant against its own
       circular-shift null at p < 0.05.

    At most one cell per feature. This is the redundancy guard
    (mRMR-style), so the three are not the same flow signal three times.
  - shape: isotonic (monotone) fit of E[r | X], direction from the
    development curve;
  - if the development curve is not monotone (a U-shape), use the
    |X|-conditioned form that E-005 shows. That is the only non-linear
    allowance.
  - If no cell passes, H-013 is not run, and that is recorded as an
    outcome.
- **Signal:** for each selected cell, the position is sign(fitted
  E[r | X]) when the fitted |E[r | X]| exceeds the development median of
  |E[r | X]|, otherwise 0.
- **Holding and P&L:**
  - hold for the target horizon, in staggered tranches (8 or 24);
  - risk-scale positions as in H-019;
  - book P&L with `alpha_search/perp_pnl.py` (simple returns, funding on
    notional, costs from drifted notional).
- **Costs:** gate at 5 bps per side taker (BTC, ETH, SOL; the alt basket
  6 bps). Report 3 bps (H-018 E1 passive) and gross.
- **Samples:**
  - selection: 2021-01 to 2023-12;
  - **test 2024-01 to 2026-05**;
  - locked confirmation 2026-06 to 2026-08: BTC and the alt panel, both
    unread. ETH and SOL events exist for those months but their returns are
    unread too.
  - prospective six months from 2026-10 before any capital.
- **Pass conditions (test; each selected cell):**
  1. Net Sharpe at the gate > 0, with Newey–West t > 2 (lag = holding
     horizon).
  2. Positive with the best and worst 10 days removed, and in more than 55%
     of months.
  3. Incremental: alpha > 0 after regressing daily P&L on the H-019
     crowding composite's P&L, on the `flow_imbalance` control's P&L, and on
     the other selected cells' P&L.
  4. Last 24 months Sharpe > 0.
- **Secondary (reported):**
  - volatility information. Does I(X; |r|) improve a next-8h volatility
    forecast over 24 h realised volatility? This feeds back into execution
    scheduling, not alpha.
  - results by year;
  - capacity.
- **Data honesty (the hard constraint)**

- No untouched historical returns at the 8–24 h horizon remain for liquid
  crypto:
  - BTC 2023–2026, ETH/SOL and the alt panel were all used by
    H-009/H-010/C-010/E-004/H-019/H-020;
  - the Bybit 2022–2026 data were used by H-020.
- **The test period has seen these returns, though not these features.** It
  guards against fitting but not against the same market episodes recurring.
- **The genuine out-of-sample evidence is the locked three months plus the
  six prospective months.** At Sharpe ≈ 1, those nine months give t ≈ 0.9,
  so they can reject but cannot confirm. A real confirmation needs about two
  years of live data.
- The specification count (now about 9,000 cells plus the E-005 screen)
  should be used for a deflated Sharpe on any H-013 survivor.
- **Researcher decisions (2026-10-01)**

- (a) Primary horizon: **8 h**, with 24 h secondary.
- (b) Features: all 7, screened by **conditional MI given flow and own
  return** (the orthogonalisation step), with a greedy redundancy guard at
  selection.
- (c) Selection: **up to 3 cells**.
- (d) Crowding features: screened non-linearly in a **separate side table**
  that cannot feed H-013's selection. Confirmed.
- **Original proposal (2026-09-30, superseded; kept per rule 2):**
  - **Status:** proposed; to be registered after H-012.
  - **Economic hypothesis:** hedgers and informed traders use the most liquid
    correlated instrument, so flow in one coin predicts continuation in
    correlated coins (cross-impact; Benzaquen et al. 2017, Pasquariello & Vega
    2015).
  - **Design constraint:** develop with BTC returns as the only target, using a
    market-wide flow factor built from BTC, ETH and SOL flow. Confirm by
    predicting ETH and SOL returns. ETH and SOL flow will have been seen, but
    not their returns (partial contamination, stated in advance).
- **Outcome (2026-10-01): not run, by the registered rule.** No E-005 cell
  passed BH, so the selection is empty (`reports/e005_mi_screen/selection.json`).
  Execution signatures aggregated to 8–24 h carry no detectable
  directional information about BTC, ETH, SOL or the alt basket, either
  beyond plain flow and past returns or (for flow itself) marginally. The
  test window (2024 to 2026-05) and the locked months stay unused by
  H-013.

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
  - True-mid robustness (2023-08 to 2024-03, 9 months, real quote mids):
    **pass.** IC at 5 s is 0.141 against 0.175 with the proxy on the same
    months (about 20% smaller, as V-001 predicted). Positive at every
    horizon in 9 of 9 months. Activity interaction +0.18 bps.
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
  - True-mid robustness (9 months): **pass.** IC at 5 s is 0.088 against
    0.111 with the proxy. The absorption interaction is *stronger* with
    real quotes (+0.21 against +0.13 bps), 9 of 9 months.

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
- **Status:** registered (2026-09-30); **tested 2026-09-30: fails** (SOL; ETH control 2026-10-01: same).
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
- **Premise correction (2026-09-30, before any simulation):** the
  registration assumed SOL's tick was 0.01 (about 5 bps at $20). In the
  2023–24 quote year SOLUSDT's tick was **0.001**: 0.53 bps at about $19
  (June 2023), falling to about 0.05 bps at $200. Binance has since raised it
  to 0.01. Spread capture per round trip is therefore at most about 0.5 bps,
  far below the 4 bps maker round trip, so pass condition 2 is expected to
  fail on SOL. Condition 1 (signals add value) remains informative. The
  registration is unchanged; the error is recorded here. Wide-tick contracts
  (current ticks: ADA about 4 bps, DOGE about 1 bp, XRP about 0.7 bp; the
  2023 values must be measured from data) would need a separate
  registration.
- **Run notes (2026-09-30; implementation choices, recorded before any test
  day was simulated):**
  - Unit size is $20 notional (Binance USD-M minimum order notional). P&L is
    reported in bps of that unit per day and as net bps per fill.
  - The inventory starts flat each UTC day and is marked to the last quote
    mid at day end. Funding is charged at the 08:00 and 16:00 settlements on
    the inventory held (the 00:00 settlement falls at a flat start).
  - S1 is selected per instrument on that instrument's calibration days.
  - The first calibration selection was invalid: the bookTicker archive
    starts at 11:49 on 2023-05-16, so funding at 08:00 was NaN and the NaN
    ranked first. Funding now skips settlements with no position or no
    quote, and selection refuses NaN. Calibration was rerun on the same
    calibration days; no test day was involved.
  - The optimistic queue sensitivity is implemented as registered:
    displayed-quantity drops not explained by executions are cancellations
    spread pro rata across the queue. An interim cap-at-displayed version
    was replaced before any test run used it.
  - SOL calibration selection: **D = 2 bps, T = 5 s, k = 3**. In calibration,
    net P&L per fill was about −3.1 bps for S0 and all 12 S1 configurations;
    S1 differs from S0 mainly by quoting 40% rather than 84% of the time.
- **Outcome (2026-09-30, SOL; ETH control pending its quotes): H-017 fails.**
  Condition 1 passes as written; condition 2 fails.

  | SOL test, 244 days | S0 | S1 (D2/T5/k3) |
  |---|---:|---:|
  | Fills per day | 63,400 | 22,900 |
  | Time quoting | 73% | 32% |
  | Net P&L per fill | −2.98 bps | −2.98 bps |
  | Spread capture per fill | −0.24 bps | −0.25 bps |
  | 60 s markout per fill | −0.99 bps | −1.00 bps |
  | Net per day (bps of unit) | −188,900 | −68,200 |

  - **Condition 1 (S1 − S0 > 0):** +120,700 bps of unit per day, t 30,
    8 of 8 months. It also holds under latency 250 ms, the optimistic queue
    and inventory ±2. **The pass is degenerate, as V-002 anticipated before
    the test:** net per fill is identical (−2.98 vs −2.98 bps), and S1
    loses less only because it quotes less than half as often. The pulls do
    not select better fills; the 1 s markout is slightly worse under S1.
  - **Condition 2 (S1 profitable) fails:** t −32, 0 of 8 months positive, as
    the premise correction predicted (tick 0.06–0.5 bps against a 2 bps
    fee).
  - **Break-even tick (reported):** about 5.5 bps, holding the measured
    adverse selection fixed. Current ADA (about 4 bps) is below this, and
    adverse selection probably grows with the tick.
  - **Conclusion:** at base-tier fees, the confirmed seconds-scale signals
    do not make passive quoting pay on SOL or on any sub-bp-tick contract.
    A wide-tick contract would need its own registration, and the evidence
    here gives little reason to expect a pass.
  - **ETH control (2026-10-01; S1 frozen at D2/T5/k3, the same as SOL):**
    S0 −2.93 and S1 −3.01 bps net per fill. S1 quotes 28% of the time
    against 80% for S0. Condition 1 passes in the same degenerate way
    (t 24, 8/8 months), condition 2 fails (t −29), and the break-even
    tick is about 5.7 bps. Same verdict as SOL.
  - Outputs: `reports/quoting_h017/`, results
    `reports/quoting-h017-results.md`.

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
- **Status:** registered (2026-09-30); **tested 2026-09-30/10-01: condition 1 passes, E2 fails** (BTC, SOL, ETH).
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
- **Pre-test amendment (2026-09-30, researcher-approved, before any H-018
  calibration or test simulation).** Justified by trigger-coverage
  measurements on calibration days (V-002), not by any execution outcome:
  - (a) Run-away bursts use only the extreme deciles: 10 when buying and 1
    when selling (was 7–9 and 2–4). H-016 showed extreme bursts continue.
  - (b) Sweep thresholds are in **ticks**: a sweep of ≥ m ticks, m ∈ {2, 5, 10}
    (was D ∈ {2, 5, 10} bps, which almost never fires on BTC). This applies to
    both the run-away and the adverse rule.
  - (c) Level-run threshold k ∈ {6, 12} (was {3, 6}).
  - The calibration grid is therefore m ∈ {2, 5, 10} × T ∈ {2, 5} s ×
    k ∈ {6, 12}. Everything else is unchanged.
- **Run notes (2026-09-30; implementation choices, recorded before any test
  day was simulated except where stated):**
  - Parent orders: 12 per UTC day (one per 2-hour block), arrival time
    uniform within the block, side from a fair coin. The stream is keyed by
    (seed 20260930, day number), so orders are identical across
    strategies, instruments and runs.
  - Passive fills are all-or-nothing. The order fills when executions at our
    price beyond the queue ahead reach the full parent size, or when the
    price trades through our level (the whole level is consumed). Partial
    fills are ignored, which is conservative.
  - Sweep depth in ticks is sweep_bps × price / tick, using the day's
    inferred tick. The level-run trigger uses single-level runs of ≥ k hits
    by aggressors in our direction, with no absorption condition (none is
    registered for H-018).
  - IS is decomposed into spread (fill price versus the quote mid at fill),
    drift (mid at fill versus mid at arrival) and fee.
  - The optimistic queue model is as in the H-017 run notes. The BTC test
    was first run with an interim cap-at-displayed version, then rerun with
    the registered version before any result was recorded. The frozen E2
    selection was unaffected.
  - E3 (secondary) is not built. It is deferred unless E2 passes.
  - Calibration selections (lowest mean IS): **BTC m = 10, T = 5 s, k = 12;
    SOL m = 10, T = 5 s, k = 12**. Every E2 grid point had higher mean IS
    than E1 in calibration.
- **Outcome (2026-09-30, BTC and SOL; ETH pending its quotes, which cannot
  change either verdict): condition 1 passes; E2 fails and is significantly
  harmful.**

  | Test, 2,925 orders each | BTC E0 | BTC E1 | BTC E2 | SOL E0 | SOL E1 | SOL E2 |
  |---|---:|---:|---:|---:|---:|---:|
  | Mean IS (bps) | 5.01 | 3.09 | 3.68 | 5.25 | 3.25 | 3.56 |
  | IS 95th percentile | 5.02 | 7.07 | 7.19 | 5.79 | 8.90 | 9.49 |
  | Drift (bps) | 0.0 | 1.01 | 0.45 | 0.0 | 0.91 | 0.75 |
  | Fee (bps) | 5.0 | 2.0 | 3.18 | 5.0 | 2.0 | 2.49 |
  | Passive share | 0 | 100% | 61% | 0 | 100% | 84% |

  - **Condition 1 (E0 − E1 > 0) passes:** BTC +1.92 bps (t 52), SOL
    +2.00 bps (t 35), 8 of 8 months each. It holds under every
    sensitivity: +1.85 to +2.32 bps on BTC, +1.96 to +2.07 bps on SOL. The
    optimistic queue raises the BTC saving to 2.32 bps.
    Passive $10k orders fill within 30 minutes every time (median 7 s on
    BTC, 2.5 s on SOL), mostly by trade-through.
  - **Condition 2 (E1 − E2 > 0) fails and is significantly negative:**
    BTC −0.60 bps (t −21), SOL −0.32 bps (t −8), 0 of 8 months positive on
    either. It is negative under every sensitivity. The run-away triggers
    do avoid drift (BTC 1.01 → 0.45 bps), but each cross adds 3 bps of fee,
    and the avoided drift is far smaller than that.
  - **Condition 3 (tail) passes:** E2's 95th percentile is +0.12 (BTC) and
    +0.59 (SOL) bps worse than E1's.
  - E3 is not run (deferred because E2 failed).
  - **Economic translation:** switching from taker to patient passive
    execution saves about 2 bps per side, about 4 bps per round trip. Any
    signal that grosses more than about 6 bps per round trip (taker 10 minus
    4) with passive entry is worth re-testing under E1 execution. Adding the
    seconds-scale signals to the executor makes it worse.
  - **Caveat:** random-direction orders. Alpha-driven orders trade with
    momentum and would face more drift under E1. The 30-minute E1 drift of
    about 1 bp is the relevant figure to stress.
  - **ETH, the third instrument (2026-10-01; E2 frozen at m10/T5/k12):**
    - IS: E0 5.02, E1 3.13, E2 3.68 bps.
    - Condition 1: +1.90 bps (t 49, 8/8 months).
    - Condition 2: −0.55 bps (t −17, 0/8 months).
    - Condition 3: +0.12 bps.
    - Sensitivities: E0 − E1 +1.83 to +2.16 bps, E1 − E2 −0.55 to −0.61
      bps.
    - **Final verdict across all three instruments:** condition 1 passes
      3/3; condition 2 fails 3/3 and is significantly negative on each.
  - Outputs: `reports/execution_h018/`, results
    `reports/execution-h018-results.md`.

### H-019 · Crowding contrarian on a pooled alt-perp panel
- **Status:** registered (2026-09-30); **tested 2026-09-30: rejected** (conditions 1–3 fail). Registered 2026-09-30, before any alt-perp return was
  examined. Revives the economic idea of H-009 and H-010 (retired) under rule
  2, with a new argument (E-004).
- **Origin:** E-004. H-010's long/short-ratio features have no consistent
  sign across instruments. The crowding features (funding, premium and past
  return) were contrarian on BTC, ETH and SOL through 2023 and have faded
  since. BTC alone gives t ≈ 1.3, so a pooled panel is needed for power.
- **Economic hypothesis:** leveraged perp demand that runs ahead of spot
  (high funding, a rich premium, a strong run-up) is crowded and mean
  reverts over about a day, because arbitrage capital is limited.
- **Arbitrage-capacity prediction:** if institutional basis capital (spot
  ETFs from January 2024, CME) compressed the premium in the majors, the
  effect persists longer in smaller, less-arbitraged perps.
- **Universe (point-in-time, no survivorship):**
  - *Candidates:* Binance USD-M USDT perpetuals, including those since
    delisted, each only while it trades (hourly klines present).
  - *Excluded:* BTC, ETH and SOL; stablecoin bases and index contracts
    (USDC, BUSD, TUSD, FDUSD, USDP, DAI, DEFI, BTCDOM, FOOTBALL,
    BLUEBIRD and similar).
  - *Monthly membership:* on the first day of each month, the **30** most
    liquid eligible perps by trailing 30-day quote volume (daily klines).
    A perp is eligible once it has at least 12 months of kline history.
- **Data per member:** hourly perp klines (close, quote volume), hourly
  premium-index klines, and funding settlements at any funding interval.
- **Signal (fixed signs, nothing fitted).** At each hour t, for each member,
  with every input known at t:
  - z_F: the last settled funding rate against its trailing **30-day** mean
    and standard deviation (by time, not settlement count);
  - z_P: the mean of the last 8 hourly premium-index closes, z-scored
    against its own trailing 30 days of hourly values;
  - z_R: the 24-hour log return, z-scored against its own trailing 30 days
    of hourly values;
  - **C = −(z_F + z_P + z_R) / 3**, each input winsorised at ±4. A missing
    input is dropped from the mean; C is undefined if all three are missing.
- **Positions:**
  - *Tail rule:* sign(C) when |C| is at or above the member's
    trailing-365-day 80th percentile of |C|, otherwise 0.
  - *Holding:* staggered 24-hour tranches (the mean of the last 24 hourly
    signals), as in H-010.
  - *Risk scaling:* × 20% / trailing 30-day annualised volatility of hourly
    returns, capped at 3×.
  - *Portfolio:* equal-weighted mean across current members. Membership
    changes flatten at the month boundary.
- **P&L:** hourly, position × next-hour return − position × funding
  settled that hour − cost × |position change|.
- **Costs:** the **gate is 6 bps per side** (5 bps taker plus 1 bp alt
  half-spread allowance). 3 bps per side (H-018 E1) and gross are reported.
- **Samples:** warm-up 2021-01 to 2021-12; **test 2022-01 to 2026-05**;
  locked confirmation 2026-06 to 2026-08 (alt panel plus the BTC holdout),
  after the test verdict. The confirmation can only reject.
- **Pass conditions (all on the test sample; all four required):**
  1. *Existence:* portfolio net Sharpe at 6 bps > 0, with Newey–West
     (24-hour lag) t on daily P&L > 2.
  2. *Not beta:* alpha from regressing daily portfolio P&L on the
     equal-weighted member return > 0, t > 2.
  3. *Breadth:* net P&L positive for more than 55% of instruments that were
     members for 12 months or more.
  4. *Current:* portfolio net Sharpe at 6 bps > 0 over 2024-06 to 2026-05
     (sign only).
- **Secondary** (reported; S1 and S2 judged by t > 2):
  - S1, arbitrage capacity: each month, split members at the median trailing
    volume. Mean gross daily P&L of the low-volume half minus the
    high-volume half > 0, over the full test and over the last 24 months.
  - S2, cross-sectional neutral: each hour, rank C across members; long the
    top quintile, short the bottom, equal risk, dollar-neutral, 24-hour
    tranches, same costs. Condition 1 applied.
  - S3: drop one of z_F, z_P, z_R at a time (descriptive).
  - S4: year-by-year portfolio Sharpe (descriptive).
- **Frozen by the researcher (2026-09-30):** gate cost 6 bps per side;
  universe top 30. The long/short-ratio features are excluded; they need a
  separate entry (H-020) if pursued.
- **Limitations:**
  - Hourly klines have no spread, so costs are allowances.
  - Ranks 25–30 can be thin.
  - Crowding is correlated across alts, so the panel is less diverse than 30
    names suggest; condition 2 addresses this.
  - Delisted contracts are held until their last kline.
- **Run notes (2026-09-30; implementation choices, before any alt return
  was read):**
  - Data: `scripts/download_alt_panel.py` downloads daily klines for all 852
    candidates, then hourly perp klines, hourly premium and funding for every
    symbol that is ever a member. Membership is at
    `shared-data/features/alpha/alt_panel_v1/membership.parquet`.
    - Eligibility: first daily kline at least 365 days before the month, and
      trading within the last 2 days before it.
    - Ranking: summed quote volume over the 30 days before the month start.
  - Runner: `scripts/run_crowding_panel_h019.py`. It truncates every input
    before 2026-06-01, so the locked months are never read.
  - Decision hours are hourly kline close times. Hour h's P&L is realised
    over (T_h, T_h + 1 h]. Funding settlements are assigned with a 1 s
    tolerance, because they are stamped a few ms after the hour.
  - Minimum samples:
    - rolling z: 360 of 720 hourly values;
    - funding z: 20 settlements within 30 days;
    - |C| threshold: trailing 365 days excluding the current hour, at least
      180 days;
    - volatility: 360 of 720 hours.
  - Hours with no kline close have zero position; nothing is carried
    through gaps.
  - The tranche average uses signals generated while a member, and the
    position is zero outside membership.
  - Portfolio P&L is the sum over members divided by the number tradable
    that hour.
  - Condition 2's market return is the equal-weighted next-hour return of
    tradable members, summed daily.
  - Newey–West uses lag 1 on daily P&L (the 24-hour holding overlap).
  - S1 halves are membership ranks 1–15 (high volume) and 16–30 (low
    volume).
  - S2 requires at least 10 members with a defined C that hour. The top and
    bottom quintiles are each normalised to one unit of risk-scaled gross
    exposure.
  - A synthetic look-ahead check passed: changing all data after hour k
    leaves every C and position at or before k unchanged.
- **Outcome (2026-09-30): rejected.** Conditions 1, 2 and 3 fail;
  condition 4 passes.

  | Test 2022-01 to 2026-05 (1,612 days, 182 symbols, 29.9 members on average) | Sharpe | Return/yr | NW t |
  |---|---:|---:|---:|
  | Net at 6 bps (gate) | −0.10 | −0.45% | −0.22 |
  | Net at 3 bps | +0.05 | +0.23% | +0.11 |
  | Gross | +0.20 | +0.91% | +0.44 |

  - Condition 1 fails: Sharpe −0.10, t −0.22.
  - Condition 2 fails: alpha +0.69%/yr, t 0.35.
  - Condition 3 fails: 43.6% of 39 instruments positive (55% needed).
  - Condition 4 passes: last 24 months Sharpe +0.10.
  - The low exposure (risk-scaled tails averaged over 30 names) makes
    returns small; Sharpe is the relevant statistic.
  - **By year, Sharpe at the gate:** 2022 −0.23, 2023 −0.30, 2024 −0.63,
    2025 +1.03, 2026 YTD −0.03. Gross: −0.04, +0.13, −0.38, +1.40, +0.43.
    There is no decay pattern; this is noise around zero, with one good
    year.
  - **S1 (low − high volume, gross):** +0.66%/yr, t 0.59 (last 24 months
    t 0.78). Not supported.
  - **S2 (cross-sectional neutral, 6 bps):** Sharpe −0.94, −31%/yr, t −1.95;
    last 24 months t −2.61. Fails.
  - **S3:** dropping any one component leaves Sharpe between −0.11 and
    −0.22, so no single component carries it.
  - **Conclusion:** time-series crowding contrarian at a 24-hour horizon has
    no edge on the alt panel. With E-004, the BTC/ETH evidence for H-009 and
    H-010 looks like a 2021–2023 effect, or BTC luck, rather than a durable
    premium. The H-009/H-010 line is closed on this evidence.
  - **Post-hoc diagnostic (not a result; found after the verdict):** S2's
    loss is a cost artefact.
    - Before costs, S2 earned +29.2%/yr (Sharpe 0.87, NW t 1.8). Price
      contributed +14.8%/yr (t 0.9) and funding carry +14.4%/yr (Sharpe 8.8).
    - Hourly re-ranking gives turnover of 2.77× gross per day, which is
      about 61%/yr at 6 bps.
    - Lead L-H019a: cross-sectional funding carry with low turnover (daily
      rebalancing and hysteresis). It needs its own registration and data
      not used here: another venue's archives, or prospective data. The
      alt panel's 2022–2026 is now used, and its locked three months are
      too short to test a Sharpe near 1.
  - Outputs: `reports/crowding_panel_h019/` (`results.json`,
    `daily_pnl.parquet`, `breadth_full_gate_6.csv`), report
    `reports/crowding-panel-h019-results.md`.

- **P&L correction (2026-09-30, raised by the researcher after the
  verdict).** v1 computed P&L as position × *log* return. A linear
  USDT-margined perp earns quantity × price change, i.e. notional × simple
  return. Log returns understate a long's gain and overstate a short's by
  about σ²/2 per hour, which is roughly 50%/yr per leg for the S2 books
  (held-name volatility 110–125%/yr).
  - v2 uses `alpha_search/perp_pnl.py`:
    - quantity held between rebalances, so notional drifts;
    - funding charged on the notional at settlement;
    - costs charged on the trade from the drifted notional.
  - v1 outputs are kept as `v1_log_returns_*`. The S2 decomposition is now
    saved (`results.json` → `S2_decomposition`,
    `s2_decomposition_daily.parquet`).
  - **v2 results (the verdict is unchanged: rejected).** Net at 6 bps
    Sharpe −0.04 (t −0.08); gross Sharpe 0.29; alpha +0.07%/yr (t 0.04);
    breadth 46% of 39; last 24 months Sharpe +0.14. S1 t 0.31. S2 net
    −28.5%/yr (t −1.80).
  - **v2 S2 decomposition** (test, %/yr):

    | Leg | v1 (log) | v2 (perp) | v2 NW t |
    |---|---:|---:|---:|
    | Price, long | −57.6 | −4.1 | −0.13 |
    | Price, short | +72.3 | +24.0 | 0.74 |
    | Funding, long | | +9.8 | 9.07 |
    | Funding, short | | +4.7 | 11.08 |
    | Gross | +29.2 | +34.4 | 2.17 |
    | Costs at 6 bps (turnover 2.87/day) | | −62.9 | |

    Lead L-H019a stands: large, reliable funding income, with a noisy
    price leg. The v1 leg split was badly wrong even though the v1 net was
    close.
  - **Also affected:** H-010 and C-010. `run_funding_positioning.py`
    defines the hourly return F1h as a log return, so their price P&L
    carries a bias of about −σ²/2 × the book's mean signed position. That
    is small for a near-market-neutral book but not measured. They are
    retired; they would need re-running before any revival.

### H-020 · Low-turnover cross-sectional crowding carry
- **Status:** registered (2026-09-30); **Stage 1 run 2026-09-30: rejected** (both signals fail on Bybit). Registered before any low-turnover variant was computed on any data and before any Bybit return was examined. Registered by the researcher as drafted.
- **Origin:** lead L-H019a, a post-hoc diagnostic run after H-019's verdict.
  H-019's cross-sectional variant (S2) earned +29%/yr gross on the Binance alt
  panel in 2022–2026:
  - Sharpe 0.87, NW t 1.8;
  - funding carry +14%/yr (Sharpe 8.8);
  - price +15%/yr (t 0.9).

  Hourly re-ranking (turnover 2.8×/day) cost about 61%/yr at 6 bps. The
  lead was found by looking, so its evidence counts for little. This entry
  changes only the turnover mechanics and tests on another venue first.
- **Economic hypothesis:** leveraged longs crowd into some alts and pay
  persistently high funding, while others are neglected or shorted. A
  dollar-neutral book short the crowded names and long the uncrowded ones
  collects the funding spread. Crowded names are not systematically
  rewarded on price, so the price leg does not give the carry back. The
  spread persists because the arbitrage (short perp, long spot) is costly
  or impossible for many alts (spot borrow, custody, venue limits).
- **What would falsify it:** the price leg systematically offsets the carry
  (crowded longs are right, or squeezes hit the short book). Or the carry
  cannot be harvested at realistic turnover.

- **Signal (frozen from H-019; nothing re-fitted)**

- C = −(z_F + z_P + z_R) / 3 exactly as in H-019. That covers:
  - the funding z (30 days, by time);
  - the premium mean over 8 hours and the 24-hour return, each z-scored
    against its own trailing 30 days of hourly values;
  - winsorisation at ±4, with missing inputs dropped.
- Each input uses the venue's own data.

- **Book (new: low turnover)**

- **Rebalance once a day at 00:00 UTC** on C at that hour.
- **Hysteresis:**
  - *enter* long when a member is in the top quintile of C (top 6 of 30);
    *enter* short in the bottom quintile;
  - *hold* while it stays in the top (bottom) 40%, i.e. 12 of 30;
  - *exit* when it leaves that band, or leaves membership at a month
    boundary.
- **Weights:** risk-scaled, 20% / trailing 30-day annualised volatility,
  capped at 3×. Each side is normalised to one unit of gross exposure every
  day, so the book is dollar-neutral.
- **P&L:** hourly, Σ w_i × next-hour return − Σ w_i × funding settled that
  hour (at the venue's actual settlement times and intervals) − cost ×
  Σ|Δw_i|.

- **Universe (per venue, point-in-time, no survivorship)**

- The top 30 USDT perpetuals by trailing 30-day traded value, set on the
  first day of each month.
- Members must have at least 12 months of history and include delisted
  contracts while they trade.
- **Excluded:**
  - BTC, ETH and SOL;
  - stablecoin bases and index contracts, as in H-019;
  - **non-crypto underlyings** (tokenised equities, commodities such as
    PAXG and XAUT, FX and stock indices). This is new; H-019 admitted
    PAXG.

- **Costs**

- **Gate:** venue base-tier taker plus 1 bp spread allowance per side.
  That is 6.5 bps on Bybit (5.5 taker) and 6 bps on Binance (5 taker).
- **Reported:** 3 bps (passive, per H-018) and gross.

- **Stages**

**Stage 1: Bybit test, 2022-06 to 2026-05.** Bybit's API history starts
around 2021-05, and 12 months of eligibility is needed first.

All conditions are required:

1. **Net Sharpe** at the gate cost > 0, with Newey–West (lag 1) t on daily
   P&L > 2.
2. **Not concentrated:** net P&L positive after removing the best and worst
   10 days, and positive in more than 55% of months.
3. **Current:** net Sharpe > 0 over 2024-06 to 2026-05.
4. **Binance consistency** (same frozen rule, Binance alt panel 2022-01 to
   2026-05): net Sharpe > 0.

   This is a check that the turnover fix works where the lead was found.
   Those data are used, so it is not evidence; it is a gate only in the
   sense that failing it ends the hypothesis.

**Stage 2: locked months, 2026-06 to 2026-08** (Binance alt panel, which is
already locked, plus Bybit, which is locked from registration). Run once,
only after Stage 1 passes. It can only reject: reject if pooled net NW t < −2.

**Stage 3: prospective, from 2026-10.** Evaluate six months of archive
data from both venues, before any capital is committed. Continue if:

- net Sharpe > 0; and
- the result is not below the 5th percentile of Stage 1's rolling
  six-month Sharpe distribution.

- **Secondary (reported, not gates)**

- **S1, pure funding-level carry:** rank on the trailing 3-day mean
  funding rate (the level, not a z-score). Same book, costs and stages. The
  textbook carry trade.
- **S2:** P&L decomposed into price and funding legs, long and short books
  separately.
- **S3:** turnover per day and holding-period distribution.
- **S4:** year-by-year Sharpe; results at 3 bps and gross.
- **S5:** capacity. The median member's traded value on the day, against a
  $100k and $1m book (participation rate).

- **Cells**

- Stage 1: 4 conditions.
- One secondary signal (S1) through the same conditions.
- Descriptive S2–S5.
- The hysteresis bands (20/40) and the daily 00:00 rebalance are frozen
  here, with no grid.

- **Data build (before registration is run)**

- **Bybit:** fetch through the v5 API:
  - funding history;
  - hourly klines with traded value;
  - hourly premium-index klines;
  - for every USDT linear perp listed under `public.bybit.com/trading/`
    (delisted included).

  The download is 2021-05 to 2026-08. Stage 1 truncates at 2026-06-01.
- **Binance:** the existing H-019 panel.

- **Limitations**

- The venues are arbitrage-linked, so Bybit 2022–2026 shares Binance's
  market episodes. Stage 1 shows the effect is not venue-specific, not
  that it persists; Stages 2 and 3 carry that burden.
- The short book in thin alts may be hard to borrow or size, and squeeze
  risk is in the tails; S5 and condition 2 address this partly.
- Hourly klines have no spread.
- Funding intervals differ across contracts and change over time.
- **Frozen by the researcher (2026-09-30):**
  - primary signal: the H-019 composite C, with the funding level as secondary S1;
  - book: hysteresis 20% entry / 40% exit, daily rebalance at 00:00 UTC;
  - the Binance consistency check stays a gate (condition 4);
  - Stage 3: six prospective months before any capital.
- **Run notes (2026-09-30; implementation choices, before any H-020 result):**
  - **Bybit data:**
    - fetched by `shared-methodology/src/crypto_market_data/bybit.py`
      (v5 API; raw pages kept) and `scripts/download_bybit_panel.py`;
    - 1,093 archive symbols;
    - kept 819 (304 of them delisted);
    - excluded 267 non-crypto contracts (Bybit `symbolType` stock / ETF /
      commodity / forex, plus the bases PAXG, XAUT, XAU and XAG, which Bybit
      types as crypto), 4 stablecoin/index contracts and BTC/ETH/SOL.
  - **Bybit panel is thin early.** Bybit's USDT-perp range was small before
    late 2021, and the rule needs 12 months of history. Members: 8 in
    2022-06, 14 in 2022-07/08, 19 in 2022-09, 23 in 2022-10, 30 from
    2022-11. Overlap with Binance's top 30: 6 of 30 in 2022-06, 24 in
    2024-06, 26 in 2026-05.
  - **Binance consistency panel:** H-019's membership rebuilt with the
    H-020 exclusions (PAXG/XAUT/XAU/XAG; Binance also types PAXG as
    `COIN`) and saved as `alt_panel_v1/membership_h020.parquet`. RED and
    VELVET entered as replacements and were downloaded.
  - **Book:**
    - flat on any day with fewer than 10 members with a defined score (the
      H-019 S2 precedent), which covers Bybit 2022-06 on days that fall
      short;
    - quintile and band sizes are round(n/5) and round(0.4n), at least 1;
    - a held name that can no longer be ranked (no score, no close, or
      left membership) is exited;
    - the long book can briefly exceed a quintile, because held names stay
      alongside new entries;
    - weights are fixed within the day, and an hour with no close has zero
      weight.
  - **S1 funding level:** funding accrued over the trailing 72 hours (the
    sum of settled rates), which does not depend on the settlement
    interval. Long the lowest, short the highest.
  - **Condition 2 (months):** share of calendar months with positive net
    P&L.
  - **Capacity:** daily traded notional divided by the member's daily
    traded value, at $100k and $1m books.
  - **Runner:** `scripts/run_carry_h020.py`, on H-019's grid (every input
    truncated before 2026-06-01). Synthetic tests of the hysteresis book
    pass.
- **Outcome, Stage 1 (2026-09-30; linear-perp P&L): rejected.** Neither
  signal passes. Stages 2 and 3 are not run, and the locked months stay
  unread.

  | Test, %/yr (Sharpe) | Bybit C | Bybit funding level | Binance C | Binance funding level |
  |---|---:|---:|---:|---:|
  | Gross | +20.5 (0.55) | +18.8 (0.49) | +35.3 (0.90) | +41.6 (1.12) |
  | Net at 3 bps | −1.7 (−0.05) | +10.3 (0.27) | +12.9 (0.33) | +34.2 (0.92) |
  | Net at gate (6.5 / 6 bps) | **−27.6 (−0.74)** | **+0.5 (0.01)** | −9.6 (−0.24) | +26.7 (0.72) |
  | Turnover per day | 2.03 | 0.77 | 2.05 | 0.68 |
  | Median holding (days) | 2 | 4 | 2 | 5 |
  | Price legs, long / short | +23.7 / −13.1 | +2.4 / −9.2 | −14.3 / +38.4 | −16.0 / +28.6 |
  | Funding legs, long / short | +3.6 / +6.3 | +13.7 / +11.8 | +7.3 / +3.9 | +18.9 / +10.0 |

  - **C (primary):**
    - condition 1 fails (t −1.53);
    - condition 2 fails (months positive 38%, negative with the best and
      worst 10 days removed);
    - condition 3 fails (last 24 months Sharpe −1.29);
    - condition 4 fails (Binance −0.24).

    Even with daily hysteresis, C churns: turnover 2.0/day, median holding
    2 days. Its 24-hour return and premium components change daily.
  - **Funding level (S1):**
    - condition 1 fails (t 0.03);
    - condition 2 fails (months positive 46%);
    - condition 3 fails (−0.02);
    - condition 4 passes (Binance Sharpe 0.72, t 1.48, but those data are
      used and this is not evidence).
  - **What the data say:**
    - The funding legs are large and extremely reliable on both venues:
      +25%/yr (Bybit) and +29%/yr (Binance) on 2 units of gross, with
      NW t 9–22.
    - The price legs are noisy and venue- and period-dependent, with
      t < 1.1 throughout. The same idea shows short-leg price +28.6%/yr on
      Binance (test from 2022-01, including the H1 2022 crash) and −9.2%/yr
      on Bybit (from 2022-06).
    - A dollar-neutral long/short alt book carries about 40%/yr of price
      volatility. That swamps the carry, giving a gross Sharpe of about
      0.5 on the test venue.
  - **Capacity** (not binding): 95th-percentile daily participation is
    below 2% of a name's traded value at a $1m book.
  - **Lead (not a result):** the income is in the funding, and the risk is
    in unhedged alt price moves. Capturing funding without the price risk
    means pairing each perp with the *same coin's* spot (cash-and-carry /
    basis) rather than with other alts. That is a different, largely
    arbitrage-style trade, with spot-borrow, custody and capacity
    constraints. It would need its own economic case and registration.
    **Declined by the researcher (2026-10-01):** the capital needed (fully
    funded spot plus perp margin, and spot shorts for the negative-funding
    leg) is too high for the return. Not pursued.
  - Outputs: `reports/carry_h020/` (`results.json`,
    `daily_{venue}_{signal}.parquet` with every leg and net series), and
    report `reports/carry-h020-results.md`.

### C-006 and C-007 · Confirmation of the 5-minute pretrend reversal and stress reversion on ETHUSDT and SOLUSDT
- **Status:** registered (2026-10-01); **C-007 not confirmed (ETH only); C-006 CONFIRMED (ETH and SOL), 2026-10-01**. Registered before any ETH or SOL 5–15 minute return was examined. Origin: E-006. Researcher decisions: rule-based primaries accepted; C-007 adds the 200/yr rate as co-primary; C-007 runs first.
- **Fees and objective:** the revised objective and target fees (frozen assumptions, 2026-10-01).

#### Why the specifications are chosen this way

- E-006's BTC survivors are the best of 576 (H-006) and 72 (stress) cells.
  Picking the best BTC cell again would carry the selection bias forward.
- So each family's primary is fixed by a **stated rule that does not use
  the BTC ranking**:
  - **C-006:** the simplest model (`return_pretrend`: bar return plus 15
    and 60 minute pretrend, the frozen rule-4 benchmark); the middle bar
    size (1,000 events, the headline size in the H-006 report); top 1%
    tail; 5 min. E-006 shows the edge is front-loaded within 5–10 min.
  - **C-007:** H-007 (v1) exactly as originally registered: primary rate
    100 episodes a year, trigger entry, with the 5-minute hold, the
    shortest registered holding horizon.
- **BTC reference values** for the frozen primaries (for comparison, not
  gates):

  | | Gross | Ordinary days | t | Trades/day |
  |---|---:|---:|---:|---:|
  | C-006 | 15.1 bps | 7.4 | 14.4 | 28 |
  | C-007 | 9.8 bps | 4.9 | 2.9 | ~0.3 |

#### C-006 · 5-minute pretrend reversal (top 1% tail)

- **Specification (frozen from H-006, unchanged):**
  - event bars of 1,000 timestamp_direction_v1 events per instrument;
  - features R (bar return ÷ trailing 24 h volatility), pre15 and pre60
    (clock returns ending at the bar start ÷ trailing volatility);
  - forecast by OLS refitted monthly on the preceding 12 months and frozen
    for the test month;
  - trade the top 1% of |forecast| (training cut) in the direction of the
    forecast. The reversal comes from the fitted negative coefficients on
    R and the pretrends, not from a hard-coded sign;
  - delayed entry at the first 200-event endpoint at least 1 s after the
    decision;
  - exit at 5 min (clock);
  - returns mid to mid from the 1-second grid close.
- **Bar size:** the same 1,000-event count. The features are
  volatility-scaled, so they are scale-free in price; bar *duration*
  differs by instrument.
  - Robustness (reported): a bar size giving the same median duration as
    BTC's 1,000-event bars, calibrated on 2022 only.
- **Samples:**
  - features from 2022-01; first test month 2023-01;
  - **test 2023-01 to 2026-05** (each instrument);
  - locked 2026-06 to 2026-08: built only after a pass, read once,
    reject-only.
- **Pass conditions (each of ETH and SOL; both required):**
  1. Mean gross edge ≥ T(5 min) = 2 bps, net of 4 bps round trip > 0, and
     day-block t > 2.
  2. Ordinary days (excluding the best 10): gross ≥ 2 bps, and positive in
     more than 60% of months.
  3. Positive mean gross in at least 3 of the 4 calendar years (2026
     partial).
- **Reported (not gates):**
  - the 10-minute horizon;
  - the other bar sizes (200 to 8,000);
  - the 5% tail and the other models;
  - best-10-day share;
  - a 2025–26 versus 2023–24 decay comparison;
  - **a book simulation**: netted, overlapping positions (each signal
    1/24 unit, held 5 min), linear-perp P&L (`perp_pnl.py`) at 1 bp maker
    / 2 bps taker, annual return and Sharpe.

#### C-007 · Stress reversion (H-007 v1 primary)

- **Specification (frozen from H-007, unchanged):** `configs/stress_episodes_v1.json`:
  - trigger windows 30/60/300 s;
  - candidate floor z 4;
  - episode merge gap 30 min;
  - threshold calibrated on each instrument's **own** 2021–2022 to give
    100 episodes a year, then frozen;
  - trigger entry; reversion over 5 min; open-interest features not used.
- **Samples:** calibration 2021-01 to 2022-12; **test 2023-01 to
  2026-05**; locked months as C-006.
- **Pass conditions (each of ETH and SOL; both required):**
  1. Mean gross reversion ≥ T(5 min) = 2 bps, net of 4 bps > 0, and
     day-block t > 2.
  2. Ordinary-day gross ≥ 2 bps.
  3. Positive in at least 3 of the 4 calendar years.
- **Reported:** 15-minute hold, rates 50/200, the v2 trigger family, best-10
  share, and episodes by year.
- **Power:** about 100 episodes a year × 3.4 years ≈ 340 trades per
  instrument. At BTC's effect size (t 2.9 on 352 trades), roughly a coin
  flip for t > 2 even if the effect is real. Reported as a known
  limitation.

#### Interpretation rules (frozen)

- **Both instruments pass:** the effect is confirmed outside BTC; then the
  locked months, prospective capture and a fee-route check.
- **One passes:** not confirmed. Reported, and kept as "potentially
  additive" per the researcher's rule, with no capital.
- **Neither passes:** retired. The BTC result is attributed to selection
  and episodes.
- **The two families are judged separately**, with no pooling across them.

#### Data and build

- **C-006** needs the event-bar cache (`impact_research_v2`) for ETH and
  SOL at 1,000 and 200 events, built with `scripts/build_impact_research_bars.py`.
  That is not yet run for these symbols, and it is a large build (ETH/SOL
  2022–2026 events).

  Only the `return_pretrend` model is needed, so no impact fit (P) is
  required. A slim runner reuses `run_residual_decomposition.py`'s feature
  and tail code, without the P-dependent models.
- **C-007** needs the 1-second grids (built to 2026-05) and runs
  `run_stress_episodes.py` with symbol configs. OI is not needed.
- **Measurement:** returns from trade-based grid closes.
  - Bid–ask bounce is small relative to the edge: about 0.05 bps for ETH,
    and up to about 0.5 bps for SOL (tick 0.001 at $20 in 2023, about 0.05
    bps later).
  - SOL bounce can be checked on the bookTicker months (2023-05 to 2024-03)
    as a robustness item.

#### C-007 co-primary (researcher, 2026-10-01, at registration)
- **Co-primaries:** rates 100 and 200 episodes a year, both trigger entry
  with a 5-minute hold.
- **Multiple testing:** per instrument, a Holm correction across the two
  rates on one-sided p-values from the day-block t, at α = 0.05. The
  smaller p must be ≤ 0.025 and the larger ≤ 0.05. This replaces "t > 2"
  in condition 1.
- **Passing:** a rate passes on an instrument if condition 1 (Holm),
  condition 2 and condition 3 all hold.
  - C-007 passes on an instrument if at least one rate passes.
  - C-007 is confirmed only if it passes on both ETH and SOL.
- **BTC reference at rate 200:** gross 9.2 bps, ordinary days 4.8, t 3.2,
  about 0.5 trades a day.

- **C-007 run notes (2026-10-01, before any ETH/SOL result was read):**
  - Configs `configs/stress_c007_{ETHUSDT,SOLUSDT}.json` are identical to
    `stress_episodes_v1.json` except for symbol, experiment id and notes.
    Run with `scripts/run_stress_episodes.py`; outputs in
    `reports/stress_c007_*`.
  - Missing open interest before 2021-12 only blanks the descriptive OI
    fields, which no gate uses.
  - **Measurement:** reversion is evaluated as a **simple** return,
    −d × (P₁/P₀ − 1), per the researcher's linear-perp correction. The
    pipeline's log version −d × ln(P₁/P₀) is reported alongside. Over 5
    minutes the difference is about r²/2 (≈ 0.1 bps for a 50 bps move).
  - Day-block t: per-day mean reversion, then t across days.
  - Test trades are those with trigger in 2023-01-01 to 2026-05-31.
- **Outcome C-007 (2026-10-01): not confirmed.** ETH passes, SOL fails.
  Under the frozen interpretation rule it is kept as **potentially
  additive (BTC and ETH), with no capital**.

  | Trigger entry, 5 min | Rate | Trades | Gross (simple) | Ordinary days | t | p (one-sided) | Years + | Best-10 share |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | ETH | 100 | 533 | 12.2 bps | 6.3 | 2.66 | 0.004 | 4/4 | 0.50 |
  | ETH | 200 | 988 | 7.1 | 4.5 | 2.19 | 0.014 | 4/4 | 0.37 |
  | SOL | 100 | 510 | 3.5 | −4.6 | 0.36 | 0.36 | 3/4 | 2.25 |
  | SOL | 200 | 839 | 2.8 | −2.0 | 0.76 | 0.22 | 3/4 | 1.71 |
  | BTC ref | 100 | 352 | 9.8 | 4.9 | 2.85 | 0.002 | 3/4 | 0.53 |
  | BTC ref | 200 | 661 | 9.3 | 4.8 | 3.24 | 0.0006 | 4/4 | 0.51 |

  - **ETH** passes all three conditions at both rates, with Holm. By year
    (rate 100): 2.2 / 30.0 / 23.2 / 4.6 bps.
  - **SOL** fails conditions 1 and 2. Its result is entirely 2024 (best-10
    share above 1, so the remaining days lose). It is −3.8 bps in 2023.
  - Simple and log measurement differ by ≤ 0.35 bps.
  - **Reported cells:**
    - On ETH and BTC the edge needs immediate entry and a short hold. At a
      15-minute hold, ETH ordinary days fall to 0.4–0.7 bps, consistent
      with the front-loaded H-006 reversal.
    - The strictest trigger (50 per year) is the strongest on BTC/ETH
      (ETH: gross 21.1, ordinary 10.1, t 3.3; BTC: 24.7, 12.5, t 3.95).
      This is reported, not selected.
  - Outputs: `reports/stress_c007_{ETHUSDT,SOLUSDT}/`,
    `reports/c007_evaluation/results.json`; evaluator
    `scripts/evaluate_c007.py`.
  - The locked months are not read (no confirmation).
- **C-006 build notes (2026-10-01; no ETH/SOL 5–15 min return read):**
  - ETH event-bar cache: `features/alpha/symbol=ETHUSDT/impact_research_v2`,
    built by `scripts/build_impact_research_bars.py --symbol ETHUSDT
    --event-sizes 200 500 1000 2000 4000 8000 --time-bars` (no clock bars,
    which C-006 does not need). Coverage 2019-11 to 2026-05; the locked
    months are excluded.
  - A monitoring loop exited early, so a rerun overlapped the first build's
    final steps. The builder keeps existing outputs and writes through a
    temp file, so nothing was overwritten. Verified afterwards:
    - each size's bars × size equals the 2,123,710,806 events, less a
      sub-bar remainder;
    - timestamps are strictly ordered;
    - no temp files remain;
    - the manifest is complete.
  - Peak memory was about 1.2 GB.
- **C-006 run notes (2026-10-01, before any ETH/SOL result was read):**
  - Runner `scripts/run_c006.py` imports H-006's helpers unchanged:
    - the price clock and price lookups;
    - the OLS;
    - month and embargo logic, training-only winsorisation and the
      training-quantile tail cut;
    - trading in the forecast direction, with delayed entry.
  - Applies H-006's row usability filter, but does not build the impact
    component P (unused by `return_pretrend`).
  - **Correction to the registered text:** H-006 measures prices on the
    event_200 trade clock, not the 1-second grid. The frozen H-006 code is
    followed.
  - **Edges:** simple returns (linear perp) are primary; log edges are
    reported. The model fit uses log targets as in H-006.
  - Day-block t: per-day mean edge, then t across days. "Months positive"
    is the share of test months with a positive mean edge.
  - BTC is recomputed with the same code for reference, and to validate
    that the runner reproduces H-006.
  - **Book (reported):**
    - each trade adds 1/24 unit from its entry minute for 5 minutes,
      netted on a 1-minute grid;
    - `perp_pnl` accounting at gross, 2 bps and 1 bp per side;
    - returns are per unit of maximum absolute position.
  - **Duration-matched robustness:** the bar size whose 2022 median
    duration is closest (in log terms) to BTC's 1,000-event bars.
- **C-006 locked-month rule (fixed 2026-10-01, before any 2026-06..08 bar
  or return was built or read; the registration said only "read once,
  reject-only"):**
  - **Data:** BTC, ETH and SOL, decisions 2026-06-01 to 2026-08-31.
    - Event bars are rebuilt through 2026-08 into a separate cache
      (`impact_research_v2_locked`); the test-period cache is untouched.
    - The walk-forward is unchanged: each locked month's OLS is fitted on
      the preceding 12 months, so July's training includes June.
    - Trades whose 5-minute exit falls after 2026-08-31 drop out.
    - **Only the frozen primary cell is computed** (`return_pretrend`,
      1,000 events, top 1%, 5 min). No grid is computed, so nothing can be
      browsed.
  - **Statistic:** per instrument, the mean simple edge per trade and the
    day-block t. Pooled: the mean over instrument-days of each day's mean
    edge, with equal weight per instrument-day.
  - **Reject C-006 if either:**
    - (a) the pooled mean gross edge is below 2 bps (the registered T(5
      min)); or
    - (b) fewer than 2 of the 3 instruments have a positive mean gross edge.
  - **Otherwise "not rejected".** This is not a further confirmation.
    Reported: net of 4 bps, per-instrument t, best-day share, daily path.
  - **Power** (from the test-period per-day SD, about 37 bps): the pooled
    SE is about 2.5–4 bps over about 90 days. If the true edge is about
    8 bps, P(reject) ≈ 1%; at about 4 bps (fee level), ≈ 20%.
- **C-006 locked-month result (2026-10-01; single use, now spent): NOT
  REJECTED under the frozen rule, but the per-trade economics are weak.**
  - Pooled day-weighted mean gross 28.8 bps (97 instrument-days, t 7.5),
    with 3 of 3 instruments positive. Neither reject condition triggers.
  - **Per-trade (trade-weighted) means**, i.e. P&L at a fixed size per
    trade:

    | | Trades | Days | Per-trade gross | Day-weighted | Ordinary days |
    |---|---:|---:|---:|---:|---:|
    | BTC | 651 | 27 | **0.2 bps** | 28.2 | −12.8 |
    | ETH | 962 | 45 | **4.9** | 25.2 | −6.2 |
    | SOL | 288 | 25 | **28.3** | 35.8 | −14.6 |

    Pooled trade-weighted: 6.9 bps.
  - **Why the two weightings disagree:**
    - most signal days were small winners (80–89% of days positive);
    - the two large-move days were heavy losers on all three instruments:
      25 June (BTC 144 trades at −26.6 bps, ETH −27.4, SOL −22.6) and
      19 August (BTC 225 at −14.4, ETH −13.1, SOL −31.3);
    - SOL's per-trade result is mostly 22 August (91 trades at +81 bps).

    In the test period the large-move days carried about 60% of the edge.
    Here the two biggest went the other way.
  - The signal fired on about 30% of days, consistent with 2025–26.
  - **Interpretation:** with only two or three stress days in 92, this
    cannot distinguish "effect intact" from "effect gone". The rule
    (day-weighted, chosen before reading) does not reject. At fixed size per
    trade, BTC is flat and ETH is near the 4 bp fee level.
  - Any capital decision should rest on prospective data, and on a
    trade-weighted (P&L-relevant) criterion registered in advance.
  - Outputs: `reports/c006_locked/` (`results.json`, `trades.parquet`,
    `daily.parquet`); runner `scripts/run_c006_locked.py`. The locked
    caches are `features/alpha/symbol=*/impact_research_v2_locked`, verified
    identical to the test caches before June.
- **Outcome C-006 (2026-10-01): CONFIRMED on ETHUSDT and SOLUSDT.** All
  three conditions pass on both instruments.

  | Primary: `return_pretrend`, 1,000 events, top 1%, 5 min, delayed entry | ETH | SOL | BTC (ref) |
  |---|---:|---:|---:|
  | Trades (per day) | 12,988 (28.4) | 6,859 (18.3) | 12,208 (27.6) |
  | Gross, simple (log) | 12.7 (11.8) bps | 29.1 (26.2) | 15.5 (15.3) |
  | Ordinary days (excl. best 10) | 5.6 | 11.0 | 7.5 |
  | Day-block t | 13.5 | 12.9 | 14.3 |
  | Months positive | 71% | 83% | 83% |
  | 2023 / 2024 / 2025 / 2026 YTD | 6.4 / 30.8 / 10.5 / 7.4 | 38.6 / 84.0 / 6.0 / 10.5 | 13.9 / 30.3 / 9.8 / 4.4 |
  | 2023–24 vs 2025–26 | 16.0 → 9.8 | 60.7 → 7.1 | 19.8 → 8.4 |
  | Best-10-day share | 0.59 | 0.66 | 0.58 |

  - **Validation:** the runner reproduces H-006's saved BTC cell exactly
    (12,208 trades, log edge 15.26 bps, t 14.32 against 14.35).
  - **Executable-price check** (quotes 2023-05 to 2024-03, the last quote
    second strictly before entry and exit):

    | | Trade clock | Mid to mid | Taker after spread | Net of 2 bp fees | t |
    |---|---:|---:|---:|---:|---:|
    | BTC | 28.0 | 27.5 | 27.0 | 23.0 | 8.9 |
    | ETH | 23.6 | 23.3 | 22.7 | 18.7 | 6.9 |
    | SOL | 72.9 | 72.0 | 69.4 | 65.4 | 6.3 |

    - Bid–ask bounce is not the driver.
    - 9–14% of trades lacked a quote within 5 s. They fall on 3 days of
      Binance bookTicker archive outages (13 gaps, 21 h), and their
      edges are not systematically higher.
  - **Duration-matched robustness:**
    - ETH's matched size is 1,000 events (same as primary).
    - SOL's is 200 events (SOL bars are about 4.6× slower). Its result is
      in `results.json` → `duration_matched`.
  - **Book (1/24 unit per trade, netted, 1-minute perp P&L):** Sharpe gross
    / 2 bp / 1 bp per side:
    - ETH 1.56 / 1.16 / 1.36;
    - SOL 1.79 / 1.62 / 1.70;
    - BTC 1.93 / 1.58 / 1.76.

    Trades cluster heavily, with up to 66–134 overlapping trades at peak,
    and the book is mostly flat. Return per unit of peak exposure is small
    (2–8%/yr), so sizing per trade, not per book, is the practical
    question.
  - **Caveats:**
    - About 60% of the edge is on the best 10 days.
    - The effect is decaying on all three, most on SOL. 2025–26 is still
      7–10 bps gross, above the 4 bp round trip.
    - It likely overlaps with stress reversion (C-007): both trade against
      large recent moves.
  - **Next, per the frozen interpretation rule:**
    - the locked months 2026-06 to 2026-08 (BTC, ETH, SOL), a single use
      that can only reject;
    - prospective capture;
    - a fee-route check;
    - depth and slippage at entry for realistic size.
  - Outputs: `reports/c006/` (`results.json`, `trades.parquet`,
    `quote_check.json`); runner `scripts/run_c006.py`.
  - **Slippage check (2026-10-01; indicative only, per the researcher):**
    level-1 quotes only, 2023-05 to 2024-03, with archive gaps.
    `scripts/check_c006_slippage.py`; outputs `reports/c006/slippage_check.{json,csv}`.
    - Notional at the touch on the side taken at entry is thin: median /
      10th percentile BTC $30k / $1k, ETH $18k / $0.6k, SOL $1.4k / $0.2k.
      Touch coverage of a $50k order is 40% (BTC), 26% (ETH), 1% (SOL).
    - Square-root impact (Y = 1, trailing 24 h volatility and volume) per
      round trip:

      | Size | BTC | ETH | SOL |
      |---:|---:|---:|---:|
      | $10k | 0.3 bps | 0.5 | 2.7 |
      | $100k | 1.1 | 1.7 | 8.7 |
      | $1m | 3.3 | 5.3 | 27.4 |

      The one-tick-worse model is only a floor (0.01–0.8 bps).
    - In that window, net edges after spread, 4 bp fees and square-root
      slippage stay large (BTC 19.7, ETH 13.4, SOL 38.1 bps at $1m) because
      2023–24 edges were high.
    - **Against the 2025–26 gross edge (7–10 bps), with 4 bp fees and about
      0.5–1 bp spread:**
      - BTC and ETH stay positive up to roughly $100k–250k per trade;
      - SOL at $100k (8.7 bps of slippage) would be negative.
    - Trades cluster (up to about 100 overlapping, same side), so aggregate
      flow, not single-trade size, sets the real impact.
    - Deeper book data (L2) would be needed for anything definitive.

### P-006 · Prospective, trade-weighted test of C-006 (5-minute pretrend reversal)
- **Status:** registered (2026-10-01), before any data from 2026-09-01 onward
  was downloaded or read. Origin: C-006 (confirmed on ETH/SOL; locked months
  not rejected, but weak per trade).
- **Why trade-weighted:** P&L at a fixed size per trade is the mean over
  *trades*. The locked-month rule weighted instrument-days equally, which
  over-weighted light winning days against heavy losing ones.
- **Specification (frozen, unchanged from C-006):**
  - `return_pretrend`, 1,000-event bars (timestamp_direction_v1), top 1%
    tail, 5-minute hold;
  - delayed entry at the first event_200 endpoint at least 1 s after the
    decision;
  - monthly OLS on the preceding 12 months with the H-006 embargo;
  - simple returns; BTCUSDT, ETHUSDT and SOLUSDT;
  - runner `scripts/run_c006.py` with dates moved only, as in
    `run_c006_locked.py`.
- **Data:** Binance USD-M monthly trade archives from **2026-09** onward. The
  pipeline is trades → canonical fills → events → event bars, in a separate
  cache (`impact_research_v2_prospective`, rebuilt from history so the bars
  are continuous).
- **No peeking:** before each scheduled look, only data-quality checks are
  run (row counts, gaps, validation). No edge is computed.
- **Primary statistic:** the pooled trade-weighted mean net edge, *m* =
  mean over all trades (BTC + ETH + SOL) of (simple gross edge − 4 bps).
  Its t uses standard errors clustered by UTC day, pooling instruments,
  because trades bunch heavily within days.
- **Looks and decisions** (two looks; O'Brien–Fleming-type boundaries,
  one-sided α ≈ 0.025):
  - **Look 1, after 12 complete months (2026-09 to 2027-08):**
    - *pass* if t > 2.80 and at least 2 of 3 instruments have positive
      trade-weighted net;
    - *retire (futility)* if m ≤ 0;
    - otherwise continue to look 2.
  - **Look 2, after 24 months (to 2028-08), using all 24 months:**
    - *pass* if t > 1.98 and at least 2 of 3 instruments positive;
    - otherwise retire.
- **Power (stated in advance)** — from 2025-06 to 2026-05, the
  day-clustered SE is about 4.5 bps per 12 months and about 3.2 bps per
  24 months:
  - if the true net edge is about +0.6 bps (that year's trade-weighted
    value), the expected t is about 0.1–0.2, so the likely outcome is
    futility or fail;
  - if the true net edge is about 4 bps, the expected t is about 0.9 (12 m)
    and 1.25 (24 m), so pass probability is about 25% even then.

  This is a guard against a vanished effect, and a detector of a
  strengthening one. It is not a likely route to confirmation.
- **Reported at each look (no decisions):**
  - gross and net per instrument;
  - net at 3 and 2 bps per round trip (fee sensitivity);
  - best-10-day share;
  - **heavy versus light days** (days with more than 50 trades pooled,
    against the rest), because heavy days decided both the test-period and
    the locked-period results;
  - per-month edges;
  - if L2 data are bought, slippage at $10k/$100k.
- **Not covered:** real fills, queue, latency and slippage at size. A
  shadow-trading or paper step would be a separate registration.
- **Context (descriptive):** at the target 4 bp round trip the recent
  trade-weighted edge is about break-even. The strategy is only viable if
  costs fall further (net about +1.6 bps at 3 bps round trip, on that year)
  or the edge strengthens.
- **Data job (2026-10-01):** `scripts/p006_monthly_data.py` (tests in
  `tests/test_p006_monthly.py`).
  - Pipeline per pending month: checksum-verified download, parse into
    validated fills and events, then a rebuild of the
    `impact_research_v2_prospective` bar cache (200 and 1,000 events).
  - Quality checks read timestamps and counts only:
    - validation fields;
    - month coverage;
    - events per day;
    - the largest gap between events;
    - bars before 2026-09 identical to the verified cache;
    - bar ordering.
  - It never computes a signal, return or edge. `reports/p006/status.json`
    states the look rule and when look 1 becomes allowed (data through
    2027-08).
- **Outcome:** pending (look 1 after 2027-08).

## Exploration log

Exploratory analysis generates hypotheses; it tests nothing. Each entry records
what was sliced, so that any hypothesis it inspires is registered with its
origin known and the slices counted.

  - **Confirmation: same pattern** on ETH and SOL. Moderate bursts continue
    (ETH +0.19, SOL +0.29 bps at 5 s, 38 of 38 months). Extreme bursts also
    continue (60 s: ETH +0.22, SOL +0.53 bps), so the inverted U is rejected
    on all three instruments. The activity interaction passes on both.
  - True-mid robustness (9 months): moderate-burst continuation +0.12 bps at
    5 s (9 of 9 months); extreme bursts +0.06 bps at 60 s (44% negative).
    The verdicts are unchanged.
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

### E-004 · Why did H-010 fail on ETH and SOL? (2026-09-30)
- **Data:** the existing C-010 outputs, plus yearly Spearman ICs of each
  hourly-panel feature against 24-hour forward returns. Samples are daily at
  00:00 UTC to avoid overlap (about 365 a year, so SE ≈ 0.05 per yearly IC).
  BTC, ETH and SOL, 2021–2026. All of these returns were already used in
  H-009, H-010 and C-010, so this is diagnosis, not evidence.
- **Slices examined:** 8 features × 6 years × 3 instruments. Primary-variant
  Sharpe by year and instrument. Model attribution (BTC only).
- **Observations:**
  - *BTC's H-010 result is not statistically strong on its own.* Taker Sharpe
    0.69 over 3.4 years is t ≈ 1.3; beta-adjusted alpha t ≈ 1.0.
  - *ETH shared BTC's 2023.* Primary gross Sharpe in 2023 was BTC 2.02, ETH
    1.28, SOL −0.27. After 2023: BTC 0.48 / 0.13 / 0.88 (2024 / 2025 /
    2026 YTD); ETH −0.01 / −0.07 / −0.90. The instruments differ mainly
    after the effect weakened. Per-year Sharpe SE is about 1, so these gaps
    are consistent with one weak common effect (pooled taker Sharpe about
    0.2) plus noise.
  - *The crowding features have a consistent sign across instruments,
    fading over time.* Funding z, premium and 24-hour return are negative
    (contrarian) for all three in 2021–2023. BTC premium: −0.16 (2022),
    −0.05 (2023). ETH funding: −0.14 (2023). From 2024 they are about zero
    or mixed.
  - *The long/short ratio features, which drove BTC's backtest, show no
    consistent univariate IC on any instrument:* |IC| < 0.05 in most
    instrument-years, with signs that vary. Their BTC contribution came
    through the multivariate refit, a fragile source.
- **Leads (not results):**
  - (a) The crowding premium is a common cross-asset effect that has
    decayed since 2024. One possible cause is institutional cash-and-carry
    capital (spot ETFs from January 2024, CME basis) compressing funding
    extremes.
  - (b) If (a) holds, the effect should survive longer in less-arbitraged
    perps, which calls for a pooled, cross-sectional test on untouched alts.

### E-005 · Mutual-information screen of execution signatures at 8–24 h (registered protocol, 2026-10-01)
- **Status:** protocol frozen 2026-10-01, before any feature was computed. Development window only. It feeds H-013's selection rule; it tests nothing by itself.
- **Protocol:**

- **Window:** 2021-01 to 2023-12. Every instrument has events for this
  window, and the 1-second grid is built.
- **Sampling:** non-overlapping decisions, every 8 h for the 8 h target and
  daily for the 24 h target. That gives about 3,300 and about 1,100
  observations per pair.
- **Estimators:**
  - *Primary:* Gaussian-copula MI (rank-transform each variable to normal
    scores, then Gaussian MI). It is a robust lower bound, stable at n ≈ 1,000
    and fast.
  - *Robustness:* KSG kNN MI (k = 5) on rank-transformed data.
- **Conditioning (the orthogonalisation step; revision 2 after review).**
  The signed signatures (`ml_imbalance`, `sweep_pressure`,
  `large_imbalance`, `burst_skew`) are likely to be strongly correlated with
  plain flow and with the window's own return. E-001 found flow is mostly a
  proxy for the same-window return (Spearman 0.75). Marginal MI would
  therefore rank near-duplicates together.
  - **The primary screening statistic is conditional MI given
    Z = {`flow_imbalance`, own return over W}**, computed in Gaussian-copula
    space (Ince et al. 2017). It is the non-linear analogue of regressing out
    Z, with no rolling-regression choices to make.
  - KSG conditional MI (Frenzel–Pompe, k = 5) is the robustness check.
  - Marginal MI is reported alongside.
  - `flow_imbalance` itself is screened marginally, as the benchmark. Its
    conditional MI given itself is zero by construction.
- **Decomposition** (the key step for tradability). For each feature X and
  target r:
  - **I(X; r)**: all dependence;
  - **I(X; |r|)**: volatility information. It cannot be traded directionally,
    but it is useful for sizing and for H-018-style execution scheduling;
  - **I(X; sign r)** and the decile conditional-mean curve E[r | X]: the
    *directional* information that pays.

  The crypto-impact-study notebooks (`DataTest*.ipynb`) found that MI was
  positive at many lags while correlation was near zero. The likely
  explanation is volatility clustering, which this decomposition separates
  out.
- **Nulls:**
  - circular shifts of the feature series against the target, with shifts of
    at least 30 days and 1,000 shifts. This preserves both autocorrelations.
  - It replaces the single random shuffle used in the earlier notebooks.
- **Multiple testing:** 7 features × 4 targets × 2 horizons ×
  (own + cross + market) pairs. Benjamini–Hochberg at q = 0.10 on the
  circular-shift p-values of the conditional I(X; sign r | Z).
- **Side table (choice d; cannot feed H-013's selection):** the same screen
  for the hourly crowding features (funding z, premium, basis, OI change,
  long/short ratios) against the same targets, both marginal and
  conditional. It tests whether H-009/H-010's linear specification missed a
  non-linear shape. It has its own BH family and is reported separately.
- **Outputs:** a ranked table of excess directional MI, excess volatility MI
  and the conditional-mean curves. The table goes into the exploration log.
  No P&L is computed in E-005.
- **Build notes (2026-10-01; no return has been read):**
  - Features are built by `scripts/build_signature_features.py`. Hourly
    tables are `features/alpha/symbol=X/signature_hourly_v2`, and panels
    `signature_panel_v2`. Coverage: BTC 2019-10, ETH 2019-12 and SOL 2020-10,
    each to 2026-05. The locked months are not built.
  - v1 was discarded before any use, for two reasons:
    - its z-scores included the current hour; they now use strictly earlier
      values, per the catalogue's `z30`;
    - `sweep_pressure` multiplied the side-aligned sweep depth by the
      aggressor sign twice, making it unsigned. A feature–feature
      correlation check showed this (0.71–0.86 with unsigned `ml_share`).
      v2 uses the registered signed depth.
  - Feature–feature rank correlations (development window, 8 h, z):
    - `ml_imbalance`–`flow_imbalance` 0.88–0.90;
    - `flow_imbalance`–own return 0.66–0.77;
    - `large_imbalance`–flow 0.51–0.71 on BTC/ETH, about 0 on SOL;
    - `absorption_imbalance`–`burst_skew` −0.63 to −0.65.

    This confirms the need for the conditional screen.
  - SOL `large_imbalance` z is undefined for 22% of development hours,
    because $1m events are rare and the 30-day variance is often zero.
    Those hours drop out of that feature's screen.
- **Run notes (implementation choices, fixed before any return was read):**
  - Feature window W = horizon H.
  - Non-overlapping decisions at 00/08/16 UTC (8 h) and 00 UTC (24 h), with
    T ≥ 2021-01-01 and T + H ≤ 2024-01-01.
  - Z = {source flow, source own return}, plus the target's own past return
    for cross pairs.
  - Null: 1,000 circular shifts of the forward return against all
    predictors (shifts ≥ 30 days; seed 20261001).
  - Zero returns dropped.
  - Validated on synthetic data:
    - the Gaussian MI value is reproduced;
    - CMI is about 0 under a common cause;
    - the discrete-target estimate matches a binned estimate;
    - null p-values are roughly calibrated (250 AR(1) trials: 2.4–7.6% of
      p < 0.05 across the three statistics).
  - Script `scripts/run_mi_screen_e005.py`; outputs in
    `reports/e005_mi_screen/`.
- **Observations (2026-10-01; development window 2021–2023):**
  - **Direction: a clean null.**
    - 0 of 192 signature cells pass BH at q = 0.10.
    - 4.2% have conditional p < 0.05 (5% expected by chance), with minimum
      p 0.009 (about the expected minimum of 192 uniforms).
    - The largest excess directional MI is 0.008 bits, and |Spearman| ≤ 0.10
      everywhere.
    - The benchmark flow (marginal) is also null (minimum p 0.021, 1 of 32
      below 0.05).
  - **Crowding side table (separate family):** 0 of 224 pass BH; 7.1%
    have p < 0.05; minimum p 0.004 (SOL 24 h OI change → alt basket).
    Consistent with E-004, H-019 and H-020: no non-linear shape was missed
    by the linear specifications.
  - **Volatility information (H-013 secondary, descriptive):**
    - unsigned urgency `ml_share` is informative about |r| in 62% of its
      cells (excess about 0.006 bits), conditional on flow and own return;
    - `large_imbalance` in 19%; the signed signatures in 0–9%.
    - Recent realised volatility was **not** in Z, so this may only proxy
      volatility clustering. It needs a check against 24 h realised
      volatility before any use in execution scheduling.
  - **Follow-up (2026-10-01): urgency's volatility information is
    volatility clustering.** `scripts/explore_urgency_volatility.py`; output
    `reports/e005_mi_screen/urgency_volatility/`.
    - With trailing log realised volatility (24 h, 7 days, and the window)
      added to Z, conditional MI of `ml_share` with |r| and with next-window
      realised volatility is null on all 12 cells (p 0.16–0.98). Without it,
      p is 0.001–0.08.
    - HAR R² gain from adding `ml_share` is ≤ 0.004.
    - The BTC coefficient is −0.04 log-RV per sd (t −3.5 at 8 h), the
      opposite sign to the hypothesis. ETH is about 0 and SOL +0.02.
    - Not useful for volatility forecasting or execution scheduling.


### E-006 · Re-score of saved results under the revised objective (2026-10-01)
- **What:** `scripts/run_rescore_e006.py`, using saved outputs only. No new
  data or returns are read. Outputs in `reports/e006_rescore/`; report
  `reports/e006-rescore-results.md`.
- **Scope:** 1,696 per-trade cells from H-003, H-004, H-006, H-007, H-008,
  H-009, H-012 and E-002, each with:
  - horizon τ;
  - gross mean return per trade and gross on ordinary days (excluding the
    best 10 days);
  - day-block t;
  - the threshold T(τ);
  - net at the target fees (4 bps round trip).

  Book strategies (H-010, H-020) are in a separate table.
- **Candidate rule** (descriptive, not a test): ordinary-day gross ≥ T(τ),
  ordinary-day net > 0, and t > 2.
- **Observations:**
  - **H-006 pretrend-reversal tails:** 37 of 576 cells qualify. All are the
    top 1% at 5–10 min, across every bar size and model. The cells overlap
    heavily: it is one effect.
    - Example: `return_pretrend`, 1,000 events, 5 min: gross 15.1 bps,
      ordinary days 7.4, t 14.4, 28 trades a day.
    - Positive every year (2023 14.0, 2024 29.7, 2025 9.5, 2026 4.4 bps).
      About half the edge is from the best 10 days, and 2026 is weakening.
    - Delayed entry (median 8–11 s) equals immediate entry.
    - The edge is front-loaded. The ordinary-day mean across all top-1%
      cells is +4.3 bps at 5 min, +2.0 at 10, −2.0 at 15 and −3.7 at 30.
  - **H-007/H-008 stress reversion:** 6 of 72 cells qualify in E-006's
    pooling across trigger rates. At the registered primary (100 episodes
    a year, trigger entry, 5 min hold): v1 gross 9.8, ordinary 4.9, t 2.9
    (2025 −6.6 bps); v2 gross 11.9, ordinary 4.8, t 3.3. About 1 trade a
    day.
  - **E-002 level runs** (BTC 2022 exploration): 4 cells pass, but on tiny
    samples (2–3 events a day, t undefined or 6.6), measured with the trade
    mid proxy (about 20% overstated, V-001). Noted only.
  - **No candidates:** H-003 (no ordinary-day measure; 2020–21 overlapping
    signals), H-009 (passes only through the loose 10 bps cap at long
    horizons), H-004 and H-012.
  - **Books at 1–2 bps per side:**
    - H-010 BTC: 18–19%/yr, Sharpe 0.87–0.93 (log-return P&L; failed
      C-010).
    - H-020 funding level: Bybit Sharpe 0.34–0.41; Binance (used data)
      0.99–1.05.
- **Leads:** confirm the H-006 5-minute pretrend reversal and the H-007
  stress reversion on ETH/SOL 5–15 minute returns, which no previous test
  has examined. See C-006/C-007 (draft).

### E-007 · Big-day caps for C-006: development exploration (2026-10-01)
- **Purpose:** design input for a refinement hypothesis (H-021), to be tested
  only on P-006's prospective data. Uses data already spent: the C-006 test
  2023-01 to 2026-05 and the locked months.
- **Grid (fixed before computing):**
  - concurrent-position cap N ∈ {1, 3, 5, 10} open trades per instrument
    (later signals are skipped while N are open; real-time implementable);
  - daily cap K ∈ {10, 25, 50}: the first K signals per instrument per UTC
    day.
- **Selection rule (fixed before computing):** the variant with the highest
  trade-weighted, day-clustered t of net edge (gross − 4 bps, pooled
  BTC + ETH + SOL) over 2023-01 to 2026-05, keeping at least 30% of
  uncapped trades. The locked months are reported descriptively only.
- **Observations (2026-10-01; outputs `reports/e007_big_day_caps/results.json`):
  the development data argue against a cap.**

  | Variant | Kept | Net per trade, 2023-01 to 2026-05 | t | Net, 2025-06 to 2026-05 | Locked net (t) |
  |---|---:|---:|---:|---:|---:|
  | Uncapped (C-006) | 100% | 13.2 bps | **3.72** | +0.6 | +2.9 (0.29) |
  | Concurrent ≤ 1 / 3 / 5 / 10 | 10 / 25 / 36 / 56% | 2.9 / 3.4 / 3.9 / 4.4 | 1.40–1.60 | −4.4 to −3.0 | −6.4 to −4.8 |
  | Daily first 10 / 25 / 50 | 28 / 51 / 71% | 5.2 / 5.1 / 4.6 | 1.04–1.47 | −3.2 to −4.0 | +9.0 / +7.3 / +2.7 |

  - The frozen rule picks `concurrent_10` among capped variants, but every
    cap has a **lower** t than no cap.
  - In the most recent test year all caps are net negative (light-day trades
    lose after fees), while uncapped is +0.6 bps.
  - Only the locked months, the observation that motivated the idea, favour
    daily caps. Selecting on them would be circular.
  - **Conclusion:** the profit is concentrated on heavy (stress) days, and
    light-day trades lose after fees. A cap is not supported, so H-021
    (cap) is not recommended for registration.
  - **Post-hoc lead (from used data):** the *opposite* refinement, trading
    only once a stress regime is established (e.g. after several signals
    that day), would concentrate on where the edge has been. It overlaps
    with C-007, and the locked months' heavy days lost. Any such variant
    needs its own registration and P-006's prospective data, with the
    multiple-testing split stated.
  - **Researcher decision (2026-10-01):** neither the cap nor the
    stress-only variant is registered. C-006 stays as frozen and P-006 runs
    unchanged, with the full α. Effort moves to the cost side (fee route,
    depth data).

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

### V-002 · Simulator fill-model validation (calibration days only, 2026-09-30)
- **Type:** measurement validation. No test month was simulated.
- **Code:** `src/alpha_search/simulator.py` (10 unit tests in
  `tests/test_simulator.py`), `scripts/build_quote_triggers.py`,
  `scripts/validate_simulator.py`; outputs in `reports/simulator_validation/`.
- **Naive quoter S0** (unit $1,000, inventory ±5, 100 ms latency), 77
  calibration days each:

  | | BTC | SOL |
  |---|---:|---:|
  | Tick | 0.03 bps | 0.50 bps |
  | Fills per day | 22,157 | 32,585 |
  | Filled by trade-through | 51% | 73% |
  | Median wait in queue | 1.2 s | 0.9 s |
  | Spread capture at the fill | 0.00 bps | −0.26 bps |
  | Markout 1 s / 60 s | −0.60 / −0.80 bps | −1.10 / −1.17 bps |
  | Net per fill after the 2 bps fee (60 s) | −2.80 bps | −3.17 bps |
  | Days with positive 60 s markout | 1% | 1% |

  Back-of-queue fills are far more adversely selected than the average maker
  fill in E-002 (about −0.1 bps). That is plausible, since the last order in
  the queue fills mainly when the level is swept.
- **Design findings for H-017 and H-018, before any test:**
  1. *H-017 condition 2 is arithmetically unattainable on BTC, ETH and SOL.*
     Even with zero adverse selection, a fill earns at most half a tick
     (about 0.25 bps on SOL) against a 2 bps fee.
  2. *H-017 condition 1 (S1 − S0 net P&L per day) is degenerate.* S0 loses on
     almost every fill, so any rule that quotes less "improves" P&L. It
     rewards not trading rather than better selection.
  3. *Trigger coverage:* the registered burst rule (deciles 1–4 and 7–10)
     keeps each side pulled 46–66% of the day. The sweep rule rarely fires
     (under 1% on BTC, 1–7% on SOL, even at D = 2 bps), so the D grid is
     nearly inert. The k = 3 level rule pulls 50–72% of the day on BTC.
     H-018's run-away rule (burst deciles 2–4 and 7–9) fires similarly
     often.

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
