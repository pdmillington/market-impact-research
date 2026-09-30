# Seconds-scale signals test v1 (H-014 to H-016): results

Registered on 2026-09-30 (`HYPOTHESIS_LEDGER.md`, common seconds-scale protocol)
and run the same day. BTCUSDT reconstructed events, 38 test months (January 2023
to May 2026, excluding May–July 2023), trade-based mid proxy. The true-mid
robustness run (August 2023 to April 2024) follows once `bookTicker` is
downloaded.

- `configs/seconds_signals_test_v1.json`, `scripts/run_seconds_signals_test.py`
- Outputs: `reports/seconds_signals_test_v1/` (`monthly_proxy.parquet`,
  `summary_proxy.csv`, `evaluation.csv`, `resiliency_residual_fit.json`)

## Registered verdicts

| Hypothesis | Condition | Result |
|---|---|---|
| H-014 sweep toxicity | 1 main effect (IC, τ = 5 s) | **pass**: 0.191, t 31.6, 38/38 months |
| | 2 robustness horizons | **pass**: 0.34 / 0.11 / 0.06 at 1 / 15 / 60 s |
| | 3 activity interaction | **pass**: +0.10 bps, t 4.6, 71% of months, *decaying* |
| | 4 resiliency orthogonal to activity | **fail**: −0.12 bps, 20% of months |
| H-015 level-depletion break | 1 main effect | **pass**: 0.105, t 44, 38/38 |
| | 2 robustness horizons | **pass** |
| | 3 absorption interaction | **pass**: +0.25 bps, t 11.5, 95% of months, *strengthening* |
| H-016 burst inverted U | 1 moderate bursts continue | **pass**: +0.17 bps, t 33, 38/38 |
| | 2 extreme bursts reverse | **fail**: +0.23 bps (continue), 26% of months negative |
| | 3 activity interaction | **pass**: +0.15 bps, t 21, 38/38 |

H-014 and H-015 survive on their main effects and primary interactions. The
inverted U of H-016 is rejected: in 2023–26 even extreme bursts continue.

## Stability by year (monthly means)

| Year | H-014 IC | H-014 activity interaction (bps) | H-015 absorption interaction (bps) | H-016 extreme bursts, 60 s (bps) |
|---|---:|---:|---:|---:|
| 2023 | 0.161 | 0.238 | 0.137 | −0.050 |
| 2024 | 0.175 | 0.150 | 0.240 | 0.418 |
| 2025 | 0.215 | 0.013 | 0.284 | 0.245 |
| 2026 (Jan–May) | 0.222 | −0.039 | 0.398 | 0.240 |

The main effects strengthen over time. The activity interaction that made
sweeps less toxic in busy markets has faded to zero since 2025. The absorption
interaction grows every year.

## Protective value (diagnostic)

Maker markout at 5 s of same-side fills in the 5 s after a flagged event:

- after a 5+ level sweep: **−0.33 bps**, against −0.09 bps for other fills;
- after a 6+ hit level run: −0.26 against −0.23 bps.

Pulling quotes on the swept side for a few seconds after a deep sweep avoids
fills that are about 3.6 times as toxic as average. Level-run flags are much
weaker as a withdrawal rule, despite their strong IC.

## Interpretation and caveats

- These are the most statistically robust effects in the project: every main
  effect is positive in all 38 months. They are seconds-scale and small in bps
  (a few tenths of a bp), and BTC cannot pay a 2 bps maker fee from them. That
  was stated in advance (no economic gate on BTC).
- The trade-based mid overstates 1–9 level continuation by about 15–20%
  (V-001). The true-mid re-run will quantify this on the test months.
- Resiliency's level has drifted (2022 median about −0.35, 2025 about −0.70),
  leaving a frozen-cut tercile empty in three months. It is consistent with
  sweeps continuing more over time, as the growing H-014 main effect also
  suggests.

## Next

1. True-mid robustness on August 2023 to April 2024.
2. Confirmation: the frozen specifications on ETHUSDT and SOLUSDT, 2023–26.
3. Economics on wider-spread contracts: protective value against spread capture
   and the 2 bps maker fee.

## Confirmation on ETHUSDT and SOLUSDT (frozen BTC specification)

The same code, cut points, thresholds, 2022 residual fit and 38 test months,
applied unchanged. Outputs in `reports/seconds_signals_test_v1/confirmation_<symbol>/`.

| Condition | BTC | ETH | SOL |
|---|---|---|---|
| H-014 main effect (IC, 5 s) | 0.191 ✓ | 0.183 ✓ | 0.170 ✓ |
| H-014 robustness horizons | ✓ | ✓ | ✓ |
| H-014 activity interaction (bps, share of months) | +0.10, 71% ✓ | +0.14, 97% ✓ | +0.26, 95% ✓ |
| H-014 resiliency orthogonal to activity | ✗ | ✗ | ✗ |
| H-015 main effect (IC, 5 s) | 0.105 ✓ | 0.108 ✓ | 0.150 ✓ |
| H-015 robustness horizons | ✓ | ✓ | ✓ |
| H-015 absorption interaction (bps) | +0.25 ✓ | +0.36 ✓ | +1.00 ✓ |
| H-016 moderate bursts continue (bps, 5 s) | +0.17 ✓ | +0.19 ✓ | +0.29 ✓ |
| H-016 extreme bursts reverse | ✗ (+0.23) | ✗ (+0.22) | ✗ (+0.53) |
| H-016 activity interaction | ✓ | ✓ | ✓ |

The verdicts are identical on all three instruments. H-014 and H-015
replicate, and H-016's inverted U is rejected everywhere, with extreme bursts
continuing.

**Protective value** (5 s markout of same-side fills after a 5+ level sweep,
against other fills): BTC −0.33 vs −0.09, ETH −0.34 vs −0.17, **SOL −0.15 vs
−0.24 (reversed)**. SOL's tick is about 1 bp, against about 0.02 bp for BTC
and about 0.03 bp for ETH, so "5 price levels" means a far larger move on SOL
and the flag is not comparable across instruments. Future protective rules
should use sweep depth in bps (scale-free) rather than a level count.

Effect sizes grow with the instrument's tick and volatility: SOL's absorption
interaction is 1.0 bps against 0.25 bps for BTC, which matters for the
economics stage.

## True-mid robustness (August 2023 to March 2024, real quote mids)

Every effect keeps its sign in 100% of the 9 months. Magnitudes are about 20%
smaller than with the trade-based mid over the same months, matching V-001:

| Metric | True mid | Proxy, same months |
|---|---:|---:|
| H-014 IC at 5 s | 0.141 | 0.175 |
| H-014 activity interaction (bps) | 0.179 | 0.195 |
| H-015 IC at 5 s | 0.088 | 0.111 |
| H-015 absorption interaction (bps) | 0.208 | 0.133 |
| H-016 moderate bursts at 5 s (bps) | 0.118 | 0.146 |
| H-016 extreme bursts at 60 s (bps) | 0.060 | 0.320 |

The verdicts are unchanged: H-014 and H-015 pass; H-016's inverted U is
rejected (extreme bursts do not reverse).
