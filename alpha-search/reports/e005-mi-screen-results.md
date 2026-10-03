# E-005 · Mutual-information screen of execution signatures at 8–24 h: results

Registered protocol 2026-10-01 (see `HYPOTHESIS_LEDGER.md`: E-005, H-013).
Script: `scripts/run_mi_screen_e005.py`. Features: `scripts/build_signature_features.py`
(catalogue section "Execution-signature features").
Outputs: `reports/e005_mi_screen/`:

- `cells.parquet` / `.csv`;
- `crowding_side_table.parquet` / `.csv`;
- `selection.json`.

## Verdict

- **No directional information.** 0 of 192 signature cells pass the
  false-discovery correction (BH, q = 0.10).
- H-013's frozen selection is therefore empty, and H-013 is not run.
- The test window and the locked months remain unused by H-013.

## Design

- **Window:** development 2021-01 to 2023-12, with non-overlapping
  decisions. That gives about 3,270 per pair at 8 h and about 1,095 at 24 h.
- **Pairs:** 16 source/target pairs. Sources are BTC, ETH, SOL and a market
  composite; targets are BTC, ETH, SOL and the alt basket.
- **Cells:** 6 signatures × 16 pairs × 2 horizons = 192.
- **Statistic:** Gaussian-copula conditional MI given source flow, source
  return and (for cross pairs) the target's own return.
- **Null:** 1,000 circular shifts of at least 30 days.

## Results

| | Cells | p < 0.05 | p < 0.01 | min p | BH passes |
|---|---:|---:|---:|---:|---:|
| Signatures, direction (conditional) | 192 | 4.2% | 0.5% | 0.009 | 0 |
| Signatures, volatility (conditional) | 192 | 17.2% | | | (descriptive) |
| Flow benchmark, direction (marginal) | 32 | 3.1% | | 0.021 | |
| Crowding side table, direction (conditional) | 224 | 7.1% | 0.4% | 0.004 | 0 |

- **Size of the strongest cells:** the largest excess directional MI is
  0.008 bits (24 h, market sweep pressure → ETH), and |Spearman| ≤ 0.10
  throughout.
- **Chance level:** the p-value distribution is what pure chance gives.
- **Volatility (descriptive):**
  - unsigned urgency (`ml_share`) is informative about the size of the next
    8–24 h move in 62% of its cells, conditional on flow and return;
  - the signed signatures show almost none.
  - Recent realised volatility was not conditioned on, so this is probably
    volatility clustering seen through urgency. Check it before any use.

## Interpretation

- **Fast, not persistent.** The seconds-scale execution signatures
  (H-014 to H-016: robust, 38/38 months) are absorbed within seconds to
  minutes. Aggregated over 8–24 h they predict neither direction nor, beyond
  plain flow and returns, anything tradeable.
- **Non-linear screening found no missed shape.** The crowding features'
  failure in H-009, H-010, H-019 and H-020 was not caused by assuming
  linearity.
- **What remains:**
  - the execution result (H-018: patient passive entry saves about 2 bps
    per side);
  - possibly volatility information for sizing and execution timing.
