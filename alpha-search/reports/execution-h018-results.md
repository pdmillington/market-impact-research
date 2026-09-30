# H-018 · Signal-timed execution: results

Registered 2026-09-30 and amended before any test (see
`HYPOTHESIS_LEDGER.md`, H-018: amendment, run notes, outcome).
Scripts: `scripts/run_execution_h018.py`, `scripts/evaluate_execution_h018.py`.
Outputs: `reports/execution_h018/`.

## Verdict (BTC and SOL; ETH pending its quotes, which cannot change either verdict)

- **Condition 1 (passive beats taker): passes.**
- **Condition 2 (signals improve passive execution): fails.** E2 is
  significantly worse than E1 on both instruments.
- **Condition 3 (tail): passes.**
- E3 is not run.

## Calibration (2023-05-16 to 2023-07-31)

- Frozen E2 for both BTC and SOL: **m = 10 ticks, T = 5 s, k = 12**. This is
  the least active corner of the grid.
- Every E2 configuration had higher mean IS than E1.

## Test (2023-08-01 to 2024-03-31, 2,925 parent orders each, $10k, W = 30 min)

| | BTC E0 | BTC E1 | BTC E2 | SOL E0 | SOL E1 | SOL E2 |
|---|---:|---:|---:|---:|---:|---:|
| Mean IS (bps) | 5.01 | 3.09 | 3.68 | 5.25 | 3.25 | 3.56 |
| IS sd | 0.18 | 2.07 | 2.11 | 0.52 | 3.16 | 3.34 |
| IS 95th / 99th percentile | 5.02 / 5.55 | 7.07 / 10.79 | 7.19 / 9.10 | 5.79 / 7.06 | 8.90 / 14.88 | 9.49 / 14.57 |
| Spread (bps) | 0.01 | 0.08 | 0.06 | 0.24 | 0.33 | 0.33 |
| Drift (bps) | −0.01 | 1.01 | 0.45 | 0.00 | 0.91 | 0.75 |
| Fee (bps) | 5.00 | 2.00 | 3.18 | 5.00 | 2.00 | 2.49 |
| Passive share | 0% | 100% | 61% | 0% | 100% | 84% |
| Median time to fill (s) | 0.1 | 6.8 | 3.8 | 0.1 | 2.5 | 2.0 |

| Difference (bps, day-clustered t, months positive) | BTC | SOL |
|---|---|---|
| E0 − E1 (condition 1) | +1.92, t 52, 8/8 | +2.00, t 35, 8/8 |
| E1 − E2 (condition 2) | −0.60, t −21, 0/8 | −0.32, t −8, 0/8 |
| E2 p95 − E1 p95 (condition 3) | +0.12 | +0.59 |

### Sensitivities (not selected on)

| | BTC E0 − E1 | BTC E1 − E2 | SOL E0 − E1 | SOL E1 − E2 |
|---|---:|---:|---:|---:|
| Latency 250 ms | +1.92 | −0.59 | +1.98 | −0.27 |
| Optimistic queue | +2.32 | −0.62 | +2.01 | −0.31 |
| $50k | +1.85 | −0.60 | +2.07 | −0.32 |
| W = 60 min | +1.91 | −0.60 | +1.96 | −0.32 |

## Interpretation

- **Patient passive execution saves about 2 bps per side** against crossing
  at base-tier fees: 3 bps of fee saved, minus about 1 bp of drift while
  waiting. The saving is stable across months, latency, queue model, size
  and window.
- **The signals do not help execution.**
  - Run-away triggers cut drift by 0.2–0.6 bps, but each cross pays 3 bps
    more in fees.
  - To pay, a trigger would need to foresee a move of at least 3 bps. The
    confirmed effects are tenths of a bp.
- **Implication for the alpha search:** use E1-style entry and exit. A
  strategy's round-trip cost falls from about 10 bps (taker) to about 6 bps
  (maker fee plus drift). Signals previously rejected at taker cost with a
  gross edge between 6 and 10 bps per round trip are candidates for
  re-registration with E1 execution.
- **Caveat:** these are random-direction orders. Alpha-driven orders trade
  with momentum, so their E1 drift would be larger than 1 bp.
