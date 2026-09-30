# H-017 · Signal-protected passive quoting: results

Registered 2026-09-30; see `HYPOTHESIS_LEDGER.md` (H-017, run notes, outcome).
Scripts: `scripts/run_quoting_h017.py`, `scripts/evaluate_quoting_h017.py`.
Simulator: `src/alpha_search/simulator.py` (validated in V-002).
Outputs: `reports/quoting_h017/`.

## Verdict (SOL; ETH control pending its quotes)

**Fails.** Condition 1 passes as written but is degenerate; condition 2 fails.

## Calibration (2023-05-16 to 2023-07-31, 77 days)

- S1 is frozen at **D = 2 bps, T = 5 s, k = 3** (`selection_SOLUSDT.json`).
- Net P&L per fill was −3.10 bps (S0) and −3.11 to −3.13 bps across all 12
  S1 configurations.
- The ranking follows time quoting: pulling longer (T = 5) means fewer fills
  and a smaller loss.

## Test (2023-08-01 to 2024-03-31, 244 days, $20 unit)

| Run | Fills/day | Time quoting | Net/fill (bps) | Spread capture | 1 s markout | 60 s markout | Net/day (bps of unit) | t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 63,442 | 73% | −2.98 | −0.24 | −0.92 | −0.99 | −188,900 | −31.5 |
| S1 | 22,859 | 32% | −2.98 | −0.25 | −0.97 | −1.00 | −68,165 | −32.4 |
| S0 latency 250 | 39,993 | 61% | −3.03 | | | | −121,048 | |
| S1 latency 250 | 17,069 | 29% | −3.04 | | | | −51,933 | |
| S0 optimistic queue | 74,244 | 73% | −2.87 | | | | −212,756 | |
| S1 optimistic queue | 26,600 | 32% | −2.89 | | | | −76,752 | |
| S0 inventory ±2 | 55,294 | 67% | −2.95 | | | | −163,134 | |
| S1 inventory ±2 | 18,678 | 28% | −2.98 | | | | −55,739 | |

- **Condition 1:** S1 − S0 = +120,700 bps of unit per day, t 30, 8 of 8
  months. This holds under all three sensitivities.
  - The difference comes only from volume: net per fill is unchanged. The
    pulls do not improve the fills S1 does take (1 s markout −0.97 vs −0.92
    bps).
  - V-002 identified this degeneracy before the test.
- **Condition 2:** S1 loses every month (t −32).
- **Break-even tick:** about 5.5 bps. This holds the measured adverse
  selection fixed: 2 × (fee − (60 s markout − spread capture)).

## By month (S1 net per fill vs tick)

| Month | S0 net/fill | S1 net/fill | Tick (bps) |
|---|---:|---:|---:|
| 2023-08 | −3.00 | −3.08 | 0.45 |
| 2023-09 | −3.06 | −3.06 | 0.52 |
| 2023-10 | −3.00 | −2.98 | 0.39 |
| 2023-11 | −2.89 | −2.92 | 0.19 |
| 2023-12 | −2.98 | −2.99 | 0.13 |
| 2024-01 | −2.89 | −2.90 | 0.10 |
| 2024-02 | −2.95 | −2.96 | 0.09 |
| 2024-03 | −3.08 | −3.07 | 0.06 |

## Interpretation

- A one-lot level-1 quoter on SOL loses about 1 bp per fill to adverse
  selection, and the 2 bps maker fee takes the rest.
- The seconds-scale signals (H-014 to H-016) predict direction, but pulling
  quotes on them does not pick out less toxic fills. This is plausibly
  because most fills are trade-throughs during moves the triggers only
  detect after the fact.
- Passive market making at base-tier fees is not viable on these contracts.
