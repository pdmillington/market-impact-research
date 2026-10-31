# E-008 · Rolling-window calibration of the activity surface: results

Registered 2026-10-16; amendment 1 (adaptive grid) 2026-10-29; run 2026-10-29.
Script `scripts/e008_rolling_surface.py`; outputs `reports/e008_rolling_surface/`
(`summary.json`, `fits.csv`, `scales.csv`, `monthly_scores.csv`, `daily_contributions.csv`).
A calibration test of the contemporaneous block response, not return prediction.

## Verdict

**No evidence that a 12-month window beats the expanding fit.** The registered rule
needs all three conditions, and none holds:

| Condition (primary: trailing 12 vs expanding, 56 test months, 2022-01 to 2026-08) | Value | Met? |
|---|---:|:-:|
| Mean relative MSE improvement D, 95% month-block CI excludes 0 | +0.24% [−0.20%, +0.67%] | no |
| D > 0 in at least 60% of test months | 46% | no |
| Mean D > 0 with the best 10 days removed | +0.05% | yes, but negligible |

Alpha-search work therefore uses the expanding (development-sample) calibration.

## Secondary results

| Comparison | Raw mean D [95% CI] | Months D > 0 | Clipped y: mean D [95% CI] |
|---|---:|---:|---:|
| Trailing 12 vs expanding (primary) | +0.24% [−0.20, +0.67] | 46% | −1.03% [−1.45, −0.65] |
| Trailing 6 vs expanding | +0.23% [−0.32, +0.77] | 46% | −1.24% [−1.71, −0.80] |
| Trailing 24 vs expanding (from 2023-01) | −0.17% [−0.50, +0.17] | 34% | −0.87% [−1.08, −0.67] |
| Trailing 12 vs expanding (from 2023-01) | +0.22% [−0.31, +0.77] | 43% | −1.21% [−1.71, −0.75] |

- **With y clipped at the training window's 0.1% / 99.9% quantiles, every trailing
  window is significantly *worse* than expanding** (by about 1% of MSE). The small,
  insignificant raw gains come from extreme blocks; on typical blocks the shorter
  windows lose.
- **By year (primary, raw):** 2022 +0.3%, 2023 +2.2%, 2024 −0.2%, 2025 −0.2%,
  2026 −1.4%. Even after the 2024 change in the full-sample by-year η, trailing
  fits do not help in 2024–26.
- **Parameter paths:** trailing-12 η ranges 0.23–0.52 (mean 0.39; expanding
  0.41–0.48), and its α hits the upper bound 8 in some months (α range 1.9–8.0;
  expanding 3.3–4.3). Short windows track η's drift but estimate the shape
  noisily, and the noise costs more than the drift gains.
- **Out-of-sample R² against a zero prediction** (all windows alike): 0.45 at
  N = 250 rising to 0.62 at N = 16,000. The surface explains about half of the
  block-level variance of the signed contemporaneous return, whichever window is
  used.
- **Coverage:** with the adaptive grid every N was kept in every window-month;
  all fits converged.

## Use in the alpha search

- An impact or execution-cost input built on the surface should use the
  expanding (or full development-sample) calibration; re-estimating monthly
  on a trailing year adds noise, not accuracy.
- This says nothing about predictive (forward-return) value. Surface residuals as
  an alpha factor remain rejected (H-004 to H-006) unless re-registered with a new
  economic argument.
