# Impact research redo: extended sample and the activity-adjusted surface (October 2026)

## What this is

The redo of the market-impact paper's P&B analysis on BTCUSDT, with two
changes:

- **More data.** The primary sample is 2021-01 to 2026-08. 2019-10 to 2020-12
  is robustness. The paper used 2025-01 to 2026-05.
- **A new leading model.** The activity-adjusted P&B surface, in which impact
  depends on both relative imbalance and relative activity.

Data from 2026-09 onward is excluded: it is reserved for the alpha-search
P-006 test.

**Researcher decisions (2026-10-05):**

- BTC only;
- 2021 to Aug 2026 primary, 2019–2020 robustness;
- no quote-based pinning for now;
- surface code from alpha-search not moved;
- volatility-normalised response primary;
- one common γ, with γ by N reported.

**Scripts** (all under `crypto-impact-study/`):

| Script | Purpose |
|---|---|
| `scripts/build_pb_blocks.py` | Paper-style continuous event blocks from the shared events |
| `scripts/run_pb_collapse.py` | The paper's P&B collapse, unchanged method |
| `scripts/run_activity_surface.py` | The activity surface and its comparisons |
| `scripts/diagnose_activity_surface.py` | Identification diagnostics |

The code lives in `src/crypto_impact/pb_collapse.py` (the paper's method,
ported from the notebook) and `src/crypto_impact/pb_surface.py` (the
extension). Tests are in `tests/test_pb_collapse.py`.

## 1. Data correction: event construction

- **The paper's input files merged every timestamp into one event.** Each took
  the first fill's sign and the gross volume of all fills. At timestamps where
  both sides traded (about 1% of timestamps), opposite-side volume was
  therefore counted with the wrong sign.
- **The text describes the right rule; the input files did not follow it.**
- **The redo uses the shared `timestamp_direction_v1` events.** Each
  contiguous same-sign run at a timestamp is its own event, so multi-price
  events are grouped correctly and both sides keep their signs.
- **This adds 2.13% more events** (and blocks at every N).

## 2. Bridge: the paper's period on corrected events (`bridge_2025_01_2026_05/`)

| | Paper | Corrected events |
|---|---:|---:|
| ξ (width), Theil–Sen | 0.955 | 0.941 |
| ψ (height) | 0.608 | 0.606 |
| β (master-curve tail) | 0.750 | 0.766 |
| α (master-curve bend) | 1.798 | 1.683 |
| H signed volume / order signs / returns | 0.629 / 0.764 / 0.507 | 0.640 / 0.764 / 0.502 |

The conclusions are unchanged. α, the least identified parameter, moves most.

## 3. The paper's P&B collapse on the extended sample (raw returns, as in the paper)

| Sample | α | β | ξ | ψ |
|---|---:|---:|---:|---:|
| 2021-01 to 2026-08 pooled | 1.49 | 0.77 | **0.83** | 0.58 |
| 2021 | 1.33 | 0.70 | 0.86 | 0.61 |
| 2022 | 1.74 | 0.64 | 0.88 | 0.62 |
| 2023 | 1.62 | 0.68 | 0.85 | 0.59 |
| 2024 | 1.83 | 0.72 | 0.94 | 0.63 |
| 2025 | 1.88 | 0.77 | 0.92 | 0.60 |

- ψ is stable at about 0.6.
- ξ varies by year (0.85–0.94), and the pooled ξ is lower than in any single
  year. Pooling across volume and volatility regimes distorts the plain P&B
  collapse.
- Hurst-style exponents (pooled): signed volume 0.64, order signs 0.79,
  returns 0.49.

## 4. The leading model: the activity-adjusted surface (`activity_surface_2021_2026/`)

**Model.** For relative imbalance z = |Q|/V, relative activity a = V/V₂₄
(block volume over the preceding 24 h of volume) and side s:

E[r/σ₂₄ | z, a, N, s] = s · R_N · F(z a^γ / Q_N), with F(x) = x/(1+|x|^α)^{β/α}.

- **γ nests the paper's two normalisations exactly:**
  - γ = 1 gives |Q|/V₂₄, the P&B coordinate;
  - γ = 0 gives |Q|/V, the imbalance ratio.
- **Response:** the block return divided by trailing 24 h realised volatility
  (the paper's main normalisation).
- **Estimation:** fitted on 20 × 10 equal-count (z, a) cells per N and side,
  with the paper's error weighting and soft-L1 loss. Free Q_N and R_N per N;
  shared α, β, γ.

**Full sample, all eight N (100–16,000).** Error-weighted RMSE is the
comparison metric:

| Model | γ | Weighted RMSE |
|---|---:|---:|
| P&B (γ = 1) | 1 | 20.65 |
| Imbalance ratio (γ = 0) | 0 | 24.11 |
| **Activity surface** | **0.362** (month-block bootstrap 95%: 0.332–0.391) | **4.72** |

**Out-of-sample** (expanding training from 2021; bins frozen from training):

| Test year | Surface | P&B | γ = 0 | Separate surface per N |
|---|---:|---:|---:|---:|
| 2023 | **4.95** | 7.96 | 16.10 | 4.49 |
| 2024 | **5.42** | 11.37 | 10.84 | 4.71 |
| 2025 | **5.64** | 11.71 | 9.24 | 5.00 |
| 2026 | **5.41** | 10.32 | 8.77 | 4.57 |

The common-γ surface beats P&B in every test year, and is within about
10–20% of fitting a separate surface per N.

**By bar size** (weighted RMSE; γ from separate per-N fits):

| N | 100 | 250 | 500 | 1,000 | 2,000 | 4,000 | 8,000 | 16,000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Surface | 9.3 | 3.7 | 3.0 | 3.7 | 4.1 | 4.0 | 3.6 | 3.1 |
| P&B | 45.4 | 27.4 | 18.2 | 12.1 | 8.2 | 5.7 | 4.1 | 3.1 |
| γ by N | 0.33 | 0.34 | 0.38 | 0.44 | 0.51 | 0.57 | 0.61 | 0.65 |

Activity matters most at short aggregation scales. By 16,000 events, P&B is
as good as the surface, and the per-N γ rises towards it.

**γ by calendar year:**

| Year | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|
| γ | 0.39 | 0.40 | 0.49 | 0.27 | 0.23 | 0.21 |

The surface beats P&B in every year, but γ has declined since 2024. This is
consistent with the alpha-side rolling-surface finding.

## 5. Caveats and open points (must be addressed in the write-up)

1. **Identification at small N.**
   - In surface coordinates the master curve is close to linear (α = 3.4,
     β = 0.11) up to a sharp bend.
   - At N = 100 no cell reaches the bend, so only R_N/Q_N is identified.
     Scaled RMSE (residual / R_N) is not meaningful there and is not used.
   - On N with at least 10% of cells beyond the bend (250–16,000), the
     surface's exponents are ξ = 0.14 and ψ = 0.58, against P&B's 0.96 and
     0.67. The activity-adjusted coordinate needs much less horizontal
     rescaling across N.
2. **The paper's "high-activity bars sit above the master curve" displacement
   is reduced but not removed.**
   - At 4,000 and 16,000 events, the high-minus-ordinary residual is 0.063
     and 0.068 under the surface, against 0.069 and 0.098 under P&B.
   - The paper's activity measure is event intensity (N/duration), while the
     surface uses volume share (V/V₂₄). The two are different dimensions of
     activity.
3. **Raw returns give γ < 0** (robustness). On raw returns, relative activity
   mostly reflects the previous day's volatility: high a means a quiet
   trailing day, hence low volatility and small raw returns. This is why the
   volatility-normalised response is primary.
4. **The early period (2019-10 to 2020-12)** gives γ = 0.64, with the surface
   again best (weighted RMSE 1.95 against P&B 4.24). Its shape parameters are
   weakly identified.
5. **Not yet redone on the extended sample:**
   - the functional-form comparison (log against power at 4,000 events) and
     monthly stability;
   - the liquidity-regime and state-dependent (threshold) effective-liquidity
     models, which the surface is meant to replace and which should be
     compared head-to-head;
   - impact decay and lead–lag;
   - figures for the thesis.
