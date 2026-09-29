# Residual/activity diagnostics: initial reading

These diagnostics use the parsimonious `pb_activity` model. All pressure and
activity boundaries are estimated from the corresponding training sample and then
frozen for the following test fold. June--August 2026 remain excluded.

## What the graphs show

- The strong negative relationship in equal-weight surface cells is real. In the
  default 2025 test fold, cell-level scaled-residual/activity correlations range
  from -0.82 around 1,000 events to -0.35 at 8,000 events.
- The relationship survives at individual-bar level and is broadly monotonic. In
  the same fold, the mean scaled residual falls from the lowest to highest activity
  decile by approximately 1.48 units at 200 events, 1.05 at 1,000 events and 0.29
  at 8,000 events.
- It is not merely a marginal relationship caused by pressure and activity moving
  together. At 1,000 events in the 2025 fold, the highest-minus-lowest activity
  residual difference becomes progressively more negative from -0.14 in the lowest
  pressure quintile to -1.38 in the highest pressure quintile.
- Pressure-bin control reduces the monthly correlation at medium and large event
  scales, but does not remove it. For 1,000 events, mean monthly correlation is
  -0.140 marginally and -0.110 after pressure-bin control in the 2025 fold.
- The problem is time-dependent. At 1,000 events the mean monthly marginal
  correlation is only -0.021 in the 2023 test fold, then roughly -0.12 to -0.14 in
  the later folds.

## Implication

A single one-dimensional activity correction is unlikely to be adequate. The
pattern is an interaction between pressure, activity, event scale and calendar
regime. Before neutralising, compare a small training-frozen pressure-by-activity
correction grid with a simpler linear correction. Any proposed correction must be
judged by out-of-period decorrelation and by whether residual-return predictability
survives on identical observations.

