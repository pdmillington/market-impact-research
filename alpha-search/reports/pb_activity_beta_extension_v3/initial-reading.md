# Activity-dependent concavity extension: initial reading

## Model

The nested extension keeps the activity-adjusted pressure coordinate and lets the
P&B concavity parameter vary with relative activity:

$$
\operatorname{logit}\beta(\widetilde a)
=b_0+b_1\log\widetilde a.
$$

The existing activity-adjusted P&B model is recovered at $b_1=0$. Beta remains
between zero and one, preserving a positive, monotonic response to pressure.

## Chronological result

The unrestricted extension is not a convincing replacement for the parsimonious
model.

- With full-history training, mean OOS cell RMSE falls from 0.004342 to 0.004329,
  an improvement of only about 0.3%.
- Full-history mean residual/log-activity correlation changes from -0.499 to
  -0.501, so the problem is not reduced.
- With post-2021 training, mean OOS RMSE falls from 0.004035 to 0.003970, about
  1.6%, and residual/activity correlation improves from -0.453 to -0.425.
- Fold-level gains are mixed rather than uniform.

The parameter behaviour is the larger concern. Full-history fits estimate a
negative activity--concavity slope, approximately -0.49 to -0.34 across expanding
folds, opposite to the proposed replenishment interpretation. Post-2021 fits use
compensating parameters: gamma approaches zero, alpha is approximately 0.19 and
the event-count pressure exponent becomes strongly negative. This is a mathematically
valid fit but not a stable, credible scale-free mechanism.

## Decision

Do not promote the unrestricted activity-dependent-beta model. The conditional
observed-versus-predicted activity-band curves are retained in the canonical
workbench because they show exactly where the one-dimensional model misses. A
more disciplined follow-up, if desired, would freeze or strongly regularise the
base P&B parameters and estimate only the incremental interaction slope. Residual
alpha should not be regenerated from the unrestricted extension.

