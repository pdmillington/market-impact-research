# Vertical-activity P&B extension: initial reading

## Question

Can the persistent negative correlation between cell residuals and log activity be
removed by allowing activity to alter the response level as well as the horizontal
pressure coordinate?

The nested extension is

$$
\widehat y_{N,s}=\mathcal R_{N,s}\widetilde a^\delta
F\left(\frac{z a^\gamma}{\mathcal Q_N}\right),
$$

where $\widetilde a$ is activity divided by its training-fitted normal level at
each event scale. The existing activity-adjusted model is recovered at $\delta=0$.

## Result

The extension does not solve the identified problem.

- With full-history training, mean out-of-sample cell RMSE falls from 0.004342 to
  0.004303, an improvement of approximately 0.9%.
- With post-2021 training, it falls from 0.004035 to 0.003968, approximately 1.7%.
- Full-history mean out-of-sample residual/log-activity correlation changes only
  from -0.499 to -0.496.
- Post-2021 correlation changes from -0.453 to -0.460, slightly worsening.
- The independently fitted power surfaces remain marginally better on RMSE and
  also retain a large negative activity correlation.

The added parameter mainly reallocates the activity effect. In the full-history
folds, $\gamma$ falls to roughly 0.03--0.12 while $\delta$ rises to roughly
0.43--0.51. At low pressure the response depends approximately on
$a^{\gamma+\delta}$, so the sum remains close to the original single activity
exponent while the individual mechanisms are weakly identified.

## Decision

Do not promote the vertical model. Retain `pb_activity` as the parsimonious base
impact model, but do not describe its raw error as a pure excess-impact factor.
The next falsification should determine whether residual predictability survives
explicit activity neutralisation and simple reversal/pretrend controls on identical
observations.

