"""Activity-adjusted Patzelt-Bouchaud impact surface.

Extends the paper's P&B estimator (crypto_impact.pb_collapse) by one parameter.
For a block with relative imbalance z = |Q|/V, relative activity a = V/V24
(gross block volume over the preceding 24 h of volume) and side s = sign(Q):

    E[r | z, a, N, s] = s * R_N * F(z * a**gamma / Q_N),
    F(x) = x / (1 + |x|**alpha)**(beta / alpha).

gamma = 1 gives the P&B coordinate |Q|/V24 used in the paper; gamma = 0 gives the
plain imbalance ratio |Q|/V. Q_N and R_N are free for each N (as in the paper)
and the shape (alpha, beta) is shared.

With scale_law=True (revision log A3g) the scales are restricted to power laws in N
inside the fit, Q_N = Q_1 N**xi and R_N = R_1 N**psi, so the surface has 7 parameters
(alpha, beta, gamma, Q_1, xi, R_1, psi) instead of 3 + 2 per N. Internally N is centred
at its geometric mean over the fitted N; Q_1 and R_1 are reported at N = 1.

Estimation uses equal-count two-dimensional cells (z by a) for each N and side.
Cells carry monthly sufficient statistics, so samples, bootstrap resamples and
chronological folds are formed by summing months. The cell representative uses
geometric means, log x* = mean(log z) + gamma * mean(log a), which is exact in
log space for every gamma. Residuals are weighted by clipped standard errors
with a soft-L1 loss, as in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from crypto_impact.pb_collapse import pb_shape


STAT_COLUMNS = ("count", "sum_y", "sum_y2", "sum_logz", "sum_loga")


def grid_cuts(z: np.ndarray, a: np.ndarray, n_z: int, n_a: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-count marginal cut points for |z| and a (interior edges only)."""

    z_cuts = np.unique(np.quantile(z, np.linspace(0, 1, n_z + 1)[1:-1]))
    a_cuts = np.unique(np.quantile(a, np.linspace(0, 1, n_a + 1)[1:-1]))
    return z_cuts, a_cuts


def monthly_cell_stats(month: np.ndarray, side: np.ndarray, z: np.ndarray, a: np.ndarray, y: np.ndarray,
                       cuts: dict) -> pd.DataFrame:
    """Sufficient statistics by (side, z bin, a bin, month); cuts[side] = (z_cuts, a_cuts)."""

    frames = []
    for s in (-1, 1):
        chosen = (side == s) & np.isfinite(z) & np.isfinite(a) & np.isfinite(y) & (z > 0) & (a > 0)
        z_cuts, a_cuts = cuts[s]
        iz = np.digitize(z[chosen], z_cuts, right=True)
        ia = np.digitize(a[chosen], a_cuts, right=True)
        frame = pd.DataFrame({"month": month[chosen], "iz": iz, "ia": ia, "y": y[chosen],
                              "logz": np.log(z[chosen]), "loga": np.log(a[chosen])})
        frame["y2"] = frame["y"] ** 2
        stats = frame.groupby(["month", "iz", "ia"], observed=True).agg(
            count=("y", "size"), sum_y=("y", "sum"), sum_y2=("y2", "sum"),
            sum_logz=("logz", "sum"), sum_loga=("loga", "sum")).reset_index()
        stats.insert(0, "side", s)
        frames.append(stats)
    return pd.concat(frames, ignore_index=True)


def aggregate_cells(stats: pd.DataFrame, months=None, weights: pd.Series | None = None,
                    min_obs: int = 100) -> pd.DataFrame:
    """Collapse monthly statistics to cells over the chosen months.

    `weights` maps month -> multiplicity (for bootstrap resamples)."""

    data = stats if months is None else stats.loc[stats["month"].isin(months)]
    if weights is not None:
        data = data.loc[data["month"].isin(weights.index)].copy()
        factor = data["month"].map(weights).to_numpy()
        for column in STAT_COLUMNS:
            data[column] = data[column] * factor
    keys = [c for c in ("n_events", "side", "iz", "ia") if c in data.columns]
    cells = data.groupby(keys, observed=True)[list(STAT_COLUMNS)].sum().reset_index()
    cells = cells.loc[cells["count"] >= min_obs].copy()
    cells["mean_y"] = cells["sum_y"] / cells["count"]
    variance = (cells["sum_y2"] - cells["count"] * cells["mean_y"] ** 2) / (cells["count"] - 1)
    cells["standard_error"] = np.sqrt(np.clip(variance, 0, None) / cells["count"])
    cells["mean_logz"] = cells["sum_logz"] / cells["count"]
    cells["mean_loga"] = cells["sum_loga"] / cells["count"]
    return cells.loc[cells["standard_error"] > 0].reset_index(drop=True)


@dataclass
class SurfaceFit:
    alpha: float
    beta: float
    gamma: float
    scales: pd.DataFrame            # n_events, q_scale, r_scale
    objective: float
    converged: bool
    gamma_fixed: bool
    scale_law: dict | None = None   # power-law scales (A3g): q1, xi, r1, psi; None for free scales

    def predict(self, cells: pd.DataFrame) -> np.ndarray:
        lookup = self.scales.set_index("n_events")
        q = cells["n_events"].map(lookup["q_scale"]).to_numpy()
        r = cells["n_events"].map(lookup["r_scale"]).to_numpy()
        x = np.exp(cells["mean_logz"].to_numpy() + self.gamma * cells["mean_loga"].to_numpy()) / q
        return cells["side"].to_numpy() * r * pb_shape(x, self.alpha, self.beta)

    def scaled_residuals(self, cells: pd.DataFrame) -> np.ndarray:
        r = cells["n_events"].map(self.scales.set_index("n_events")["r_scale"]).to_numpy()
        return (cells["mean_y"].to_numpy() - self.predict(cells)) / r


def _clipped_sigma(cells: pd.DataFrame) -> np.ndarray:
    sigma = cells["standard_error"].to_numpy().copy()
    for n in cells["n_events"].unique():
        mask = (cells["n_events"] == n).to_numpy()
        lo, hi = np.quantile(sigma[mask], [0.10, 0.90])
        sigma[mask] = np.clip(sigma[mask], lo, hi)
    return sigma


def fit_activity_surface(cells: pd.DataFrame, *, gamma: float | None = None, n_starts: int = 8,
                         random_state: int = 1207, initial: SurfaceFit | None = None,
                         scale_law: bool = False) -> SurfaceFit:
    """Fit the shared-shape surface; gamma free when None.

    Scales are free per N by default; scale_law=True restricts them to Q_N = Q_1 N^xi and
    R_N = R_1 N^psi inside the fit (A3g). The default path is unchanged.
    """

    if scale_law:
        return _fit_scale_law(cells, gamma=gamma, n_starts=n_starts, random_state=random_state, initial=initial)

    n_values = tuple(sorted(cells["n_events"].unique()))
    index = {n: i for i, n in enumerate(n_values)}
    position = cells["n_events"].map(index).to_numpy()
    side = cells["side"].to_numpy()
    y = cells["mean_y"].to_numpy()
    logz, loga = cells["mean_logz"].to_numpy(), cells["mean_loga"].to_numpy()
    sigma = _clipped_sigma(cells)
    k = len(n_values)
    free_gamma = gamma is None
    seed_gamma = 0.5 if free_gamma else float(gamma)
    x_seed = np.exp(logz + seed_gamma * loga)
    q0 = np.array([np.median(x_seed[position == i]) for i in range(k)])
    r0 = np.array([max(np.quantile(np.abs(y[position == i]), 0.90), 1e-10) for i in range(k)])
    shape0 = np.log([1.2, 1.0])
    base = np.r_[shape0, [seed_gamma] if free_gamma else [], np.log(q0), np.log(r0)]
    lower = np.r_[np.log([0.10, 0.01]), [-0.5] if free_gamma else [], np.log(q0) - 6, np.log(r0) - 6]
    upper = np.r_[np.log([8.0, 5.0]), [1.8] if free_gamma else [], np.log(q0) + 6, np.log(r0) + 6]
    g_offset = 2 + (1 if free_gamma else 0)

    def unpack(p):
        alpha, beta = np.exp(p[:2])
        g = p[2] if free_gamma else seed_gamma
        q = np.exp(p[g_offset:g_offset + k])
        r = np.exp(p[g_offset + k:])
        return alpha, beta, g, q, r

    def residuals(p):
        alpha, beta, g, q, r = unpack(p)
        x = np.exp(logz + g * loga) / q[position]
        return (y - side * r[position] * pb_shape(x, alpha, beta)) / sigma

    starts = []
    if initial is not None:
        warm = np.r_[np.log([initial.alpha, initial.beta]), [initial.gamma] if free_gamma else [],
                     np.log(initial.scales.set_index("n_events").loc[list(n_values), "q_scale"].to_numpy()),
                     np.log(initial.scales.set_index("n_events").loc[list(n_values), "r_scale"].to_numpy())]
        starts.append(np.clip(warm, lower + 1e-8, upper - 1e-8))
    rng = np.random.default_rng(random_state)
    starts.append(np.clip(base, lower + 1e-8, upper - 1e-8))
    while len(starts) < n_starts:
        draw = base + rng.normal(0, 0.20, len(base))
        if free_gamma:
            draw[2] = rng.uniform(0.0, 1.2)
        starts.append(np.clip(draw, lower + 1e-8, upper - 1e-8))
    best = None
    for start in starts:
        result = least_squares(residuals, start, bounds=(lower, upper), loss="soft_l1", f_scale=1.0, max_nfev=20_000)
        objective = float(np.sum(2 * (np.sqrt(1 + result.fun ** 2) - 1)))
        if best is None or objective < best[0]:
            best = (objective, result)
    alpha, beta, g, q, r = unpack(best[1].x)
    scales = pd.DataFrame({"n_events": n_values, "q_scale": q, "r_scale": r})
    return SurfaceFit(float(alpha), float(beta), float(g), scales, best[0], bool(best[1].success), not free_gamma)


def weighted_rmse(fit: SurfaceFit, cells: pd.DataFrame) -> float:
    """Root mean squared standard-error-weighted residual (comparable across models on the same cells)."""

    sigma = _clipped_sigma(cells)
    return float(np.sqrt(np.mean(((cells["mean_y"].to_numpy() - fit.predict(cells)) / sigma) ** 2)))


def _power_law_fit(log_n: np.ndarray, log_values: np.ndarray) -> tuple[float, float]:
    """(intercept at the centred log N = 0, slope) of an OLS line."""

    slope, intercept = np.polyfit(log_n, log_values, 1)
    return float(intercept), float(slope)


def _fit_scale_law(cells: pd.DataFrame, *, gamma: float | None, n_starts: int, random_state: int,
                   initial: SurfaceFit | None) -> SurfaceFit:
    """Surface with Q_N = Q_1 N^xi and R_N = R_1 N^psi imposed inside the fit (A3g)."""

    n_values = tuple(sorted(cells["n_events"].unique()))
    log_n_ref = float(np.mean(np.log(n_values)))
    centred = np.log(cells["n_events"].to_numpy(float)) - log_n_ref
    side = cells["side"].to_numpy()
    y = cells["mean_y"].to_numpy()
    logz, loga = cells["mean_logz"].to_numpy(), cells["mean_loga"].to_numpy()
    sigma = _clipped_sigma(cells)
    free_gamma = gamma is None
    seed_gamma = 0.5 if free_gamma else float(gamma)
    x_seed = np.exp(logz + seed_gamma * loga)
    position = cells["n_events"].map({n: i for i, n in enumerate(n_values)}).to_numpy()
    k = len(n_values)
    q0 = np.array([np.median(x_seed[position == i]) for i in range(k)])
    r0 = np.array([max(np.quantile(np.abs(y[position == i]), 0.90), 1e-10) for i in range(k)])
    log_n_c = np.log(n_values) - log_n_ref
    lq, xi0 = _power_law_fit(log_n_c, np.log(q0)) if k > 1 else (float(np.log(q0[0])), 0.0)
    lr, psi0 = _power_law_fit(log_n_c, np.log(r0)) if k > 1 else (float(np.log(r0[0])), 0.0)
    # parameters: log alpha, log beta, [gamma], log Q at the reference N, xi, log R at the reference N, psi
    base = np.r_[np.log([1.2, 1.0]), [seed_gamma] if free_gamma else [], lq, xi0, lr, psi0]
    lower = np.r_[np.log([0.10, 0.01]), [-0.5] if free_gamma else [], lq - 6, -3.0, lr - 6, -3.0]
    upper = np.r_[np.log([8.0, 5.0]), [1.8] if free_gamma else [], lq + 6, 3.0, lr + 6, 3.0]
    g_offset = 2 + (1 if free_gamma else 0)

    def unpack(p):
        alpha, beta = np.exp(p[:2])
        g = p[2] if free_gamma else seed_gamma
        return alpha, beta, g, p[g_offset], p[g_offset + 1], p[g_offset + 2], p[g_offset + 3]

    def residuals(p):
        alpha, beta, g, log_q, xi, log_r, psi = unpack(p)
        q = np.exp(log_q + xi * centred)
        r = np.exp(log_r + psi * centred)
        return (y - side * r * pb_shape(np.exp(logz + g * loga) / q, alpha, beta)) / sigma

    starts = []
    if initial is not None:
        sc = initial.scales.set_index("n_events")
        common = [n for n in n_values if n in sc.index]
        if initial.scale_law is not None:
            law = initial.scale_law
            iq, ixi = np.log(law["q1"]) + law["xi"] * log_n_ref, law["xi"]
            ir, ipsi = np.log(law["r1"]) + law["psi"] * log_n_ref, law["psi"]
        elif len(common) > 1:
            lc = np.log(common) - log_n_ref
            iq, ixi = _power_law_fit(lc, np.log(sc.loc[common, "q_scale"].to_numpy()))
            ir, ipsi = _power_law_fit(lc, np.log(sc.loc[common, "r_scale"].to_numpy()))
        else:
            iq, ixi, ir, ipsi = lq, xi0, lr, psi0
        warm = np.r_[np.log([initial.alpha, initial.beta]), [initial.gamma] if free_gamma else [], iq, ixi, ir, ipsi]
        starts.append(np.clip(warm, lower + 1e-8, upper - 1e-8))
    rng = np.random.default_rng(random_state)
    starts.append(np.clip(base, lower + 1e-8, upper - 1e-8))
    while len(starts) < n_starts:
        draw = base + rng.normal(0, 0.20, len(base))
        if free_gamma:
            draw[2] = rng.uniform(0.0, 1.2)
        starts.append(np.clip(draw, lower + 1e-8, upper - 1e-8))
    best = None
    for start in starts:
        result = least_squares(residuals, start, bounds=(lower, upper), loss="soft_l1", f_scale=1.0, max_nfev=20_000)
        objective = float(np.sum(2 * (np.sqrt(1 + result.fun ** 2) - 1)))
        if best is None or objective < best[0]:
            best = (objective, result)
    alpha, beta, g, log_q, xi, log_r, psi = unpack(best[1].x)
    q1, r1 = float(np.exp(log_q - xi * log_n_ref)), float(np.exp(log_r - psi * log_n_ref))
    scales = pd.DataFrame({"n_events": n_values, "q_scale": q1 * np.asarray(n_values, float) ** xi,
                           "r_scale": r1 * np.asarray(n_values, float) ** psi})
    return SurfaceFit(float(alpha), float(beta), float(g), scales, best[0], bool(best[1].success), not free_gamma,
                      scale_law={"q1": q1, "xi": float(xi), "r1": r1, "psi": float(psi), "log_n_ref": log_n_ref})

