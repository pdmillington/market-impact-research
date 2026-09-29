"""Transparent two-factor impact surfaces fitted to frozen bin-cell means."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SurfaceCells:
    """Training-frozen relative-imbalance by relative-activity cells."""

    imbalance_cuts: np.ndarray
    activity_cuts: np.ndarray
    mean_imbalance: np.ndarray
    mean_activity: np.ndarray
    mean_impact: np.ndarray
    standard_error: np.ndarray
    observations: np.ndarray
    imbalance_bin: np.ndarray
    activity_bin: np.ndarray


@dataclass(frozen=True)
class ImpactSurfaceFit:
    """A zero-origin, side-specific surface fitted in response space."""

    family: str
    amplitude: float
    imbalance_exponent: float
    activity_exponent: float
    imbalance_scale: float
    curvature: float
    cell_rmse: float
    cells: int

    @property
    def collapse_exponent(self) -> float:
        """Return gamma in x*=z*a**gamma for a power surface."""

        if self.family != "power" or self.imbalance_exponent <= 0:
            return math.nan
        return self.activity_exponent / self.imbalance_exponent


@dataclass(frozen=True)
class CommonShapePowerFit:
    """Power surface with common exponents and optionally side-specific levels."""

    imbalance_exponent: float
    activity_exponent: float
    buy_amplitude: float
    sell_amplitude: float
    cell_rmse: float

    @property
    def collapse_exponent(self) -> float:
        return self.activity_exponent / self.imbalance_exponent


def _finite_positive(*values: np.ndarray) -> np.ndarray:
    mask = np.ones(len(values[0]), dtype=bool)
    for value in values:
        mask &= np.isfinite(value)
    mask &= values[0] > 0
    mask &= values[1] > 0
    return mask


def surface_cells(
    imbalance: np.ndarray,
    activity: np.ndarray,
    impact: np.ndarray,
    *,
    imbalance_bins: int = 12,
    activity_bins: int = 6,
    imbalance_cuts: np.ndarray | None = None,
    activity_cuts: np.ndarray | None = None,
) -> SurfaceCells:
    """Aggregate observations on an equal-count, frozen two-dimensional grid."""

    z = np.asarray(imbalance, dtype=float)
    a = np.asarray(activity, dtype=float)
    y = np.asarray(impact, dtype=float)
    if not (z.ndim == a.ndim == y.ndim == 1 and len(z) == len(a) == len(y)):
        raise ValueError("imbalance, activity, and impact must be aligned vectors.")
    if imbalance_bins < 2 or activity_bins < 2:
        raise ValueError("Both surface dimensions require at least two bins.")
    selected = _finite_positive(z, a, y)
    z, a, y = z[selected], a[selected], y[selected]
    if len(z) < imbalance_bins * activity_bins:
        raise ValueError("Insufficient observations for the requested surface.")

    if imbalance_cuts is None:
        imbalance_cuts = np.unique(
            np.quantile(z, np.linspace(0, 1, imbalance_bins + 1)[1:-1])
        )
    else:
        imbalance_cuts = np.asarray(imbalance_cuts, dtype=float)
    if activity_cuts is None:
        activity_cuts = np.unique(
            np.quantile(a, np.linspace(0, 1, activity_bins + 1)[1:-1])
        )
    else:
        activity_cuts = np.asarray(activity_cuts, dtype=float)

    iz = np.digitize(z, imbalance_cuts, right=True)
    ia = np.digitize(a, activity_cuts, right=True)
    rows: list[tuple[float, float, float, float, int, int, int]] = []
    for activity_bin in range(len(activity_cuts) + 1):
        for imbalance_bin in range(len(imbalance_cuts) + 1):
            chosen = (iz == imbalance_bin) & (ia == activity_bin)
            count = int(chosen.sum())
            if count < 3:
                continue
            cell_y = y[chosen]
            rows.append(
                (
                    float(z[chosen].mean()),
                    float(a[chosen].mean()),
                    float(cell_y.mean()),
                    float(cell_y.std(ddof=1) / math.sqrt(count)),
                    count,
                    imbalance_bin,
                    activity_bin,
                )
            )
    if not rows:
        raise ValueError("No populated surface cells were produced.")
    columns = list(zip(*rows, strict=True))
    return SurfaceCells(
        imbalance_cuts=imbalance_cuts,
        activity_cuts=activity_cuts,
        mean_imbalance=np.asarray(columns[0], dtype=float),
        mean_activity=np.asarray(columns[1], dtype=float),
        mean_impact=np.asarray(columns[2], dtype=float),
        standard_error=np.asarray(columns[3], dtype=float),
        observations=np.asarray(columns[4], dtype=np.int64),
        imbalance_bin=np.asarray(columns[5], dtype=np.int16),
        activity_bin=np.asarray(columns[6], dtype=np.int16),
    )


def _amplitude_and_rmse(basis: np.ndarray, response: np.ndarray) -> tuple[float, float]:
    denominator = float(np.dot(basis, basis))
    amplitude = max(0.0, float(np.dot(basis, response) / denominator))
    rmse = float(np.sqrt(np.mean((response - amplitude * basis) ** 2)))
    return amplitude, rmse


def fit_power_surface(
    cells: SurfaceCells,
    *,
    imbalance_exponents: np.ndarray | None = None,
    activity_exponents: np.ndarray | None = None,
) -> ImpactSurfaceFit:
    """Fit A*z**p*a**q with equal weight for each populated cell."""

    p_grid = np.asarray(
        imbalance_exponents if imbalance_exponents is not None else np.linspace(0.05, 1.50, 146)
    )
    q_grid = np.asarray(
        activity_exponents if activity_exponents is not None else np.linspace(-0.25, 1.50, 176)
    )
    # All cells are strictly positive in z and a.  Evaluating the complete
    # exponent grid at once is much faster for monthly rolling calibration and
    # remains small: the default 146 x 176 x 72 array is about 15 MB.
    basis = np.exp(
        p_grid[:, None, None] * np.log(cells.mean_imbalance)[None, None, :]
        + q_grid[None, :, None] * np.log(cells.mean_activity)[None, None, :]
    )
    numerator = np.einsum("pqc,c->pq", basis, cells.mean_impact)
    denominator = np.einsum("pqc,pqc->pq", basis, basis)
    amplitude = np.maximum(0.0, numerator / denominator)
    error = cells.mean_impact[None, None, :] - amplitude[:, :, None] * basis
    rmse = np.sqrt(np.mean(error**2, axis=2))
    best_index = np.unravel_index(np.argmin(rmse), rmse.shape)
    best_p = float(p_grid[best_index[0]])
    best_q = float(q_grid[best_index[1]])
    best_amplitude = float(amplitude[best_index])
    best_rmse = float(rmse[best_index])
    return ImpactSurfaceFit(
        family="power",
        amplitude=best_amplitude,
        imbalance_exponent=best_p,
        activity_exponent=best_q,
        imbalance_scale=1.0,
        curvature=math.nan,
        cell_rmse=best_rmse,
        cells=len(cells.mean_impact),
    )


def fit_common_shape_power_surface(
    buy_cells: SurfaceCells,
    sell_cells: SurfaceCells,
    *,
    common_amplitude: bool = False,
    imbalance_exponents: np.ndarray | None = None,
    activity_exponents: np.ndarray | None = None,
) -> CommonShapePowerFit:
    """Fit common p and q to buy/sell cells, with one or two amplitudes."""

    p_grid = np.asarray(
        imbalance_exponents if imbalance_exponents is not None else np.linspace(0.05, 1.50, 146)
    )
    q_grid = np.asarray(
        activity_exponents if activity_exponents is not None else np.linspace(-0.25, 1.50, 176)
    )

    def grid(cells: SurfaceCells) -> np.ndarray:
        return np.exp(
            p_grid[:, None, None] * np.log(cells.mean_imbalance)[None, None, :]
            + q_grid[None, :, None] * np.log(cells.mean_activity)[None, None, :]
        )

    buy_basis = grid(buy_cells)
    sell_basis = grid(sell_cells)
    if common_amplitude:
        numerator = (
            np.einsum("pqc,c->pq", buy_basis, buy_cells.mean_impact)
            + np.einsum("pqc,c->pq", sell_basis, sell_cells.mean_impact)
        )
        denominator = (
            np.einsum("pqc,pqc->pq", buy_basis, buy_basis)
            + np.einsum("pqc,pqc->pq", sell_basis, sell_basis)
        )
        buy_amplitude = sell_amplitude = np.maximum(0.0, numerator / denominator)
    else:
        buy_amplitude = np.maximum(
            0.0,
            np.einsum("pqc,c->pq", buy_basis, buy_cells.mean_impact)
            / np.einsum("pqc,pqc->pq", buy_basis, buy_basis),
        )
        sell_amplitude = np.maximum(
            0.0,
            np.einsum("pqc,c->pq", sell_basis, sell_cells.mean_impact)
            / np.einsum("pqc,pqc->pq", sell_basis, sell_basis),
        )
    buy_error = (
        buy_cells.mean_impact[None, None, :]
        - buy_amplitude[:, :, None] * buy_basis
    )
    sell_error = (
        sell_cells.mean_impact[None, None, :]
        - sell_amplitude[:, :, None] * sell_basis
    )
    mse = (
        np.sum(buy_error**2, axis=2) + np.sum(sell_error**2, axis=2)
    ) / (len(buy_cells.mean_impact) + len(sell_cells.mean_impact))
    best = np.unravel_index(np.argmin(mse), mse.shape)
    return CommonShapePowerFit(
        imbalance_exponent=float(p_grid[best[0]]),
        activity_exponent=float(q_grid[best[1]]),
        buy_amplitude=float(buy_amplitude[best]),
        sell_amplitude=float(sell_amplitude[best]),
        cell_rmse=float(np.sqrt(mse[best])),
    )


def fit_relative_imbalance_power(cells: SurfaceCells) -> ImpactSurfaceFit:
    """Fit the restriction I=A*z**p, deliberately omitting activity."""

    return fit_power_surface(cells, activity_exponents=np.asarray([0.0]))


def fit_normalised_flow_power(cells: SurfaceCells) -> ImpactSurfaceFit:
    """Fit the restriction I=A*(z*a)**p, so both elasticities are equal."""

    best: tuple[float, float, float] | None = None
    x = cells.mean_imbalance * cells.mean_activity
    for exponent in np.linspace(0.05, 1.50, 146):
        basis = x ** float(exponent)
        amplitude, rmse = _amplitude_and_rmse(basis, cells.mean_impact)
        if best is None or rmse < best[0]:
            best = (rmse, amplitude, float(exponent))
    assert best is not None
    return ImpactSurfaceFit(
        family="normalised_flow_power",
        amplitude=best[1],
        imbalance_exponent=best[2],
        activity_exponent=best[2],
        imbalance_scale=1.0,
        curvature=math.nan,
        cell_rmse=best[0],
        cells=len(cells.mean_impact),
    )


def fit_normalised_flow_log(cells: SurfaceCells) -> ImpactSurfaceFit:
    """Fit the previous one-dimensional zero-origin log benchmark on z*a."""

    x = cells.mean_imbalance * cells.mean_activity
    scale = float(np.median(x))
    best: tuple[float, float, float] | None = None
    for curvature in np.logspace(-3, 3, 121):
        basis = np.log1p(float(curvature) * x / scale)
        amplitude, rmse = _amplitude_and_rmse(basis, cells.mean_impact)
        if best is None or rmse < best[0]:
            best = (rmse, amplitude, float(curvature))
    assert best is not None
    return ImpactSurfaceFit(
        family="normalised_flow_log",
        amplitude=best[1],
        imbalance_exponent=math.nan,
        activity_exponent=math.nan,
        imbalance_scale=scale,
        curvature=best[2],
        cell_rmse=best[0],
        cells=len(cells.mean_impact),
    )


def fit_log_activity_surface(
    cells: SurfaceCells,
    *,
    curvatures: np.ndarray | None = None,
    activity_exponents: np.ndarray | None = None,
) -> ImpactSurfaceFit:
    """Fit A*log(1+c*z/z0)*a**q with z0 fixed at the cell median."""

    curvature_grid = np.asarray(
        curvatures if curvatures is not None else np.logspace(-3, 3, 121)
    )
    q_grid = np.asarray(
        activity_exponents if activity_exponents is not None else np.linspace(-0.25, 1.50, 176)
    )
    scale = float(np.median(cells.mean_imbalance))
    best: tuple[float, float, float, float] | None = None
    for curvature in curvature_grid:
        z_basis = np.log1p(float(curvature) * cells.mean_imbalance / scale)
        for q in q_grid:
            basis = z_basis * cells.mean_activity ** float(q)
            amplitude, rmse = _amplitude_and_rmse(basis, cells.mean_impact)
            if best is None or rmse < best[0]:
                best = (rmse, amplitude, float(curvature), float(q))
    assert best is not None
    return ImpactSurfaceFit(
        family="log_activity",
        amplitude=best[1],
        imbalance_exponent=math.nan,
        activity_exponent=best[3],
        imbalance_scale=scale,
        curvature=best[2],
        cell_rmse=best[0],
        cells=len(cells.mean_impact),
    )


def predict_surface(
    fit: ImpactSurfaceFit,
    imbalance: np.ndarray,
    activity: np.ndarray,
) -> np.ndarray:
    """Apply a frozen fit without clipping observations to its training range."""

    z = np.asarray(imbalance, dtype=float)
    a = np.asarray(activity, dtype=float)
    prediction = np.full(np.broadcast(z, a).shape, np.nan)
    valid = np.isfinite(z) & np.isfinite(a) & (z >= 0) & (a > 0)
    if fit.family in {"power", "normalised_flow_power"}:
        prediction[valid] = (
            fit.amplitude
            * z[valid] ** fit.imbalance_exponent
            * a[valid] ** fit.activity_exponent
        )
    elif fit.family == "log_activity":
        prediction[valid] = (
            fit.amplitude
            * np.log1p(fit.curvature * z[valid] / fit.imbalance_scale)
            * a[valid] ** fit.activity_exponent
        )
    elif fit.family == "normalised_flow_log":
        prediction[valid] = fit.amplitude * np.log1p(
            fit.curvature * z[valid] * a[valid] / fit.imbalance_scale
        )
    else:
        raise ValueError(f"Unknown impact-surface family: {fit.family!r}.")
    return prediction
