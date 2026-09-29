"""Scale-free Patzelt--Bouchaud-style impact models."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit


@dataclass(frozen=True)
class PBScalingFit:
    """A common response shape with event-count power-law scales."""

    family: str
    reference_events: int
    q_reference: float
    q_exponent_xi: float
    r_buy_reference: float
    r_sell_reference: float
    r_exponent_psi: float
    shape_alpha: float
    shape_beta: float
    central_exponent_p: float
    activity_exponent_gamma: float
    cell_rmse: float
    converged: bool
    nfev: int
    activity_vertical_exponent_delta: float = 0.0
    activity_reference: float = 1.0
    activity_scale_exponent_eta: float = 0.0
    shape_beta_activity_slope: float = 0.0

    @property
    def tail_exponent(self) -> float:
        return self.central_exponent_p - self.shape_beta


def pb_shape(
    values: np.ndarray,
    *,
    alpha: float,
    beta: float,
    central_exponent: float = 1.0,
) -> np.ndarray:
    """Positive half of the generalised P&B sigmoid."""

    x = np.asarray(values, dtype=float)
    return np.power(x, central_exponent) / np.power(
        1.0 + np.power(x, alpha), beta / alpha
    )


def predict_pb_scaling(
    fit: PBScalingFit,
    n_events: np.ndarray,
    side: np.ndarray,
    relative_imbalance: np.ndarray,
    relative_activity: np.ndarray,
) -> np.ndarray:
    """Predict side-aligned, volatility-normalised impact."""

    n = np.asarray(n_events, dtype=float)
    side_values = np.asarray(side)
    z = np.asarray(relative_imbalance, dtype=float)
    activity = np.asarray(relative_activity, dtype=float)
    ratio = n / float(fit.reference_events)
    q_scale = fit.q_reference * np.power(ratio, fit.q_exponent_xi)
    r_reference = np.where(
        side_values > 0, fit.r_buy_reference, fit.r_sell_reference
    )
    r_scale = r_reference * np.power(ratio, fit.r_exponent_psi)
    activity_scale = fit.activity_reference * np.power(
        ratio, fit.activity_scale_exponent_eta
    )
    vertical_activity = np.power(
        activity / activity_scale,
        fit.activity_vertical_exponent_delta,
    )
    beta_fraction = np.clip(
        fit.shape_beta / fit.central_exponent_p, 1e-8, 1.0 - 1e-8
    )
    beta_logit = math.log(beta_fraction / (1.0 - beta_fraction))
    local_beta = fit.central_exponent_p * expit(
        beta_logit
        + fit.shape_beta_activity_slope * np.log(activity / activity_scale)
    )
    scaled_pressure = (
        z * np.power(activity, fit.activity_exponent_gamma) / q_scale
    )
    return r_scale * vertical_activity * pb_shape(
        scaled_pressure,
        alpha=fit.shape_alpha,
        beta=local_beta,
        central_exponent=fit.central_exponent_p,
    )


def fit_pb_scaling(
    n_events: np.ndarray,
    side: np.ndarray,
    relative_imbalance: np.ndarray,
    relative_activity: np.ndarray,
    impact: np.ndarray,
    *,
    family: str,
    reference_events: int = 1_000,
    n_starts: int = 10,
    random_state: int = 12_071,
) -> PBScalingFit:
    """Fit a nested P&B model to equally weighted impact-surface cells.

    Families are ``pb_classic`` (gamma=p=1), ``pb_activity`` (free gamma,
    p=1), ``pb_activity_generalised`` (free gamma and p), and
    ``pb_activity_vertical`` (free gamma and a vertical activity exponent,
    p=1), and ``pb_activity_beta`` (free gamma and an activity-dependent
    concavity parameter, p=1). Horizontal and response scales are constrained to be powers of event
    count in every case. The vertical activity term is centred on a training
    activity scale that is itself a power of event count, preserving nesting
    and avoiding mechanical confounding with the response-scale exponent.
    """

    allowed = {
        "pb_classic",
        "pb_activity",
        "pb_activity_generalised",
        "pb_activity_vertical",
        "pb_activity_beta",
    }
    if family not in allowed:
        raise ValueError(f"Unknown family {family!r}; expected one of {sorted(allowed)}")
    arrays = [
        np.asarray(n_events, dtype=float),
        np.asarray(side, dtype=np.int8),
        np.asarray(relative_imbalance, dtype=float),
        np.asarray(relative_activity, dtype=float),
        np.asarray(impact, dtype=float),
    ]
    if any(value.ndim != 1 for value in arrays) or len({len(v) for v in arrays}) != 1:
        raise ValueError("All inputs must be aligned one-dimensional arrays.")
    n, direction, z, activity, y = arrays
    valid = (
        np.isfinite(n)
        & np.isfinite(z)
        & np.isfinite(activity)
        & np.isfinite(y)
        & (n > 0)
        & (z > 0)
        & (activity > 0)
        & (direction != 0)
    )
    n, direction, z, activity, y = (
        value[valid] for value in (n, direction, z, activity, y)
    )
    if len(y) < 20:
        raise ValueError("At least 20 valid cells are required.")

    event_counts = np.unique(n)
    median_activity = np.asarray(
        [np.median(activity[n == value]) for value in event_counts], dtype=float
    )
    log_event_ratio = np.log(event_counts / float(reference_events))
    if len(event_counts) > 1:
        activity_eta, log_activity_reference = np.polyfit(
            log_event_ratio, np.log(median_activity), 1
        )
    else:
        activity_eta = 0.0
        log_activity_reference = float(np.log(median_activity[0]))
    activity_reference = float(np.exp(log_activity_reference))
    centred_activity = activity / (
        activity_reference
        * np.power(n / float(reference_events), float(activity_eta))
    )

    near_reference = n == float(reference_events)
    seed_gamma = 1.0 if family == "pb_classic" else 0.5
    seed_p = 1.0 if family != "pb_activity_generalised" else 0.8
    seed_pressure = z * np.power(activity, seed_gamma)
    q_seed = float(np.median(seed_pressure[near_reference]))
    if not math.isfinite(q_seed) or q_seed <= 0:
        q_seed = float(np.median(seed_pressure))

    def response_seed(direction_value: int) -> float:
        chosen = near_reference & (direction == direction_value) & (y > 0)
        if not chosen.any():
            chosen = (direction == direction_value) & (y > 0)
        return max(float(np.quantile(y[chosen], 0.75)), 1e-8)

    base = [
        math.log(q_seed),
        0.95,
        math.log(response_seed(1)),
        math.log(response_seed(-1)),
        0.60,
        math.log(1.8),
        math.log(0.75 / 0.25),
    ]
    lower = [math.log(q_seed) - 8, -1.0, -20.0, -20.0, -1.0, math.log(0.10), -5.0]
    upper = [math.log(q_seed) + 8, 2.0, 5.0, 5.0, 2.0, math.log(8.0), 5.0]
    if family != "pb_classic":
        base.append(seed_gamma)
        lower.append(0.0)
        upper.append(1.5)
    if family == "pb_activity_generalised":
        base.append(seed_p)
        lower.append(0.10)
        upper.append(1.50)
    if family == "pb_activity_vertical":
        base.append(0.0)
        lower.append(-1.50)
        upper.append(1.50)
    if family == "pb_activity_beta":
        base.append(0.0)
        lower.append(-2.00)
        upper.append(2.00)
    base_array = np.asarray(base, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    response_scale = max(float(np.median(np.abs(y))), 1e-6)

    def decode(params: np.ndarray) -> tuple[float, ...]:
        gamma = 1.0 if family == "pb_classic" else float(params[7])
        next_index = 8
        p = 1.0
        if family == "pb_activity_generalised":
            p = float(params[next_index])
            next_index += 1
        delta = 0.0
        if family == "pb_activity_vertical":
            delta = float(params[next_index])
            next_index += 1
        beta_activity_slope = 0.0
        if family == "pb_activity_beta":
            beta_activity_slope = float(params[next_index])
        beta_fraction = 1.0 / (1.0 + math.exp(-float(params[6])))
        beta = p * beta_fraction
        return (
            math.exp(float(params[0])),
            float(params[1]),
            math.exp(float(params[2])),
            math.exp(float(params[3])),
            float(params[4]),
            math.exp(float(params[5])),
            beta,
            gamma,
            p,
            delta,
            beta_activity_slope,
        )

    def prediction(params: np.ndarray) -> np.ndarray:
        (
            q0,
            xi,
            r_buy,
            r_sell,
            psi,
            alpha,
            beta,
            gamma,
            p,
            delta,
            beta_activity_slope,
        ) = decode(params)
        ratio = n / float(reference_events)
        q_scale = q0 * np.power(ratio, xi)
        r0 = np.where(direction > 0, r_buy, r_sell)
        r_scale = r0 * np.power(ratio, psi)
        u = z * np.power(activity, gamma) / q_scale
        beta_fraction = np.clip(beta / p, 1e-8, 1.0 - 1e-8)
        beta_logit = math.log(beta_fraction / (1.0 - beta_fraction))
        local_beta = p * expit(
            beta_logit + beta_activity_slope * np.log(centred_activity)
        )
        return (
            r_scale
            * np.power(centred_activity, delta)
            * pb_shape(u, alpha=alpha, beta=local_beta, central_exponent=p)
        )

    def residuals(params: np.ndarray) -> np.ndarray:
        return (y - prediction(params)) / response_scale

    rng = np.random.default_rng(random_state)
    starts = [base_array]
    for _ in range(max(1, n_starts) - 1):
        draw = base_array + rng.normal(0.0, 0.18, len(base_array))
        starts.append(np.clip(draw, lower_array + 1e-8, upper_array - 1e-8))
    best = None
    for start in starts:
        result = least_squares(
            residuals,
            start,
            bounds=(lower_array, upper_array),
            loss="soft_l1",
            f_scale=0.5,
            max_nfev=30_000,
        )
        raw_prediction = prediction(result.x)
        rmse = float(np.sqrt(np.mean(np.square(y - raw_prediction))))
        objective = float(np.sum(2.0 * (np.sqrt(1.0 + result.fun**2) - 1.0)))
        candidate = (objective, rmse, result)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    _, rmse, result = best
    (
        q0,
        xi,
        r_buy,
        r_sell,
        psi,
        alpha,
        beta,
        gamma,
        p,
        delta,
        beta_activity_slope,
    ) = decode(result.x)
    return PBScalingFit(
        family=family,
        reference_events=reference_events,
        q_reference=q0,
        q_exponent_xi=xi,
        r_buy_reference=r_buy,
        r_sell_reference=r_sell,
        r_exponent_psi=psi,
        shape_alpha=alpha,
        shape_beta=beta,
        central_exponent_p=p,
        activity_exponent_gamma=gamma,
        cell_rmse=rmse,
        converged=bool(result.success),
        nfev=int(result.nfev),
        activity_vertical_exponent_delta=delta,
        activity_reference=activity_reference,
        activity_scale_exponent_eta=float(activity_eta),
        shape_beta_activity_slope=beta_activity_slope,
    )
