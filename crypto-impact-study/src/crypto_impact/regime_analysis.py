"""PCA and conditional market-regime analysis for sampled trade bars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class PCAResult:
    """Results and preprocessing parameters from a standardised PCA."""

    scores: pd.DataFrame
    weights: pd.DataFrame
    explained_variance: pd.Series
    feature_means: pd.Series
    feature_stds: pd.Series
    lower_limits: pd.Series
    upper_limits: pd.Series


@dataclass(frozen=True)
class RegressionResult:
    """A fitted regression together with its frozen feature transformation."""

    model: Any
    target: str
    predictors: tuple[str, ...]
    standardised_features: tuple[str, ...]
    interactions: tuple[tuple[str, str], ...]
    feature_means: pd.Series
    feature_stds: pd.Series

    def design_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new observations using the training-sample parameters."""

        missing = set(self.predictors).difference(data.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")
        design = data.loc[:, self.predictors].apply(pd.to_numeric, errors="coerce").copy()
        for feature in self.standardised_features:
            design[feature] = (
                design[feature] - self.feature_means[feature]
            ) / self.feature_stds[feature]
        for left, right in self.interactions:
            design[_interaction_name(left, right)] = design[left] * design[right]
        design = sm.add_constant(design, has_constant="add")
        return design.reindex(columns=self.model.model.exog_names)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict returns using the transformation fitted on the training sample."""

        prediction = self.model.predict(self.design_matrix(data))
        return pd.Series(prediction, index=data.index, name=f"predicted_{self.target}")


def _interaction_name(left: str, right: str) -> str:
    return f"{left}_x_{right}"


def build_analysis_table(
    bars: pd.DataFrame,
    horizons: Sequence[int] = (1, 2, 3, 5, 10),
) -> pd.DataFrame:
    """Add PCA features and current/future return targets to trade bars."""

    required = {
        "duration_seconds",
        "gross_volume",
        "signed_imbalance",
        "end_price",
        "log_return_bps",
    }
    missing = required.difference(bars.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if not bars.index.is_unique:
        raise ValueError("Bar index must be unique.")
    if (bars["duration_seconds"] <= 0).any():
        raise ValueError("Bar durations must be positive.")
    if (bars["gross_volume"] <= 0).any():
        raise ValueError("Gross volume must be positive.")

    horizon_values = tuple(int(value) for value in horizons)
    if any(value < 1 for value in horizon_values):
        raise ValueError("Return horizons must be positive integers.")
    if len(set(horizon_values)) != len(horizon_values):
        raise ValueError("Return horizons must be unique.")

    analysis = bars.copy()
    analysis["signed_imbalance_ratio"] = (
        analysis["signed_imbalance"] / analysis["gross_volume"]
    )
    if "absolute_imbalance_ratio" not in analysis:
        analysis["absolute_imbalance_ratio"] = analysis["signed_imbalance_ratio"].abs()
    analysis["log_duration"] = np.log(analysis["duration_seconds"])
    analysis["log_gross_volume"] = np.log(analysis["gross_volume"])
    if "high_low_range_bps" in analysis:
        if (analysis["high_low_range_bps"] <= 0).any():
            raise ValueError("High-low ranges must be positive before log transformation.")
        analysis["log_range"] = np.log(analysis["high_low_range_bps"])
    analysis["current_return_bps"] = analysis["log_return_bps"]

    for horizon in horizon_values:
        future_return = 10_000 * np.log(
            analysis["end_price"].shift(-horizon) / analysis["end_price"]
        )
        analysis[f"future_return_{horizon}_bps"] = future_return
        analysis[f"signed_future_return_{horizon}_bps"] = (
            np.sign(analysis["signed_imbalance"]) * future_return
        )
    return analysis


def add_market_regimes(analysis: pd.DataFrame) -> pd.DataFrame:
    """Add equal-count imbalance-ratio, activity, volume, and volatility regimes."""

    required = {
        "absolute_imbalance_ratio",
        "duration_seconds",
        "gross_volume",
        "high_low_range_bps",
    }
    missing = required.difference(analysis.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    result = analysis.copy()
    result["absolute_imbalance_ratio_regime"] = pd.qcut(
        result["absolute_imbalance_ratio"],
        q=5,
        labels=["very_low", "low", "medium", "high", "very_high"],
    )
    result["activity_regime"] = pd.qcut(
        result["duration_seconds"],
        q=3,
        labels=["high_activity", "medium_activity", "low_activity"],
    )
    result["duration_regime"] = pd.qcut(
        result["duration_seconds"],
        q=5,
        labels=[
            "very_short_duration",
            "short_duration",
            "medium_duration",
            "long_duration",
            "very_long_duration",
        ],
    )
    result["volume_regime"] = pd.qcut(
        result["gross_volume"],
        q=5,
        labels=["very_low_volume", "low_volume", "medium_volume", "high_volume", "very_high_volume"],
    )
    result["volatility_regime"] = pd.qcut(
        result["high_low_range_bps"],
        q=3,
        labels=["low_volatility", "medium_volatility", "high_volatility"],
    )
    if "flow_rate_ratio" in result:
        result["flow_rate_regime"] = pd.qcut(
            result["flow_rate_ratio"],
            q=5,
            labels=[
                "very_low_flow_rate",
                "low_flow_rate",
                "medium_flow_rate",
                "high_flow_rate",
                "very_high_flow_rate",
            ],
        )
    if "normalised_imbalance" in result:
        result["normalised_imbalance_regime"] = pd.qcut(
            result["normalised_imbalance"],
            q=5,
            labels=[
                "very_low_imbalance",
                "low_imbalance",
                "medium_imbalance",
                "high_imbalance",
                "very_high_imbalance",
            ],
        )
    if "end_time" in result:
        end_time = pd.to_datetime(result["end_time"], utc=True)
        result["utc_block"] = pd.cut(
            end_time.dt.hour,
            bins=[0, 8, 16, 24],
            labels=["00-08", "08-16", "16-24"],
            right=False,
            include_lowest=True,
        )
    return result


def fit_pca(
    features: pd.DataFrame,
    winsor_limits: tuple[float, float] = (0.01, 0.99),
) -> PCAResult:
    """Winsorise, standardise, and fit PCA using singular-value decomposition."""

    if features.empty:
        raise ValueError("PCA features must not be empty.")
    if not features.index.is_unique:
        raise ValueError("PCA feature index must be unique.")
    lower_quantile, upper_quantile = winsor_limits
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("winsor_limits must satisfy 0 <= lower < upper <= 1.")

    numeric = features.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if len(numeric) < 2:
        raise ValueError("At least two complete observations are required for PCA.")

    lower_limits = numeric.quantile(lower_quantile)
    upper_limits = numeric.quantile(upper_quantile)
    winsorised = numeric.clip(lower=lower_limits, upper=upper_limits, axis="columns")
    feature_means = winsorised.mean()
    feature_stds = winsorised.std(ddof=1)
    constant = feature_stds.index[feature_stds <= 0].tolist()
    if constant:
        raise ValueError(f"PCA features have zero variance: {constant}")

    standardised = (winsorised - feature_means) / feature_stds
    u, singular_values, vt = np.linalg.svd(
        standardised.to_numpy(), full_matrices=False
    )
    component_names = [f"PC{number}" for number in range(1, len(vt) + 1)]
    weights = pd.DataFrame(
        vt.T, index=standardised.columns, columns=component_names
    )
    scores = pd.DataFrame(
        u * singular_values,
        index=standardised.index,
        columns=component_names,
    )

    # PCA signs are arbitrary. Orient the largest absolute weight positively.
    for component in component_names:
        dominant_feature = weights[component].abs().idxmax()
        if weights.loc[dominant_feature, component] < 0:
            weights[component] *= -1
            scores[component] *= -1

    explained_variance = pd.Series(
        singular_values**2 / np.sum(singular_values**2),
        index=component_names,
        name="explained_variance",
    )
    return PCAResult(
        scores=scores,
        weights=weights,
        explained_variance=explained_variance,
        feature_means=feature_means,
        feature_stds=feature_stds,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )


def fit_regression(
    data: pd.DataFrame,
    target: str,
    predictors: Sequence[str],
    standardise: Sequence[str] = (),
    interactions: Sequence[tuple[str, str]] = (),
    hac_lags: int | None = 1,
) -> RegressionResult:
    """Fit an OLS model with explicit training-sample standardisation.

    The dependent variable is not standardised, so fitted values and
    coefficients remain expressed in the target's units (normally basis
    points). Means and standard deviations are retained for later prediction.
    """

    predictor_names = tuple(predictors)
    standardised_features = tuple(standardise)
    interaction_pairs = tuple(tuple(pair) for pair in interactions)
    if not predictor_names:
        raise ValueError("At least one predictor is required.")
    if len(set(predictor_names)) != len(predictor_names):
        raise ValueError("Predictor names must be unique.")
    if hac_lags is not None and hac_lags < 0:
        raise ValueError("hac_lags must be non-negative or None.")

    unknown_standardised = set(standardised_features).difference(predictor_names)
    if unknown_standardised:
        raise ValueError(
            f"Standardised features must be predictors: {sorted(unknown_standardised)}"
        )
    invalid_interactions = [
        pair
        for pair in interaction_pairs
        if len(pair) != 2 or pair[0] not in predictor_names or pair[1] not in predictor_names
    ]
    if invalid_interactions:
        raise ValueError(f"Interaction features must be predictors: {invalid_interactions}")

    required = {target, *predictor_names}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    sample = (
        data.loc[:, [target, *predictor_names]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(sample) < 3:
        raise ValueError("At least three complete observations are required.")

    feature_means = sample.loc[:, standardised_features].mean()
    feature_stds = sample.loc[:, standardised_features].std(ddof=1)
    constant = feature_stds.index[feature_stds <= 0].tolist()
    if constant:
        raise ValueError(f"Standardised features have zero variance: {constant}")

    design = sample.loc[:, predictor_names].copy()
    for feature in standardised_features:
        design[feature] = (
            design[feature] - feature_means[feature]
        ) / feature_stds[feature]
    for left, right in interaction_pairs:
        design[_interaction_name(left, right)] = design[left] * design[right]
    design = sm.add_constant(design, has_constant="add")

    estimator = sm.OLS(sample[target], design)
    if hac_lags is None:
        fitted = estimator.fit()
    else:
        fitted = estimator.fit(
            cov_type="HAC",
            cov_kwds={"maxlags": hac_lags},
        )

    return RegressionResult(
        model=fitted,
        target=target,
        predictors=predictor_names,
        standardised_features=standardised_features,
        interactions=interaction_pairs,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )


def conditional_correlations(
    data: pd.DataFrame,
    regimes: str | Sequence[str],
    predictors: Sequence[str],
    targets: Sequence[str],
    minimum_observations: int = 100,
) -> pd.DataFrame:
    """Calculate Pearson and Spearman correlations within market regimes."""

    regime_columns = [regimes] if isinstance(regimes, str) else list(regimes)
    required = set(regime_columns) | set(predictors) | set(targets)
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if minimum_observations < 2:
        raise ValueError("minimum_observations must be at least 2.")

    group_key: str | list[str]
    group_key = regime_columns[0] if len(regime_columns) == 1 else regime_columns
    rows: list[dict[str, object]] = []
    for regime_values, group in data.groupby(group_key, observed=True):
        if len(regime_columns) == 1:
            regime_values = (regime_values,)
        elif not isinstance(regime_values, tuple):
            regime_values = (regime_values,)
        regime_description = dict(zip(regime_columns, regime_values, strict=True))

        for predictor in predictors:
            for target in targets:
                sample = group[[predictor, target]].dropna()
                if len(sample) < minimum_observations:
                    continue
                rows.append(
                    {
                        **regime_description,
                        "predictor": predictor,
                        "target": target,
                        "n": len(sample),
                        "pearson": sample[predictor].corr(sample[target], method="pearson"),
                        "spearman": sample[predictor].corr(sample[target], method="spearman"),
                    }
                )
    return pd.DataFrame(rows)


def summarise_signed_returns(
    data: pd.DataFrame,
    regimes: str | Sequence[str],
    horizon: int,
) -> pd.DataFrame:
    """Summarise signed future returns within one or more market regimes."""

    regime_columns = [regimes] if isinstance(regimes, str) else list(regimes)
    target = f"signed_future_return_{horizon}_bps"
    required = set(regime_columns) | {target}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    summary = (
        data.groupby(regime_columns, observed=True)[target]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            p10=lambda values: values.quantile(0.10),
            p90=lambda values: values.quantile(0.90),
        )
        .reset_index()
    )
    summary["standard_error"] = summary["std"] / np.sqrt(summary["count"])
    return summary

def conditioned_mean_reversion(
    data: pd.DataFrame,
    conditioning_column: str,
    current_return_column: str = "current_return_bps",
    future_return_column: str = "future_return_1_bps",
    q: int = 5,
    labels=None,
) -> pd.DataFrame:
    """
    Measure next-period reversal conditional on buckets of a chosen variable.

    Positive mean_reversal_bps means the future return moved against
    the current return direction.
    """

    if labels is None:
        labels = [f"Q{i+1}" for i in range(q)]

    required_columns = list(dict.fromkeys([
        conditioning_column,
        current_return_column,
        future_return_column,
    ]))

    sample = (
        data[required_columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    sample["conditioning_bucket"] = pd.qcut(
        sample[conditioning_column],
        q=q,
        labels=labels,
        duplicates="drop",
    )

    sample["reversal_bps"] = (
        -np.sign(sample[current_return_column]) *
        sample[future_return_column]
    )

    result = (
        sample
        .groupby("conditioning_bucket", observed=True)
        .agg(
            count=("reversal_bps", "size"),
            mean_conditioning_value=(conditioning_column, "mean"),
            mean_reversal_bps=("reversal_bps", "mean"),
            median_reversal_bps=("reversal_bps", "median"),
            hit_rate=("reversal_bps", lambda x: (x > 0).mean()),
            p10=("reversal_bps", lambda x: x.quantile(0.10)),
            p25=("reversal_bps", lambda x: x.quantile(0.25)),
            p75=("reversal_bps", lambda x: x.quantile(0.75)),
            p90=("reversal_bps", lambda x: x.quantile(0.90)),
            std=("reversal_bps", "std"),
        )
        .reset_index()
    )

    result["standard_error"] = result["std"] / np.sqrt(result["count"])
    result["t_stat"] = result["mean_reversal_bps"] / result["standard_error"]

    return result
