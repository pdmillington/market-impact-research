"""Small, interpretable impact models for alpha feature construction."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .bars import marginal_impact_proxy


@dataclass(frozen=True)
class SideImpactFit:
    """One side of a logarithmic normalized-impact curve."""

    side: int
    observations: int
    intercept: float
    slope: float
    r_squared: float


@dataclass(frozen=True)
class AsymmetricImpactFit:
    """Training-fitted buy and sell curves with a common x scale."""

    imbalance_scale: float
    buy: SideImpactFit
    sell: SideImpactFit


def _fit_side(data: pl.DataFrame, side: int) -> SideImpactFit:
    sample = (
        data.filter(pl.col("_side") == side)
        .with_columns(
            (pl.col("_x") / pl.first("_scale") + 1.0).log().alias("_z")
        )
    )
    if sample.height < 3:
        raise ValueError(f"At least three observations are required for side {side}.")
    moments = sample.select(
        pl.len().alias("n"),
        pl.col("_z").mean().alias("mean_z"),
        pl.col("_y").mean().alias("mean_y"),
        pl.col("_z").var().alias("var_z"),
        pl.cov("_z", "_y").alias("cov_zy"),
    ).row(0, named=True)
    if moments["var_z"] is None or moments["var_z"] <= 0:
        raise ValueError(f"Impact predictor has no variation for side {side}.")
    slope = float(moments["cov_zy"] / moments["var_z"])
    intercept = float(moments["mean_y"] - slope * moments["mean_z"])
    fit_stats = sample.select(
        (
            (
                pl.col("_y")
                - (pl.lit(intercept) + pl.lit(slope) * pl.col("_z"))
            ).pow(2)
        ).sum().alias("sse"),
        ((pl.col("_y") - pl.col("_y").mean()).pow(2)).sum().alias("sst"),
    ).row(0, named=True)
    r_squared = 1.0 - float(fit_stats["sse"] / fit_stats["sst"])
    return SideImpactFit(
        side=side,
        observations=int(moments["n"]),
        intercept=intercept,
        slope=slope,
        r_squared=r_squared,
    )


def fit_asymmetric_log_impact(
    training: pl.LazyFrame | pl.DataFrame,
    *,
    imbalance_col: str = "normalised_imbalance",
    impact_col: str = "normalised_signed_impact",
    side_col: str = "signed_imbalance",
) -> AsymmetricImpactFit:
    """Fit separate buy/sell logarithmic impact curves on training data."""

    lazy = training.lazy() if isinstance(training, pl.DataFrame) else training
    required = {imbalance_col, impact_col, side_col}
    missing = required.difference(lazy.collect_schema().names())
    if missing:
        raise KeyError(f"Missing impact-model columns: {sorted(missing)}")
    data = (
        lazy.select(
            pl.col(imbalance_col).cast(pl.Float64).alias("_x"),
            pl.col(impact_col).cast(pl.Float64).alias("_y"),
            pl.col(side_col).sign().cast(pl.Int8).alias("_side"),
        )
        .filter(
            pl.col("_x").is_finite()
            & pl.col("_y").is_finite()
            & (pl.col("_x") > 0)
            & (pl.col("_side") != 0)
        )
        .collect()
    )
    scale = data["_x"].median()
    if scale is None or scale <= 0:
        raise ValueError("Training normalized imbalance must have a positive median.")
    data = data.with_columns(pl.lit(float(scale)).alias("_scale"))
    return AsymmetricImpactFit(
        imbalance_scale=float(scale),
        buy=_fit_side(data, 1),
        sell=_fit_side(data, -1),
    )


def add_asymmetric_impact_features(
    observations: pl.LazyFrame,
    fit: AsymmetricImpactFit,
) -> pl.LazyFrame:
    """Apply a frozen training fit and add expectation, residual, and derivative."""

    side = pl.col("signed_imbalance").sign()
    log_x = (
        pl.col("normalised_imbalance") / pl.lit(fit.imbalance_scale) + 1.0
    ).log()
    expected = (
        pl.when(side > 0)
        .then(pl.lit(fit.buy.intercept) + pl.lit(fit.buy.slope) * log_x)
        .when(side < 0)
        .then(pl.lit(fit.sell.intercept) + pl.lit(fit.sell.slope) * log_x)
        .otherwise(None)
    )
    marginal = (
        pl.when(side > 0)
        .then(
            marginal_impact_proxy(
                normalised_imbalance=pl.col("normalised_imbalance"),
                trailing_volume=pl.col("trailing_market_volume"),
                trailing_volatility_bps=pl.col("trailing_realised_volatility_bps"),
                log_slope=fit.buy.slope,
                imbalance_scale=fit.imbalance_scale,
            )
        )
        .when(side < 0)
        .then(
            marginal_impact_proxy(
                normalised_imbalance=pl.col("normalised_imbalance"),
                trailing_volume=pl.col("trailing_market_volume"),
                trailing_volatility_bps=pl.col("trailing_realised_volatility_bps"),
                log_slope=fit.sell.slope,
                imbalance_scale=fit.imbalance_scale,
            )
        )
        .otherwise(None)
    )
    return observations.with_columns(
        expected.alias("expected_normalised_impact"),
        marginal.alias("marginal_impact_bps_per_btc"),
    ).with_columns(
        (
            pl.col("normalised_signed_impact")
            - pl.col("expected_normalised_impact")
        ).alias("impact_residual")
    )
