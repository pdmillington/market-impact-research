from __future__ import annotations

import math
import sys
from pathlib import Path

import polars as pl


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from alpha_search.impact_model import (  # noqa: E402
    add_asymmetric_impact_features,
    fit_asymmetric_log_impact,
)


def test_asymmetric_fit_recovers_side_specific_curves() -> None:
    scale = 0.1
    x = [0.025, 0.05, 0.1, 0.2, 0.4]
    rows = []
    for side, intercept, slope in [(1, 0.2, 0.7), (-1, 0.1, 0.4)]:
        for value in x:
            rows.append(
                {
                    "normalised_imbalance": value,
                    "normalised_signed_impact": intercept
                    + slope * math.log1p(value / scale),
                    "signed_imbalance": float(side),
                    "trailing_market_volume": 1_000.0,
                    "trailing_realised_volatility_bps": 100.0,
                }
            )
    data = pl.DataFrame(rows)

    fit = fit_asymmetric_log_impact(data)
    result = add_asymmetric_impact_features(data.lazy(), fit).collect()

    assert math.isclose(fit.imbalance_scale, scale)
    assert math.isclose(fit.buy.intercept, 0.2, abs_tol=1e-12)
    assert math.isclose(fit.buy.slope, 0.7, abs_tol=1e-12)
    assert math.isclose(fit.sell.intercept, 0.1, abs_tol=1e-12)
    assert math.isclose(fit.sell.slope, 0.4, abs_tol=1e-12)
    assert result["impact_residual"].abs().max() < 1e-12
    buy_marginal = result.filter(pl.col("signed_imbalance") > 0)[
        "marginal_impact_bps_per_btc"
    ]
    assert buy_marginal[0] > buy_marginal[-1] > 0
