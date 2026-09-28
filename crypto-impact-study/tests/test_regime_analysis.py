import numpy as np
import pandas as pd
import pytest

from crypto_impact.regime_analysis import (
    add_market_regimes,
    build_analysis_table,
    conditional_correlations,
    fit_pca,
    fit_regression,
    summarise_signed_returns,
)


def sample_bars(n: int = 30) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    end_price = 100 + np.arange(n, dtype=float)
    signed = np.linspace(-20, 20, n)
    return pd.DataFrame(
        {
            "end_time": index + pd.Timedelta(minutes=4),
            "duration_seconds": np.linspace(100, 500, n),
            "gross_volume": np.linspace(100, 300, n),
            "signed_imbalance": signed,
            "absolute_imbalance_ratio": np.abs(signed) / np.linspace(100, 300, n),
            "end_price": end_price,
            "log_return_bps": np.linspace(-5, 5, n),
            "high_low_range_bps": np.linspace(2, 20, n),
        },
        index=index,
    )


def test_build_analysis_table_adds_nonoverlapping_future_targets() -> None:
    bars = sample_bars()
    analysis = build_analysis_table(bars, horizons=(1, 3))

    expected = 10_000 * np.log(bars["end_price"].iloc[1] / bars["end_price"].iloc[0])
    assert analysis["future_return_1_bps"].iloc[0] == pytest.approx(expected)
    assert analysis["future_return_1_bps"].iloc[-1:].isna().all()
    assert analysis["future_return_3_bps"].iloc[-3:].isna().all()
    assert analysis["signed_imbalance_ratio"].iloc[0] == pytest.approx(-0.2)


def test_add_market_regimes_preserves_index_and_assigns_labels() -> None:
    raw = build_analysis_table(sample_bars())
    raw["flow_rate_ratio"] = np.linspace(0.5, 2.0, len(raw))
    raw["normalised_imbalance"] = np.linspace(0.001, 0.01, len(raw))
    analysis = add_market_regimes(raw)

    assert analysis.index.equals(sample_bars().index)
    assert analysis["absolute_imbalance_ratio_regime"].notna().all()
    assert set(analysis["activity_regime"].astype(str)) == {
        "high_activity",
        "medium_activity",
        "low_activity",
    }
    assert set(analysis["utc_block"].astype(str)) == {"00-08"}
    assert analysis["duration_regime"].notna().all()
    assert analysis["flow_rate_regime"].notna().all()
    assert analysis["normalised_imbalance_regime"].notna().all()


def test_fit_pca_preserves_complete_row_index_and_orthogonal_scores() -> None:
    index = pd.date_range("2025-01-01", periods=100, freq="min", tz="UTC")
    values = np.linspace(-2, 2, 100)
    features = pd.DataFrame(
        {
            "activity": values,
            "volume": values**2,
            "direction": np.sin(values),
        },
        index=index,
    )
    result = fit_pca(features)

    assert result.scores.index.equals(index)
    assert result.explained_variance.sum() == pytest.approx(1.0)
    off_diagonal = result.scores.corr().to_numpy() - np.eye(3)
    assert np.max(np.abs(off_diagonal)) < 1e-12


def test_conditional_correlations_and_signed_summary() -> None:
    analysis = add_market_regimes(build_analysis_table(sample_bars(), horizons=(1,)))
    analysis["predictor"] = analysis["future_return_1_bps"]

    correlations = conditional_correlations(
        analysis,
        regimes="activity_regime",
        predictors=["predictor"],
        targets=["future_return_1_bps"],
        minimum_observations=5,
    )
    assert len(correlations) == 3
    assert np.allclose(correlations["pearson"].dropna(), 1.0)

    summary = summarise_signed_returns(analysis, "absolute_imbalance_ratio_regime", horizon=1)
    assert summary["count"].sum() == len(analysis) - 1
    assert "standard_error" in summary


def test_fit_regression_freezes_training_standardisation_for_prediction() -> None:
    x = np.linspace(-2, 2, 100)
    activity = np.linspace(10, 30, 100)
    data = pd.DataFrame(
        {
            "return_bps": 2 + 3 * x + 0.5 * activity + 1.5 * x * activity,
            "signed_imbalance_ratio": x,
            "activity": activity,
        }
    )

    result = fit_regression(
        data,
        target="return_bps",
        predictors=["signed_imbalance_ratio", "activity"],
        standardise=["activity"],
        interactions=[("signed_imbalance_ratio", "activity")],
        hac_lags=1,
    )

    assert result.feature_means["activity"] == pytest.approx(data["activity"].mean())
    assert result.design_matrix(data)["activity"].mean() == pytest.approx(0.0)
    assert result.design_matrix(data)["activity"].std(ddof=1) == pytest.approx(1.0)
    assert np.max(np.abs(result.predict(data) - data["return_bps"])) < 1e-10

    live = pd.DataFrame({"signed_imbalance_ratio": [0.1], "activity": [100.0]})
    expected_z = (
        live["activity"].iloc[0] - result.feature_means["activity"]
    ) / result.feature_stds["activity"]
    assert result.design_matrix(live)["activity"].iloc[0] == pytest.approx(expected_z)
