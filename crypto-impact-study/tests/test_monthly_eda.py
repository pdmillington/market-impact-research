import numpy as np
import pandas as pd
import pytest

from crypto_impact import monthly_eda
from crypto_impact.monthly_eda import build_fixed_event_bars, make_time_bars


def _events(start: str, count: int, *, price_offset: float = 0.0) -> pd.DataFrame:
    index = pd.date_range(start, periods=count, freq="1s", tz="UTC")
    prices = 100.0 + price_offset + np.arange(count, dtype=float)
    quantities = np.ones(count, dtype=float)
    signs = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
    return pd.DataFrame(
        {
            "price": prices,
            "quantity": quantities,
            "signed_volume": signs * quantities,
        },
        index=index,
    )


def test_make_time_bars_uses_aggressor_signed_volume() -> None:
    index = pd.to_datetime(
        [
            "2025-01-01 00:00:01+00:00",
            "2025-01-01 00:01:01+00:00",
            "2025-01-01 00:05:01+00:00",
        ]
    )
    trades = pd.DataFrame(
        {
            "price": [100.0, 101.0, 99.0],
            "quantity": [2.0, 1.0, 4.0],
            "signed_volume": [2.0, -1.0, -4.0],
        },
        index=index,
    )

    bars = make_time_bars(trades, "5min")

    assert len(bars) == 2
    assert bars.iloc[0]["gross_volume"] == 3.0
    assert bars.iloc[0]["signed_imbalance"] == 1.0
    assert bars.iloc[0]["absolute_imbalance_ratio"] == pytest.approx(1 / 3)
    assert bars.iloc[0]["log_return_bps"] == pytest.approx(10_000 * np.log(1.01))
    assert bars.iloc[1]["trade_count"] == 1


def test_make_time_bars_rejects_missing_columns() -> None:
    with pytest.raises(KeyError, match="signed_volume"):
        make_time_bars(pd.DataFrame({"price": [], "quantity": []}))


def test_build_fixed_event_bars_carries_remainder_across_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events_by_name = {
        "month-1.parquet": _events("2025-01-01", 7),
        "month-2.parquet": _events(
            "2025-02-01", 8, price_offset=10.0
        ),
    }

    def fake_load(path):
        return events_by_name[path.name].copy()

    monkeypatch.setattr(monthly_eda, "load_monthly_trades", fake_load)

    bars = build_fixed_event_bars(
        list(events_by_name),
        events_per_bar=5,
    )

    assert len(bars) == 3
    assert bars["trade_count"].eq(5).all()
    assert bars["events_per_bar"].eq(5).all()
    assert bars["gross_volume"].eq(5.0).all()
    assert bars.index.is_monotonic_increasing
    assert bars.index.is_unique

    boundary_bar = bars.iloc[1]
    assert boundary_bar.name == events_by_name["month-1.parquet"].index[5]
    assert boundary_bar["end_time"] == events_by_name["month-2.parquet"].index[2]


def test_build_fixed_event_bars_discards_only_final_incomplete_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _events("2025-01-01", 6)
    monkeypatch.setattr(
        monthly_eda,
        "load_monthly_trades",
        lambda path: events.copy(),
    )

    bars = build_fixed_event_bars(
        ["month-1.parquet"],
        events_per_bar=5,
    )

    assert len(bars) == 1
    assert bars.iloc[0]["trade_count"] == 5
    assert bars.iloc[0]["end_time"] == events.index[4]


def test_build_fixed_event_bars_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="events_per_bar must be positive"):
        build_fixed_event_bars(["month-1.parquet"], events_per_bar=0)

    with pytest.raises(ValueError, match="At least one input file"):
        build_fixed_event_bars([], events_per_bar=5)


def test_build_fixed_event_bars_rejects_sample_without_complete_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _events("2025-01-01", 4)
    monkeypatch.setattr(
        monthly_eda,
        "load_monthly_trades",
        lambda path: events.copy(),
    )

    with pytest.raises(ValueError, match="No complete 5-event bars"):
        build_fixed_event_bars(
            ["month-1.parquet"],
            events_per_bar=5,
        )
