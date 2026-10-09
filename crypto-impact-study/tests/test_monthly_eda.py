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


def _write_month(path, events: pd.DataFrame) -> str:
    """Write events in the corrected monthly event-file schema (paper_events_v2)."""

    frame = pd.DataFrame({
        "price": events["price"].to_numpy(),
        "qty": events["quantity"].to_numpy(),
        "time": events.index.as_unit("ms").asi8,
        "time_str": events.index.tz_localize(None).as_unit("ms"),
        "is_buyer_maker": events["signed_volume"].to_numpy() < 0,
    })
    frame.to_parquet(path, index=False)
    return str(path)


BUILDERS = [monthly_eda.build_fixed_event_bars, monthly_eda.build_fixed_event_bars_reference]


@pytest.mark.parametrize("builder", BUILDERS)
def test_build_fixed_event_bars_carries_remainder_across_files(tmp_path, builder) -> None:
    month_1 = _events("2025-01-01", 7)
    month_2 = _events("2025-02-01", 8, price_offset=10.0)
    files = [_write_month(tmp_path / "month-1.parquet", month_1),
             _write_month(tmp_path / "month-2.parquet", month_2)]

    bars = builder(files, events_per_bar=5)

    assert len(bars) == 3
    assert bars["trade_count"].eq(5).all()
    assert bars["events_per_bar"].eq(5).all()
    assert bars["gross_volume"].eq(5.0).all()
    assert bars.index.is_monotonic_increasing
    assert bars.index.is_unique

    boundary_bar = bars.iloc[1]
    assert boundary_bar.name == month_1.index[5]
    assert boundary_bar["end_time"] == month_2.index[2]


@pytest.mark.parametrize("builder", BUILDERS)
def test_build_fixed_event_bars_discards_only_final_incomplete_bar(tmp_path, builder) -> None:
    events = _events("2025-01-01", 6)
    bars = builder([_write_month(tmp_path / "month-1.parquet", events)], events_per_bar=5)

    assert len(bars) == 1
    assert bars.iloc[0]["trade_count"] == 5
    assert bars.iloc[0]["end_time"] == events.index[4]


@pytest.mark.parametrize("builder", BUILDERS)
def test_build_fixed_event_bars_rejects_invalid_arguments(builder) -> None:
    with pytest.raises(ValueError, match="events_per_bar must be positive"):
        builder(["month-1.parquet"], events_per_bar=0)

    with pytest.raises(ValueError, match="At least one input file"):
        builder([], events_per_bar=5)


@pytest.mark.parametrize("builder", BUILDERS)
def test_build_fixed_event_bars_rejects_sample_without_complete_bar(tmp_path, builder) -> None:
    events = _events("2025-01-01", 4)
    with pytest.raises(ValueError, match="No complete"):
        builder([_write_month(tmp_path / "month-1.parquet", events)], events_per_bar=5)


def test_multi_size_builder_matches_single_size_reference(tmp_path) -> None:
    rng = np.random.default_rng(7)
    months = []
    for k, start in enumerate(["2025-01-01", "2025-02-01", "2025-03-01"]):
        count = 997 + 113 * k
        index = pd.date_range(start, periods=count, freq="250ms", tz="UTC")
        quantity = np.round(rng.uniform(0.001, 2.0, count), 3)
        sign = rng.choice([-1.0, 1.0], count)
        events = pd.DataFrame({"price": np.round(60_000 + rng.normal(0, 50, count).cumsum(), 1),
                               "quantity": quantity, "signed_volume": sign * quantity}, index=index)
        months.append(_write_month(tmp_path / f"m{k}.parquet", events))
    fast = monthly_eda.build_fixed_event_bars_multi(months, [7, 50, 333])
    for n, bars in fast.items():
        reference = monthly_eda.build_fixed_event_bars_reference(months, events_per_bar=n)
        pd.testing.assert_frame_equal(bars, reference, check_exact=False, rtol=1e-12, check_index_type=False)
        assert (bars.index.as_unit("ms").asi8 == reference.index.as_unit("ms").asi8).all()


def test_time_bar_builder_matches_make_time_bars(tmp_path) -> None:
    rng = np.random.default_rng(11)
    files, expected = [], {"1min": [], "150s": [], "5min": []}
    for k, start in enumerate(["2025-01-31 23:50", "2025-02-01 00:00"]):
        count = 3_000
        index = pd.date_range(start, periods=count, freq="137ms", tz="UTC")
        quantity = np.round(rng.uniform(0.001, 2.0, count), 3)
        sign = rng.choice([-1.0, 1.0], count)
        events = pd.DataFrame({"price": np.round(60_000 + rng.normal(0, 5, count).cumsum(), 1),
                               "quantity": quantity, "signed_volume": sign * quantity}, index=index)
        files.append(_write_month(tmp_path / f"m{k}.parquet", events))
        for window in expected:
            expected[window].append(make_time_bars(monthly_eda.load_monthly_trades(files[-1]), window))
    fast = monthly_eda.build_time_bars(files, list(expected))
    for window, parts in expected.items():
        reference = pd.concat(parts).sort_index()
        assert (fast[window].index.as_unit("ms").asi8 == reference.index.as_unit("ms").asi8).all()
        pd.testing.assert_frame_equal(fast[window].reset_index(drop=True), reference.reset_index(drop=True),
                                      check_exact=False, rtol=1e-12, check_dtype=False)


def test_bar_volumes_are_rounded_to_lot_precision():
    # Offsetting fills whose float sum leaves a ~1e-17 remainder must give an exact zero.
    index = pd.date_range("2025-01-01", periods=4, freq="s", tz="UTC")
    trades = pd.DataFrame(
        {"price": [100.0, 100.0, 100.0, 100.0], "quantity": [0.1, 0.2, 0.3, 0.6],
         "signed_volume": [0.1, 0.2, 0.3, -0.6]},
        index=index,
    )
    assert trades["signed_volume"].sum() != 0.0  # the float remainder this guards against
    bars = monthly_eda.make_trade_bars(trades, trades_per_bar=4)
    assert bars.iloc[0]["signed_imbalance"] == 0.0
    assert bars.iloc[0]["gross_volume"] == 1.2
    time_bars = make_time_bars(trades, window="1min")
    assert time_bars.iloc[0]["signed_imbalance"] == 0.0


def test_bars_spanning_a_known_data_gap_are_dropped():
    # 2023-11-21 02:43:24.004 -> 08:51:39.707 UTC is a recorded unrecoverable gap.
    times = pd.to_datetime(["2023-11-21 02:43:20", "2023-11-21 02:43:24.004",
                            "2023-11-21 08:51:39.707", "2023-11-21 08:51:40",
                            "2023-11-21 09:00:00", "2023-11-21 09:00:01"], utc=True, format="ISO8601")
    trades = pd.DataFrame({"price": 100.0, "quantity": 1.0, "signed_volume": 1.0}, index=times)
    bars = monthly_eda.make_trade_bars(trades, trades_per_bar=3)
    # first bar 02:43:20 -> 08:51:39.707 spans the gap; second bar 08:51:40 -> 09:00:01 does not
    assert bars.index.tolist() == [pd.Timestamp("2023-11-21 08:51:40", tz="UTC")]
    time_bars = make_time_bars(trades, window="5min")
    assert pd.Timestamp("2023-11-21 02:40", tz="UTC") not in time_bars.index
    assert pd.Timestamp("2023-11-21 08:50", tz="UTC") not in time_bars.index
    assert pd.Timestamp("2023-11-21 09:00", tz="UTC") in time_bars.index
