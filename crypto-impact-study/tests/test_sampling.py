from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_impact.sampling import aggregate_time_bars, aggregate_volume_bars
from crypto_impact.trade_signing import add_signed_volume


def _trades() -> pd.DataFrame:
    trades = pd.DataFrame(
        {
            "price": [100.0, 101.0, 102.0, 103.0],
            "quantity": [2.0, 3.0, 4.0, 5.0],
            "buyer_maker": [False, True, False, True],
        },
        index=pd.to_datetime(
            ["2024-01-01 00:00:00", "2024-01-01 00:01:00", "2024-01-01 00:02:00", "2024-01-01 00:03:00"],
            utc=True,
        ),
    )
    return add_signed_volume(trades)


def test_rolling_time_bars_use_window_length_and_step() -> None:
    bars = aggregate_time_bars(_trades(), window_length="2min", window_step="1min")

    assert len(bars) >= 3
    first = bars.iloc[0]
    assert first["signed_imbalance"] == -1.0
    assert first["gross_volume"] == 5.0
    assert first["participation_ratio"] == 0.2
    assert np.isclose(first["log_return"], np.log(101.0 / 100.0))


def test_volume_bars_close_when_threshold_is_reached() -> None:
    bars = aggregate_volume_bars(_trades(), volume_bar_size=5.0)

    assert len(bars) == 2
    assert bars.iloc[0]["gross_volume"] == 5.0
    assert bars.iloc[0]["signed_imbalance"] == -1.0
    assert bars.iloc[1]["gross_volume"] == 9.0
    assert bars.iloc[1]["signed_imbalance"] == -1.0
