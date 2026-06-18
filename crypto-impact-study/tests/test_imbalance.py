from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_impact.data_load import parse_bool
from crypto_impact.imbalance import compute_window_imbalance, filter_nonzero_imbalance
from crypto_impact.trade_signing import add_signed_volume, infer_trade_sign


def test_parse_bool_handles_string_values() -> None:
    assert parse_bool("True") is True
    assert parse_bool("False") is False


def test_infer_trade_sign_uses_buyer_maker_flag() -> None:
    trades = pd.DataFrame({"buyer_maker": [True, False, True]})

    signs = infer_trade_sign(trades)

    assert signs.tolist() == [-1, 1, -1]


def test_compute_window_imbalance_and_signed_impact() -> None:
    trades = pd.DataFrame(
        {
            "price": [100.0, 101.0, 102.0, 99.0],
            "quantity": [2.0, 3.0, 1.0, 4.0],
            "buyer_maker": [False, True, False, True],
        },
        index=pd.to_datetime(
            [
                "2024-01-01 00:00:01",
                "2024-01-01 00:00:20",
                "2024-01-01 00:01:10",
                "2024-01-01 00:01:30",
            ],
            utc=True,
        ),
    )
    signed = add_signed_volume(trades)

    windows = compute_window_imbalance(signed, "1min")

    assert windows.loc[pd.Timestamp("2024-01-01 00:00:00", tz="UTC"), "q_t"] == -1.0
    assert windows.loc[pd.Timestamp("2024-01-01 00:00:00", tz="UTC"), "absolute_volume"] == 5.0
    assert windows.loc[pd.Timestamp("2024-01-01 00:00:00", tz="UTC"), "start_price"] == 100.0
    assert windows.loc[pd.Timestamp("2024-01-01 00:00:00", tz="UTC"), "end_price"] == 101.0
    expected = -np.log(101.0 / 100.0)
    assert np.isclose(windows.loc[pd.Timestamp("2024-01-01 00:00:00", tz="UTC"), "signed_impact"], expected)


def test_filter_nonzero_imbalance() -> None:
    windows = pd.DataFrame({"q_t": [1.0, 0.0, -2.0]})

    filtered = filter_nonzero_imbalance(windows)

    assert filtered["q_t"].tolist() == [1.0, -2.0]
