from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_impact.data_load import parse_bool
from crypto_impact.imbalance import compute_window_imbalance, filter_valid_windows, filter_nonzero_imbalance
from crypto_impact.trade_signing import add_signed_volume, infer_trade_sign


def test_parse_bool_handles_string_values() -> None:
    assert parse_bool("True") is True
    assert parse_bool("False") is False


def test_infer_trade_sign_uses_buyer_maker_flag() -> None:
    trades = pd.DataFrame({"buyer_maker": [True, False, True]})

    signs = infer_trade_sign(trades)

    assert signs.tolist() == [-1, 1, -1]


def test_compute_window_fields_are_hand_calculable() -> None:
    trades = pd.DataFrame(
        {
            "price": [100.0, 110.0, 120.0, 108.0],
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
    first = windows.loc[pd.Timestamp("2024-01-01 00:00:00", tz="UTC")]

    assert first["signed_imbalance"] == -1.0
    assert first["q_t"] == -1.0
    assert first["abs_signed_imbalance"] == 1.0
    assert first["gross_volume"] == 5.0
    assert first["absolute_volume"] == 5.0
    assert first["participation_ratio"] == 0.2
    assert first["start_price"] == 100.0
    assert first["end_price"] == 110.0
    assert first["n_trades"] == 2
    assert first["trade_count"] == 2
    expected_return = np.log(110.0 / 100.0)
    assert np.isclose(first["log_return"], expected_return)
    assert np.isclose(first["signed_impact"], -expected_return)
    assert np.isclose(first["abs_log_return"], expected_return)


def test_filter_valid_windows_removes_zero_imbalance_and_zero_volume() -> None:
    windows = pd.DataFrame(
        {
            "signed_imbalance": [1.0, 0.0, 2.0, 1.0],
            "gross_volume": [10.0, 10.0, 0.0, 0.0],
            "participation_ratio": [0.1, 0.0, np.inf, np.nan],
        }
    )

    filtered = filter_valid_windows(windows)

    assert filtered["signed_imbalance"].tolist() == [1.0]


def test_filter_nonzero_imbalance_supports_legacy_q_t() -> None:
    windows = pd.DataFrame({"q_t": [1.0, 0.0, -2.0]})

    filtered = filter_nonzero_imbalance(windows)

    assert filtered["q_t"].tolist() == [1.0, -2.0]
