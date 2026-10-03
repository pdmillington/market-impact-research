from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from alpha_search.perp_pnl import book_pnl  # noqa: E402


def test_long_and_short_use_simple_price_changes_not_log_returns():
    price = np.array([[100.0, 110.0, 99.0], [100.0, 110.0, 99.0]])
    target = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    rebalance = np.array([True, False, False])
    pnl = book_pnl(target, price, np.zeros_like(price), rebalance)
    # Quantity 0.01 held throughout: long +0.10 then −0.11; short the mirror image.
    assert np.allclose(pnl.price[0, :2], [0.10, -0.11])
    assert np.allclose(pnl.price[1, :2], [-0.10, 0.11])
    assert np.isclose(pnl.price[0].sum(), 99.0 / 100.0 - 1.0)


def test_notional_drifts_between_rebalances_and_trade_is_from_drifted_notional():
    price = np.array([[100.0, 120.0, 120.0]])
    target = np.array([[1.0, 0.0, 1.0]])
    rebalance = np.array([True, False, True])
    pnl = book_pnl(target, price, np.zeros_like(price), rebalance)
    assert np.isclose(pnl.notional[0, 1], 1.2)            # drifted with price
    assert np.isclose(pnl.traded[0, 0], 1.0)
    assert np.isclose(pnl.traded[0, 2], 0.2)              # 1.2 back down to 1.0
    assert np.isclose(pnl.net(10.0)[0], 0.2 - 1e-3 * 1.0)


def test_funding_paid_on_notional_at_settlement():
    price = np.array([[100.0, 200.0]])
    funding = np.array([[0.001, 0.0]])                   # settled at the end of hour 0
    pnl = book_pnl(np.array([[1.0, 0.0]]), price, funding, np.array([True, False]))
    assert np.isclose(pnl.funding[0, 0], -0.01 * 200.0 * 0.001)   # long pays on the doubled notional


def test_gap_realises_move_on_next_observed_price_and_cannot_open_before_listing():
    price = np.array([[100.0, np.nan, 110.0], [np.nan, np.nan, 50.0]])
    target = np.array([[1.0, 0, 0], [1.0, 0, 0]])
    pnl = book_pnl(target, price, np.zeros_like(price), np.array([True, False, False]))
    assert np.isclose(pnl.price[0].sum(), 0.10)
    assert np.all(pnl.notional[1] == 0) and pnl.traded[1, 0] == 0
