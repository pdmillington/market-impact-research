from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from alpha_search.simulator import Day, rest_order, run_quoting, taker_fill  # noqa: E402


def make_day(quotes, fills, start=0, end=100_000):
    """quotes: (t, bid, bid_qty, ask, ask_qty); fills: (t, price, qty, sign)."""

    q = np.array(quotes, dtype=float)
    f = np.array(fills, dtype=float) if fills else np.zeros((0, 4))
    return Day(
        symbol="TEST", start=start, end=end,
        q_t=q[:, 0], bid=q[:, 1], bid_qty=q[:, 2], ask=q[:, 3], ask_qty=q[:, 4],
        f_t=f[:, 0], f_price=f[:, 1], f_qty=f[:, 2], f_sign=f[:, 3].astype(np.int8),
        tick=0.1,
    )


def test_buy_fills_only_after_queue_ahead_is_executed():
    day = make_day(
        [(0, 100.0, 5.0, 100.1, 5.0)],
        [(500, 100.0, 3.0, -1), (600, 100.0, 1.9, -1), (700, 100.0, 0.2, -1)],
    )
    episode = rest_order(day, +1, decision_time=0, size=0.1, latency=100, stop_time=10_000)
    # 3.0 + 1.9 = 4.9 < 5.1 (queue 5 plus our 0.1); 5.1 is reached at t = 700.
    assert episode.filled and episode.reason == "fill_queue" and episode.end == 700
    assert episode.queue_ahead == 5.0


def test_trade_through_fills_immediately():
    day = make_day([(0, 100.0, 50.0, 100.1, 5.0)], [(400, 99.9, 1.0, -1)])
    episode = rest_order(day, +1, 0, 0.1, 100, 10_000)
    assert episode.filled and episode.reason == "fill_through" and episode.end == 400


def test_buyer_initiated_fills_do_not_fill_a_resting_bid():
    day = make_day([(0, 100.0, 1.0, 100.1, 5.0)], [(400, 100.1, 10.0, 1), (500, 100.0, 5.0, 1)])
    episode = rest_order(day, +1, 0, 0.1, 100, 10_000)
    assert not episode.filled


def test_order_can_fill_during_cancel_latency_after_best_moves():
    day = make_day(
        [(0, 100.0, 1.0, 100.1, 5.0), (1_000, 100.1, 2.0, 100.2, 5.0)],
        [(1_050, 100.0, 1.5, -1)],
    )
    episode = rest_order(day, +1, 0, 0.1, 100, 10_000)
    # The bid moved up at 1000; our cancel is effective at 1100, and the fill at 1050 counts.
    assert episode.filled and episode.end == 1_050


def test_moved_during_latency_gives_no_queue_position():
    day = make_day([(0, 100.0, 1.0, 100.1, 5.0), (50, 100.1, 1.0, 100.2, 5.0)], [])
    episode = rest_order(day, +1, 0, 0.1, 100, 10_000)
    assert not episode.filled and episode.reason == "moved" and episode.end == 100


def test_level_disappearing_without_trades_is_unfilled():
    day = make_day([(0, 100.0, 1.0, 100.1, 5.0), (1_000, 99.9, 3.0, 100.1, 5.0)], [])
    episode = rest_order(day, +1, 0, 0.1, 100, 10_000)
    assert not episode.filled and episode.reason == "gone" and episode.end == 1_100


def test_stop_time_cancels_before_later_fill():
    day = make_day([(0, 100.0, 1.0, 100.1, 5.0)], [(2_000, 99.9, 1.0, -1)])
    episode = rest_order(day, +1, 0, 0.1, 100, stop_time=1_500)
    assert not episode.filled and episode.reason == "stopped" and episode.end == 1_500


def test_sell_side_is_symmetric():
    day = make_day([(0, 100.0, 5.0, 100.1, 2.0)], [(300, 100.1, 2.5, 1)])
    episode = rest_order(day, -1, 0, 0.1, 100, 10_000)
    assert episode.filled and episode.price == 100.1 and episode.end == 300


def test_taker_excess_fills_one_tick_worse():
    day = make_day([(0, 100.0, 1.0, 100.1, 0.5)], [])
    price, when = taker_fill(day, +1, 0, size=1.0, latency=100)
    assert when == 100
    assert abs(price - (0.5 * 100.1 + 0.5 * 100.2)) < 1e-9


def test_quoting_respects_inventory_limit():
    quotes = [(0, 100.0, 0.0, 100.1, 1e9)]
    fills = [(t, 99.9, 1.0, -1) for t in range(1_000, 20_000, 1_000)]  # repeated trade-throughs on the bid
    day = make_day(quotes, fills, end=30_000)
    result = run_quoting(day, unit_notional=10.0, inventory_limit=3, latency=100)
    buys = result.fills.filter(result.fills["side"] == 1)
    assert buys.height == 3


def test_proportional_queue_recognises_cancellations_ahead():
    # Queue 5 at 100.0; displayed quantity drops to 0.5 through cancellations, then 1.0 executes.
    day = make_day(
        [(0, 100.0, 5.0, 100.1, 5.0), (300, 100.0, 0.5, 100.1, 5.0)],
        [(500, 100.0, 1.0, -1)],
    )
    conservative = rest_order(day, +1, 0, 0.1, 100, 10_000)
    optimistic = rest_order(day, +1, 0, 0.1, 100, 10_000, queue_model="proportional")
    assert not conservative.filled
    assert optimistic.filled and optimistic.end == 500


def test_daily_pnl_components_add_up():
    import polars as pl
    from alpha_search.simulator import daily_pnl
    day = make_day([(0, 100.0, 5.0, 100.2, 5.0), (50_000, 101.0, 5.0, 101.2, 5.0)], [], end=100_000)
    fills = pl.DataFrame({"time_ms": [10_000.0], "side": [1], "price": [100.0], "size": [1.0]})
    result = daily_pnl(day, fills, maker_fee_bps=2.0, funding=None, unit_notional=100.0)
    assert abs(result["spread_usd"] - 0.1) < 1e-9          # bought 0.1 below mid
    assert abs(result["mark_usd"] - 1.0) < 1e-9            # mid rose from 100.1 to 101.1
    assert abs(result["fees_usd"] - 0.02) < 1e-9
    assert abs(result["pnl_usd"] - (0.1 + 1.0 - 0.02)) < 1e-9


def test_e1_passive_fill_and_deadline_fallback():
    from alpha_search.simulator import execute_parent
    day = make_day([(0, 100.0, 1.0, 100.1, 1.0)], [(5_000, 99.9, 1.0, -1)], end=100_000)
    passive = execute_parent(day, +1, 0, 0.5, 10_000, 100, strategy="E1")
    assert passive.passive and passive.price == 100.0 and passive.time == 5_000
    late = execute_parent(day, +1, 0, 0.5, 2_000, 100, strategy="E1")
    assert not late.passive and late.reason == "deadline" and late.price == 100.1 and late.time == 2_100


def test_e2_runaway_crosses_and_adverse_pull_waits():
    from alpha_search.simulator import execute_parent
    day = make_day([(0, 100.0, 1.0, 100.1, 1.0)], [(5_000, 99.9, 1.0, -1)], end=100_000)
    run = execute_parent(day, +1, 0, 0.5, 10_000, 100, strategy="E2", runaway=np.array([1_000.0]),
                         adverse=np.array([]))
    assert not run.passive and run.reason == "runaway" and run.time == 1_100
    # Pulled from 4_000 (+100 ms latency) for 5 s: misses the 5_000 trade-through, then times out.
    pulled = execute_parent(day, +1, 0, 0.5, 8_000, 100, strategy="E2", runaway=np.array([]),
                            adverse=np.array([4_000.0]), pull_seconds=5)
    assert pulled.reason == "deadline"


def test_proportional_queue_shrinks_pro_rata():
    # Displayed 4 halves to 2 by cancellation: ahead 4 -> 2. Then 2.5 executes: 0.5 beyond ahead < 0.6.
    day = make_day([(0, 100.0, 4.0, 100.1, 5.0), (300, 100.0, 2.0, 100.1, 5.0)],
                   [(500, 100.0, 2.5, -1), (600, 100.0, 0.5, -1)])
    episode = rest_order(day, +1, 0, 0.6, 100, 10_000, queue_model="proportional")
    assert episode.filled and episode.end == 600
    assert not rest_order(day, +1, 0, 0.6, 100, 10_000).filled
