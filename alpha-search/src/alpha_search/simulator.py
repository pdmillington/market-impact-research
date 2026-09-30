"""Level-1 replay simulator for passive quoting (H-017) and execution (H-018).

The simulator replays one UTC day of time-sorted `bookTicker` quotes together
with canonical exchange fills. Our orders are assumed small enough not to change
anyone else's behaviour.

Frozen fill model (HYPOTHESIS_LEDGER.md, H-017/H-018):

- Every action (post, cancel, cross) takes effect `latency_ms` after the market
  event that triggered it; the queue is read at the effective time.
- A resting buy at price P joins behind all displayed bid quantity at P.
  Only executions at P (seller-initiated fills) reduce the queue ahead; all
  cancellations are assumed to be behind us. The order fills when the
  cumulative executed quantity at P exceeds the queue ahead plus our size,
  or when a seller-initiated fill prints strictly below P (trade-through).
- If the best bid moves away from P, we cancel; the cancel takes effect after
  the latency and the order can still fill until then. If the level P
  disappears without our fill (e.g. through cancellations), the order is
  treated as unfilled (conservative).
- Sells are symmetric.
- Taker orders fill at the touch while the level-1 quantity covers the size;
  any excess fills one tick worse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from crypto_market_data.book_ticker import canonical_month as quote_month  # noqa: E402
from crypto_market_data.layout import DataLayout  # noqa: E402


DAY_MS = 86_400_000


@dataclass
class Day:
    """Quotes and fills for one UTC day (plus a tail for markouts)."""

    symbol: str
    start: int
    end: int
    q_t: np.ndarray
    bid: np.ndarray
    bid_qty: np.ndarray
    ask: np.ndarray
    ask_qty: np.ndarray
    f_t: np.ndarray
    f_price: np.ndarray
    f_qty: np.ndarray
    f_sign: np.ndarray
    tick: float
    bid_change: np.ndarray = field(init=False)
    ask_change: np.ndarray = field(init=False)
    side_fills: dict = field(init=False)

    def __post_init__(self) -> None:
        self.bid_change = np.flatnonzero(np.diff(self.bid) != 0) + 1
        self.ask_change = np.flatnonzero(np.diff(self.ask) != 0) + 1
        # Fills that execute against resting orders on each side, precomputed once:
        # sellers hit bids (side +1), buyers lift asks (side -1).
        self.side_fills = {}
        for side, mask in ((1, self.f_sign < 0), (-1, self.f_sign > 0)):
            self.side_fills[side] = (self.f_t[mask], self.f_price[mask], self.f_qty[mask])

    def quote_index(self, t: float) -> int:
        return int(np.searchsorted(self.q_t, t, side="right")) - 1

    def mid_at(self, times: np.ndarray) -> np.ndarray:
        index = np.searchsorted(self.q_t, times, side="right") - 1
        out = np.full(len(times), np.nan)
        ok = (index >= 0) & (times <= self.q_t[-1])
        out[ok] = 0.5 * (self.bid[index[ok]] + self.ask[index[ok]])
        return out

    def next_change_time(self, index: int, side: int) -> float:
        changes = self.bid_change if side > 0 else self.ask_change
        k = np.searchsorted(changes, index + 1, side="left")
        return float(self.q_t[changes[k]]) if k < len(changes) else float("inf")


def month_of(ms: int) -> str:
    return np.datetime64(ms, "ms").astype("datetime64[M]").astype(str)


def load_day(root: Path, symbol: str, day_start: int, tail_ms: int = 120_000) -> Day | None:
    start, end = day_start, day_start + DAY_MS
    months = sorted({month_of(start), month_of(end + tail_ms - 1)})
    quotes, fills = [], []
    layout = DataLayout(Path(root))
    for month in months:
        qpath = quote_month(root, symbol, month)
        fpath = layout.canonical_month(symbol, month)
        if not qpath.exists() or not fpath.exists():
            continue
        quotes.append(
            pl.scan_parquet(qpath)
            .filter((pl.col("transaction_time") >= start - 60_000) & (pl.col("transaction_time") < end + tail_ms))
            .select("transaction_time", "update_id", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty")
            .collect()
        )
        fills.append(
            pl.scan_parquet(fpath)
            .filter((pl.col("timestamp_ms") >= start - 60_000) & (pl.col("timestamp_ms") < end + tail_ms))
            .select("timestamp_ms", "trade_id", "price", "quantity", "aggressor_sign")
            .collect()
        )
    if not quotes or not fills:
        return None
    q = pl.concat(quotes).sort("transaction_time", "update_id")
    f = pl.concat(fills).sort("timestamp_ms", "trade_id")
    if q.height < 1_000 or f.height < 100:
        return None
    bids = np.unique(q["best_bid_price"].to_numpy())
    tick = float(np.min(np.diff(bids))) if len(bids) > 1 else float("nan")
    return Day(
        symbol=symbol, start=start, end=end,
        q_t=q["transaction_time"].to_numpy().astype(np.float64),
        bid=q["best_bid_price"].to_numpy(), bid_qty=q["best_bid_qty"].to_numpy(),
        ask=q["best_ask_price"].to_numpy(), ask_qty=q["best_ask_qty"].to_numpy(),
        f_t=f["timestamp_ms"].to_numpy().astype(np.float64), f_price=f["price"].to_numpy(),
        f_qty=f["quantity"].to_numpy(), f_sign=f["aggressor_sign"].to_numpy().astype(np.int8),
        tick=tick,
    )


@dataclass
class Episode:
    """Outcome of one resting order at a fixed price."""

    side: int              # +1 buy (rests on bid), -1 sell (rests on ask)
    price: float
    posted: float          # effective time the order joined the queue
    end: float             # fill time or effective cancel time
    filled: bool
    reason: str            # fill_queue, fill_through, moved, gone, stopped, no_quote
    queue_ahead: float


def rest_order(day: Day, side: int, decision_time: float, size: float, latency: float,
               stop_time: float, queue_model: str = "conservative") -> Episode:
    """Rest at our best price from `decision_time`; run until fill, re-peg or stop.

    `stop_time` is the latest effective cancel time imposed by the policy (a
    pull trigger plus latency, a deadline, or the end of the day).
    """

    i_dec = day.quote_index(decision_time)
    if i_dec < 0:
        return Episode(side, float("nan"), decision_time, decision_time + latency, False, "no_quote", 0.0)
    price = day.bid[i_dec] if side > 0 else day.ask[i_dec]
    live = decision_time + latency
    i_live = day.quote_index(live)
    best_live = day.bid[i_live] if side > 0 else day.ask[i_live]
    if best_live != price or live >= stop_time:
        # The best moved during the latency: no queue position; decide again at `live`.
        return Episode(side, price, live, live, False, "moved", 0.0)
    queue = float(day.bid_qty[i_live] if side > 0 else day.ask_qty[i_live])
    change = day.next_change_time(i_live, side)
    new_best_index = day.quote_index(change) if np.isfinite(change) else -1
    if np.isfinite(change):
        new_best = day.bid[new_best_index] if side > 0 else day.ask[new_best_index]
        level_gone = (new_best < price) if side > 0 else (new_best > price)
    else:
        level_gone = False
    # The order can fill until the cancel after a quote change becomes effective.
    cancel_effective = min(change + latency, stop_time)
    # Opposite-side aggressors execute against us: sellers hit bids, buyers lift asks.
    times, all_prices, quantities = day.side_fills[side]
    lo = np.searchsorted(times, live, side="left")
    hi = np.searchsorted(times, cancel_effective, side="right")
    prices = all_prices[lo:hi]
    through = np.flatnonzero(prices < price) if side > 0 else np.flatnonzero(prices > price)
    executed = np.where(prices == price, quantities[lo:hi], 0.0)
    cumulative = np.cumsum(executed)
    if queue_model == "proportional":
        queue_fill_time = _proportional_queue_fill(day, side, price, queue, size, i_live, cancel_effective,
                                                   times[lo:hi], executed)
    else:
        queue_fill = np.flatnonzero(cumulative >= queue + size)
        queue_fill_time = times[lo + queue_fill[0]] if len(queue_fill) else None
    candidates = []
    if len(through):
        candidates.append((times[lo + through[0]], "fill_through"))
    if queue_fill_time is not None:
        candidates.append((queue_fill_time, "fill_queue"))
    if candidates:
        when, reason = min(candidates)
        return Episode(side, price, live, float(when), True, reason, queue)
    reason = "stopped" if cancel_effective == stop_time and stop_time < change + latency else (
        "gone" if level_gone else "moved")
    return Episode(side, price, live, float(cancel_effective), False, reason, queue)


def _proportional_queue_fill(day: Day, side: int, price: float, queue: float, size: float, i_live: int,
                             cancel_effective: float, exec_times: np.ndarray, executed: np.ndarray) -> float | None:
    """Registered optimistic queue model: displayed-quantity decreases not explained
    by executions are cancellations spread pro rata across the queue, so the
    quantity ahead of us shrinks by its share. Executions consume the queue ahead
    first; we fill once executions beyond it reach our size."""

    i_hi = int(np.searchsorted(day.q_t, cancel_effective, side="right"))
    q_idx = np.arange(i_live + 1, i_hi)
    best = day.bid if side > 0 else day.ask
    qty = day.bid_qty if side > 0 else day.ask_qty
    q_idx = q_idx[best[q_idx] == price]
    q_times, q_qty = day.q_t[q_idx], qty[q_idx]
    keep = executed > 0
    e_times, e_qty = exec_times[keep], executed[keep]
    ahead, displayed, done = queue, queue, 0.0
    i = j = 0
    while i < len(e_times) or j < len(q_times):
        if i < len(e_times) and (j >= len(q_times) or e_times[i] <= q_times[j]):
            x = e_qty[i]
            if x <= ahead:
                ahead -= x
            else:
                done += x - ahead
                ahead = 0.0
                if done >= size:
                    return float(e_times[i])
            displayed = max(displayed - x, 0.0)
            i += 1
        else:
            new = q_qty[j]
            if new < displayed and displayed > 0:
                ahead = max(ahead - (displayed - new) * ahead / displayed, 0.0)
            displayed = new
            j += 1
    return None


def taker_fill(day: Day, side: int, decision_time: float, size: float, latency: float) -> tuple[float, float]:
    """Average price and effective time of a marketable order of `size`."""

    when = decision_time + latency
    i = day.quote_index(when)
    touch = day.ask[i] if side > 0 else day.bid[i]
    available = day.ask_qty[i] if side > 0 else day.bid_qty[i]
    filled_at_touch = min(size, available)
    excess = size - filled_at_touch
    worse = touch + side * day.tick
    return float((filled_at_touch * touch + excess * worse) / size), when


# ------------------------------------------------------------------- quoting


@dataclass
class QuotingResult:
    fills: pl.DataFrame
    episodes: int
    time_quoting_share: dict


def run_quoting(day: Day, *, unit_notional: float, inventory_limit: int, latency: float,
                pulls: dict[int, np.ndarray] | None = None, pull_seconds: float = 0.0,
                queue_model: str = "conservative") -> QuotingResult:
    """Two-sided quoting (S0 when `pulls` is None; S1 with pull triggers).

    `pulls[side]` holds sorted trigger times at which the resting order on that
    side should be pulled for `pull_seconds`.
    """

    size = unit_notional / float(np.nanmedian(0.5 * (day.bid + day.ask)))
    inventory = 0
    clock = {1: float(day.start), -1: float(day.start)}
    fills, episodes = [], 0
    quoting_time = {1: 0.0, -1: 0.0}
    pulls = pulls or {1: np.array([]), -1: np.array([])}
    pause_ms = pull_seconds * 1_000.0

    while True:
        # Advance the side whose clock is earliest; skip sides blocked by inventory.
        active = [s for s in (1, -1) if clock[s] < day.end]
        if not active:
            break
        side = min(active, key=lambda s: clock[s])
        now = clock[side]
        if side * inventory >= inventory_limit:
            # Blocked on this side until the other side trades; move on with the other clock.
            clock[side] = max(now, clock[-side]) if clock[-side] < day.end else day.end
            if clock[side] == now:
                clock[side] = now + 1_000.0
            continue
        trigger_times = pulls[side]
        k = np.searchsorted(trigger_times, now, side="left")
        next_trigger = trigger_times[k] if k < len(trigger_times) else float("inf")
        stop = min(next_trigger + latency, float(day.end))
        episode = rest_order(day, side, now, size, latency, stop, queue_model)
        episodes += 1
        if episode.filled:
            inventory += side
            fills.append((episode.end, side, episode.price, size, episode.reason, episode.queue_ahead,
                          episode.end - episode.posted))
            quoting_time[side] += max(episode.end - episode.posted, 0.0)
            clock[side] = episode.end
        else:
            quoting_time[side] += max(episode.end - episode.posted, 0.0)
            if episode.reason == "stopped" and np.isfinite(next_trigger):
                clock[side] = next_trigger + pause_ms
            else:
                clock[side] = max(episode.end, now + 1.0)
    frame = pl.DataFrame(fills, schema=["time_ms", "side", "price", "size", "reason", "queue_ahead", "wait_ms"], orient="row")
    share = {s: quoting_time[s] / (day.end - day.start) for s in (1, -1)}
    return QuotingResult(frame, episodes, share)


def fill_markouts(day: Day, fills: pl.DataFrame, horizons_s=(1, 5, 60)) -> pl.DataFrame:
    """Per-fill spread capture and markouts (bps, positive = good for us)."""

    if fills.is_empty():
        return fills
    t = fills["time_ms"].to_numpy()
    side = fills["side"].to_numpy()
    price = fills["price"].to_numpy()
    mid_at_fill = day.mid_at(t)
    columns = {"spread_capture_bps": side * 1e4 * np.log(mid_at_fill / price)}
    for h in horizons_s:
        mid = day.mid_at(t + h * 1_000)
        columns[f"markout_{h}s_bps"] = side * 1e4 * np.log(mid / price)
    return fills.with_columns([pl.Series(k, v) for k, v in columns.items()])


# ------------------------------------------------------------------ triggers


def pull_times(triggers: pl.DataFrame, *, sweep_bps: float | None, min_hits: int | None,
               burst: bool) -> dict[int, np.ndarray]:
    """Registered H-017 pull triggers per quoting side (+1 bid, -1 ask).

    (a) a sweep of depth >= sweep_bps threatens the side it is moving towards:
        a buy sweep pulls the ask, a sell sweep pulls the bid;
    (b) a level run of >= min_hits single-level hits in the high absorption
        tercile threatens the side being hit;
    (c) burst deciles 7-10 pull the ask, deciles 1-4 pull the bid.
    """

    t = triggers["time_ms"].to_numpy().astype(np.float64)
    sign = triggers["sign"].to_numpy()
    masks = {1: np.zeros(len(t), bool), -1: np.zeros(len(t), bool)}
    if sweep_bps is not None:
        hit = triggers["sweep_bps"].to_numpy() >= sweep_bps
        masks[-1] |= hit & (sign > 0)
        masks[1] |= hit & (sign < 0)
    if min_hits is not None:
        run = ((triggers["levels"].to_numpy() == 1) & (triggers["hits"].to_numpy() >= min_hits)
               & (triggers["absorption_tercile"].to_numpy() == 3))
        masks[-1] |= run & (sign > 0)
        masks[1] |= run & (sign < 0)
    if burst:
        decile = triggers["burst_decile"].to_numpy()
        masks[-1] |= (decile >= 7) & (decile <= 10)
        masks[1] |= (decile >= 1) & (decile <= 4)
    return {side: t[mask] for side, mask in masks.items()}


def pulled_share(times: np.ndarray, pause_ms: float, start: float, end: float) -> float:
    """Share of [start, end) covered by the union of [t, t + pause) windows."""

    times = times[(times >= start) & (times < end)]
    if not len(times):
        return 0.0
    gaps = np.diff(times)
    covered = np.minimum(gaps, pause_ms).sum() + min(pause_ms, end - times[-1])
    return float(covered / (end - start))


# ------------------------------------------------------------------- P&L


SETTLEMENT_HOURS = (0, 8, 16)


def daily_pnl(day: Day, fills: pl.DataFrame, *, maker_fee_bps: float, funding: tuple[np.ndarray, np.ndarray] | None,
              unit_notional: float) -> dict:
    """Daily P&L of a quoting run: inventory starts flat and is marked to mid at day end.

    Components (USD): spread (fill vs mid at fill), adverse (mid move from fill
    to day end on the filled inventory), fees, funding at 08:00/16:00 settlements.
    """

    last = max(int(np.searchsorted(day.q_t, day.end, side="left")) - 1, 0)
    end_mid = 0.5 * float(day.bid[last] + day.ask[last])
    if fills.is_empty():
        return {"fills": 0, "pnl_usd": 0.0, "spread_usd": 0.0, "mark_usd": 0.0, "fees_usd": 0.0,
                "funding_usd": 0.0, "pnl_bps_of_unit": 0.0, "end_inventory_units": 0.0}
    t = fills["time_ms"].to_numpy()
    side = fills["side"].to_numpy().astype(float)
    price = fills["price"].to_numpy()
    size = fills["size"].to_numpy()
    mid_at_fill = day.mid_at(t)
    spread = float(np.nansum(side * (mid_at_fill - price) * size))
    mark = float(np.nansum(side * (end_mid - mid_at_fill) * size))
    fees = float(np.sum(price * size) * maker_fee_bps * 1e-4)
    funding_usd = 0.0
    if funding is not None:
        f_times, f_rates = funding
        for hour in SETTLEMENT_HOURS[1:]:
            settle = day.start + hour * 3_600_000
            k = np.searchsorted(f_times, settle, side="left")
            if k < len(f_times) and abs(f_times[k] - settle) < 60_000:
                inventory = float(np.sum((side * size)[t <= settle]))
                q = int(np.searchsorted(day.q_t, settle, side="right")) - 1
                if inventory == 0.0 or q < 0:
                    continue
                mid = 0.5 * float(day.bid[q] + day.ask[q])
                funding_usd -= inventory * mid * f_rates[k]
    pnl = spread + mark - fees + funding_usd
    return {"fills": int(len(t)), "pnl_usd": pnl, "spread_usd": spread, "mark_usd": mark, "fees_usd": fees,
            "funding_usd": funding_usd, "pnl_bps_of_unit": pnl / unit_notional * 1e4,
            "end_inventory_units": float(np.sum(side * size))}


# ----------------------------------------------------------------- execution


@dataclass
class Execution:
    price: float
    time: float
    passive: bool
    reason: str           # passive, deadline, runaway
    excess_share: float   # taker quantity beyond level-1 (filled one tick worse)


def execution_triggers(triggers: pl.DataFrame, direction: int, tick: float, *, sweep_ticks: float,
                       min_hits: int) -> tuple[np.ndarray, np.ndarray]:
    """H-018 (amended) trigger times for a parent order in `direction`.

    Run-away (cross now): a sweep of >= sweep_ticks in our direction; a level run
    of >= min_hits single-level hits on the opposite best (aggressors on our side);
    a burst in decile 10 when buying or 1 when selling.
    Adverse (pull): a sweep of >= sweep_ticks against us.
    """

    t = triggers["time_ms"].to_numpy().astype(np.float64)
    sign = triggers["sign"].to_numpy()
    ticks = triggers["sweep_bps"].to_numpy() * 1e-4 * triggers["price"].to_numpy() / tick
    sweep = ticks >= sweep_ticks - 1e-6
    run = (triggers["levels"].to_numpy() == 1) & (triggers["hits"].to_numpy() >= min_hits) & (sign == direction)
    burst = triggers["burst_decile"].to_numpy() == (10 if direction > 0 else 1)
    runaway = (sweep & (sign == direction)) | run | burst
    adverse = sweep & (sign == -direction)
    return t[runaway], t[adverse]


def _cross(day: Day, direction: int, decision: float, size: float, latency: float, reason: str) -> Execution:
    price, when = taker_fill(day, direction, decision, size, latency)
    i = day.quote_index(when)
    available = day.ask_qty[i] if direction > 0 else day.bid_qty[i]
    return Execution(price, when, False, reason, max(size - available, 0.0) / size)


def execute_parent(day: Day, direction: int, arrival: float, size: float, window_ms: float, latency: float, *,
                   strategy: str, runaway: np.ndarray | None = None, adverse: np.ndarray | None = None,
                   pull_seconds: float = 0.0, queue_model: str = "conservative") -> Execution:
    """E0 immediate taker; E1 passive re-pegged with a deadline taker fallback;
    E2 = E1 plus run-away crossing and adverse pulls. Fills are all-or-nothing:
    a passive fill needs executions at our price beyond queue + full size, or a
    trade-through."""

    if strategy == "E0":
        return _cross(day, direction, arrival, size, latency, "immediate")
    deadline = arrival + window_ms
    empty = np.array([])
    runaway = runaway if strategy == "E2" and runaway is not None else empty
    adverse = adverse if strategy == "E2" and adverse is not None else empty
    now = arrival
    while now < deadline:
        k = np.searchsorted(runaway, now, side="left")
        cross_at = runaway[k] if k < len(runaway) else float("inf")
        if cross_at < deadline and cross_at <= now:
            return _cross(day, direction, now, size, latency, "runaway")
        j = np.searchsorted(adverse, now, side="left")
        pull_at = adverse[j] if j < len(adverse) else float("inf")
        stop = min(deadline, cross_at + latency, pull_at + latency)
        episode = rest_order(day, direction, now, size, latency, stop, queue_model)
        if episode.filled:
            return Execution(episode.price, episode.end, True, "passive", 0.0)
        if episode.reason == "stopped":
            if stop == deadline:
                break
            if stop == cross_at + latency:
                return _cross(day, direction, cross_at, size, latency, "runaway")
            now = pull_at + pull_seconds * 1_000.0
            continue
        if episode.reason == "no_quote":
            break
        now = max(episode.end, now + 1.0)
    return _cross(day, direction, max(deadline, now), size, latency, "deadline")


def shortfall(day: Day, direction: int, arrival: float, execution: Execution, *, maker_fee_bps: float,
              taker_fee_bps: float) -> dict:
    """Implementation shortfall vs the arrival quote mid (bps, positive = cost)."""

    i0 = day.quote_index(arrival)
    mid0 = 0.5 * float(day.bid[i0] + day.ask[i0])
    i1 = day.quote_index(execution.time)
    mid1 = 0.5 * float(day.bid[i1] + day.ask[i1])
    fee = maker_fee_bps if execution.passive else taker_fee_bps
    spread = direction * (execution.price - mid1) / mid0 * 1e4
    drift = direction * (mid1 - mid0) / mid0 * 1e4
    return {"is_bps": spread + drift + fee, "spread_bps": spread, "drift_bps": drift, "fee_bps": fee,
            "passive": execution.passive, "reason": execution.reason, "time_to_fill_s": (execution.time - arrival) / 1e3,
            "excess_share": execution.excess_share}
