"""P&L of a book of linear (USDT-margined) perpetual positions on an hourly grid.

A linear perp position is a fixed quantity q: its P&L over an hour is
q × (P₁ − P₀), i.e. notional × simple return, not a log return. Between
rebalances the quantity is held, so notional drifts with price. Funding is paid
on the notional at settlement (q × P at the end of the hour in which the
settlement falls; long pays a positive rate). Trading costs are charged on the
notional traded at a rebalance: |target − drifted notional|.

Missing prices are forward-filled for marking, so a gap earns nothing until the
next observed price and the move across the gap is then realised. A position
cannot be opened where no price has been observed yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BookPnl:
    price: np.ndarray        # [symbols, hours] q × (P_{h+1} − P_h)
    funding: np.ndarray      # [symbols, hours] −q × P_{h+1} × rate settled in (T_h, T_{h+1}]
    traded: np.ndarray       # [symbols, hours] |notional traded| at the start of hour h
    notional: np.ndarray     # [symbols, hours] signed notional held over hour h (after any trade)

    def net(self, cost_bps: float) -> np.ndarray:
        """Portfolio P&L per hour after costs."""

        return (self.price + self.funding).sum(axis=0) - cost_bps * 1e-4 * self.traded.sum(axis=0)

    def leg(self, component: str, side: int) -> np.ndarray:
        """Portfolio P&L per hour of one component ('price' or 'funding') for longs (+1) or shorts (−1)."""

        values = getattr(self, component)
        mask = self.notional > 0 if side > 0 else self.notional < 0
        return np.where(mask, values, 0.0).sum(axis=0)


def forward_fill(values: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along the last axis."""

    index = np.where(np.isfinite(values), np.arange(values.shape[-1]), 0)
    np.maximum.accumulate(index, axis=-1, out=index)
    filled = np.take_along_axis(values, index, axis=-1)
    return filled


def book_pnl(target: np.ndarray, price: np.ndarray, funding: np.ndarray, rebalance: np.ndarray) -> BookPnl:
    """Simulate a book that trades to `target` notional at every hour where `rebalance` is true.

    target:   [symbols, hours] desired signed notional (read only at rebalance hours)
    price:    [symbols, hours] price at the decision time T_h (NaN if unobserved)
    funding:  [symbols, hours] summed funding rate settled in (T_h, T_{h+1}]
    rebalance:[hours] bool
    """

    n_sym, n_hours = price.shape
    marked = forward_fill(price)
    observed = np.isfinite(marked)
    marked_next = np.concatenate([marked[:, 1:], marked[:, -1:]], axis=1)
    quantity = np.zeros(n_sym)
    out_price = np.zeros((n_sym, n_hours))
    out_funding = np.zeros((n_sym, n_hours))
    traded = np.zeros((n_sym, n_hours))
    notional = np.zeros((n_sym, n_hours))
    for h in range(n_hours):
        p = marked[:, h]
        if rebalance[h]:
            drifted = np.where(observed[:, h], quantity * p, 0.0)
            wanted = np.where(observed[:, h], np.nan_to_num(target[:, h]), 0.0)
            traded[:, h] = np.abs(wanted - drifted)
            quantity = np.where(observed[:, h], wanted / np.where(observed[:, h], p, 1.0), 0.0)
        held = np.where(observed[:, h], quantity * p, 0.0)
        notional[:, h] = held
        step = np.where(np.isfinite(marked_next[:, h]) & observed[:, h], marked_next[:, h] - p, 0.0)
        out_price[:, h] = quantity * step
        out_funding[:, h] = -quantity * np.where(np.isfinite(marked_next[:, h]), marked_next[:, h], 0.0) * np.nan_to_num(funding[:, h])
    return BookPnl(out_price, out_funding, traded, notional)
