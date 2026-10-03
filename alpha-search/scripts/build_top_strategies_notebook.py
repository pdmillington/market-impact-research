"""Build the top-strategies workbench (C-006, C-007, H-018) from saved outputs."""

from pathlib import Path

import nbformat as nbf


REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "alpha-search" / "notebooks" / "TopStrategiesWorkbench.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(r"""
# Top strategies workbench

A hands-on view of the three results that survived testing (October 2026):

1. **C-006: 5-minute pretrend reversal.** After the sharpest 1% of short-term
   moves, trade the predicted snap-back for 5 minutes. Confirmed on ETH and SOL
   (BTC was the discovery market).
2. **C-007: stress reversion.** After rare extreme moves, fade them for
   5 minutes. Passes on BTC and ETH, fails on SOL.
3. **H-018: patient passive execution.** Resting a limit order at the best
   price instead of crossing saves about 2 bps per side. This is a cost saver,
   not an alpha.

Everything here is computed from saved outputs, so nothing is re-run. Change the
**controls** cell and re-run the notebook from there. The registered (frozen)
choices are the defaults. Exploring other settings is fine for intuition,
but it is *exploration*: any setting picked here would need a new registration
and fresh data before it counts as evidence.

Plain-English summary: `alpha-search/reports/STRATEGY_SUMMARY.md`.
"""),
    code(r"""
from pathlib import Path
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/alpha-search-matplotlib')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'alpha-search').exists())
REPORTS = REPO / 'alpha-search' / 'reports'
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(200); pl.Config.set_tbl_cols(20)
plt.rcParams.update({'figure.figsize': (11, 4), 'axes.grid': True, 'grid.alpha': 0.3})
"""),
    md(r"""
## Controls

| Control | Meaning | Registered value |
|---|---|---|
| `SYMBOLS` | instruments to show | BTC (reference), ETH, SOL |
| `N_EVENTS` | event-bar size used to build the signal (200 to 8,000 trades per bar) | 1000 |
| `MODEL` | `return_pretrend`, `pretrend_only`, `bar_return`, `return_pretrend_flow` | `return_pretrend` |
| `HORIZON` | holding time in minutes (5 or 10) | 5 |
| `TAIL` | share of signals traded (0.01 = the most extreme 1%) | 0.01 |
| `FEE_PER_SIDE_BPS` | your fee per side; the researcher's estimate is 2 taker, 1 maker | 2 |
| `TRADE_SIZE_USD` | size per trade, for the slippage estimate | 100,000 |
| `DROP_BEST_DAYS` | days removed in the "ordinary days" view | 10 |
"""),
    code(r"""
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
N_EVENTS = 1000
MODEL = 'return_pretrend'
HORIZON = 5
TAIL = 0.01
FEE_PER_SIDE_BPS = 2.0
TRADE_SIZE_USD = 100_000
DROP_BEST_DAYS = 10
"""),
    code(r"""
TRADES = REPORTS / 'c006' / 'trades.parquet'

def load_trades(symbol, n_events=N_EVENTS, model=MODEL, horizon=HORIZON, tail=TAIL):
    return (pl.scan_parquet(TRADES)
            .filter((pl.col('symbol') == symbol) & (pl.col('n_events') == n_events) & (pl.col('model') == model)
                    & (pl.col('horizon') == horizon) & (pl.col('tail') == tail))
            .with_columns(pl.from_epoch('time_ms', time_unit='ms').alias('time'))
            .collect())

def summarise(t):
    daily = t.group_by('day').agg(pl.col('edge_simple_bps').sum().alias('s'), pl.len().alias('n')).sort('s', descending=True)
    per_day = (daily['s'] / daily['n']).to_numpy()
    ordinary = daily.tail(max(daily.height - DROP_BEST_DAYS, 0))
    gross = t['edge_simple_bps'].mean()
    return {
        'trades': t.height, 'trades_per_day': t.height / max(daily.height, 1),
        'gross_bps': gross, 'net_bps': gross - 2 * FEE_PER_SIDE_BPS,
        'ordinary_gross_bps': ordinary['s'].sum() / max(ordinary['n'].sum(), 1),
        't_stat': per_day.mean() / (per_day.std(ddof=1) / np.sqrt(len(per_day))),
        'best_days_share': daily.head(DROP_BEST_DAYS)['s'].sum() / daily['s'].sum(),
        'hit_rate': (t['edge_simple_bps'] > 0).mean(),
    }

data = {s: load_trades(s) for s in SYMBOLS}
pl.DataFrame([{'symbol': s, **summarise(t)} for s, t in data.items()]).with_columns(pl.selectors.float().round(2))
"""),
    md(r"""
## 1. What does a typical trade look like?

The per-trade return is noisy: the hit rate is barely above 50%. The edge
comes from winners being larger than losers. That is why the *average*
(gross bps) and the *t-statistic* matter more than the hit rate.
"""),
    code(r"""
fig, axes = plt.subplots(1, len(SYMBOLS), figsize=(14, 3.5))
for ax, (s, t) in zip(np.atleast_1d(axes), data.items()):
    v = t['edge_simple_bps'].to_numpy()
    ax.hist(np.clip(v, -150, 150), bins=120, color='steelblue')
    ax.axvline(v.mean(), color='red', lw=1.5, label=f'mean {v.mean():.1f} bps')
    ax.set_title(f'{s}: per-trade return (clipped ±150 bps)'); ax.legend()
plt.tight_layout()
"""),
    md(r"""
## 2. Cumulative edge over time, and how much depends on a few days

The solid line adds up every trade's bps. The dashed line removes the
`DROP_BEST_DAYS` best days. The gap between them shows how much rests on
crash and spike days.
"""),
    code(r"""
fig, axes = plt.subplots(len(SYMBOLS), 1, figsize=(12, 3.2 * len(SYMBOLS)), sharex=True)
for ax, (s, t) in zip(np.atleast_1d(axes), data.items()):
    daily = t.group_by('day').agg(pl.col('edge_simple_bps').sum().alias('s')).sort('day')
    best = set(daily.sort('s', descending=True).head(DROP_BEST_DAYS)['day'].to_list())
    days = daily['day'].to_numpy().astype('datetime64[D]')
    ax.plot(days, np.cumsum(daily['s'].to_numpy()), label='all days')
    ax.plot(days, np.cumsum(np.where(np.isin(daily['day'].to_numpy(), list(best)), 0, daily['s'].to_numpy())), '--', label=f'without best {DROP_BEST_DAYS} days')
    ax.set_title(f'{s}: cumulative gross bps (sum over trades)'); ax.legend(loc='upper left')
plt.tight_layout()
"""),
    md(r"""
## 3. Is it fading? Average edge per trade by year, and a rolling 90-day mean
"""),
    code(r"""
rows = []
for s, t in data.items():
    for (year,), g in t.group_by(pl.col('time').dt.year()):
        rows.append({'symbol': s, 'year': year, 'gross_bps': g['edge_simple_bps'].mean(), 'trades': g.height})
by_year = pl.DataFrame(rows).pivot(on='symbol', index='year', values='gross_bps').sort('year')
display(by_year.with_columns(pl.selectors.float().round(1)))

fig, ax = plt.subplots(figsize=(12, 3.5))
for s, t in data.items():
    daily = t.group_by('day').agg(pl.col('edge_simple_bps').sum().alias('s'), pl.len().alias('n')).sort('day')
    s90 = daily['s'].rolling_sum(90, min_samples=30) / daily['n'].rolling_sum(90, min_samples=30)
    ax.plot(daily['day'].to_numpy().astype('datetime64[D]'), s90.to_numpy(), label=s)
ax.axhline(2 * FEE_PER_SIDE_BPS, color='black', ls=':', label=f'round-trip fee {2 * FEE_PER_SIDE_BPS:.0f} bps')
ax.set_title('Rolling 90-trading-day mean edge per trade (bps)'); ax.legend()
"""),
    md(r"""
## 4. Robustness grid: bar size × model (gross and ordinary-day bps)

The registered cell is `return_pretrend` at 1,000 events. Neighbouring cells
being similar is reassuring. A lone bright cell would be a warning sign.
"""),
    code(r"""
SYMBOL_FOR_GRID = 'ETHUSDT'
grid = (pl.scan_parquet(TRADES)
        .filter((pl.col('symbol') == SYMBOL_FOR_GRID) & (pl.col('horizon') == HORIZON) & (pl.col('tail') == TAIL))
        .group_by('n_events', 'model').agg(pl.col('edge_simple_bps').mean().alias('gross_bps'), pl.len().alias('trades'))
        .collect())
grid.pivot(on='model', index='n_events', values='gross_bps').sort('n_events').with_columns(pl.selectors.float().round(1))
"""),
    md(r"""
## 5. Clustering: trades arrive in bursts

Many trades fire together in fast markets. Size per trade × trades at once is
the real exposure, and the real market impact.
"""),
    code(r"""
fig, axes = plt.subplots(1, len(SYMBOLS), figsize=(14, 3.5))
for ax, (s, t) in zip(np.atleast_1d(axes), data.items()):
    per_day = t.group_by('day').len()['len'].to_numpy()
    ax.hist(per_day, bins=60, color='darkorange'); ax.set_yscale('log')
    ax.set_title(f'{s}: trades per day (median {np.median(per_day):.0f}, max {per_day.max()})')
plt.tight_layout()
"""),
    md(r"""
## 6. Costs: fees, spread and slippage (indicative only)

Uses level-1 quotes from 2023-05 to 2024-03 (`check_c006_slippage.py`). Top-of-book
data is imperfect and short, and depth beyond the best price is unknown. The
square-root law gives a rough slippage estimate.

The right-hand chart applies today's settings to the **recent** (2025–26)
edge, which is the relevant level going forward.
"""),
    code(r"""
slip = pl.read_csv(REPORTS / 'c006' / 'slippage_check.csv')
display(slip.select('symbol', 'size_usd', 'fits_at_touch_entry', 'sqrt_law_slippage_bps', 'net_edge_sqrt_law_bps')
        .with_columns(pl.selectors.float().round(2)))

recent = {s: t.filter(pl.col('time') >= pl.datetime(2025, 1, 1))['edge_simple_bps'].mean() for s, t in data.items()}
fig, ax = plt.subplots(figsize=(10, 3.5))
for s in SYMBOLS:
    sub = slip.filter(pl.col('symbol') == s).sort('size_usd')
    sizes = sub['size_usd'].to_numpy()
    net = recent[s] - 2 * FEE_PER_SIDE_BPS - 0.75 - sub['sqrt_law_slippage_bps'].to_numpy()   # 0.75 bp ~ spread paid
    ax.plot(sizes, net, marker='o', label=f'{s} (2025–26 gross {recent[s]:.1f} bps)')
ax.axhline(0, color='black', lw=1); ax.set_xscale('log')
ax.set_xlabel('size per trade, USD'); ax.set_ylabel('net bps per trade')
ax.set_title('Recent edge after fees, spread and square-root slippage (rough)'); ax.legend()
"""),
    md(r"""
## 7. Stress reversion (C-007): fading rare extreme moves

`RATE` sets how rare the trigger is (50, 100 or 200 episodes a year; registered:
100 and 200). Entry is at the trigger, with a 5-minute hold.
"""),
    code(r"""
RATE = 100
rows = []
for s, path in [('BTCUSDT', 'stress_episodes_v1'), ('ETHUSDT', 'stress_c007_ETHUSDT'), ('SOLUSDT', 'stress_c007_SOLUSDT')]:
    tr = pl.read_parquet(REPORTS / path / 'trades.parquet')
    ep = pl.read_parquet(REPORTS / path / 'episodes.parquet').select('trigger_second', 'target_per_year', 'direction')
    tr = (tr.join(ep, on=['trigger_second', 'target_per_year'])
          .filter((pl.col('target_per_year') == RATE) & (pl.col('rule') == 'trigger') & (pl.col('holding_seconds') == 300)
                  & (pl.col('trigger_second') >= 1_672_531_200))
          .with_columns((-pl.col('direction') * ((-pl.col('direction') * pl.col('gross_bps') / 1e4).exp() - 1) * 1e4).alias('bps'),
                        pl.from_epoch('trigger_second').dt.year().alias('year')))
    for (year,), g in tr.group_by('year'):
        rows.append({'symbol': s, 'year': year, 'mean_bps': g['bps'].mean(), 'trades': g.height})
pl.DataFrame(rows).pivot(on='symbol', index='year', values='mean_bps').sort('year').with_columns(pl.selectors.float().round(1))
"""),
    md(r"""
## 8. Execution (H-018): taking vs resting

These are implementation shortfalls (bps, lower is better) for random $10k
orders over 2023-08 to 2024-03:

- **E0:** take immediately.
- **E1:** rest at the best price, re-peg, and take at a 30-minute deadline.
- **E2:** E1 plus signal-timed crossing.

The fees in this test were 2 bps maker and 5 bps taker.
"""),
    code(r"""
ev = json.loads((REPORTS / 'execution_h018' / 'test_evaluation.json').read_text())
rows = []
for s, r in ev.items():
    for row in r['summary']:
        if row['run'] in ('E0', 'E1', 'E2'):
            rows.append({'symbol': s, 'strategy': row['run'], 'is_bps': row['is_bps'], 'p95_bps': row['is_p95'],
                         'passive_share': row['passive_share']})
pl.DataFrame(rows).pivot(on='strategy', index='symbol', values='is_bps').with_columns(pl.selectors.float().round(2))
"""),
    md(r"""
## Where to go from here

- **Change the controls** and re-run, to get a feel for the effect's shape
  (horizon, tail, bar size).
- **Remember:** the registered result is the default cell. Anything you find by
  browsing is a lead for a *new* registration, not a result.
- **The next formal step** is the locked months (2026-06 to 2026-08). It is
  single-use and can only reject.
"""),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Alpha Search (.venv)", "language": "python", "name": "alpha-search-venv"},
        "language_info": {"name": "python", "version": "3.12"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
