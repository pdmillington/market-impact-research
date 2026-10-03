"""Build the crowding backtest workbench from run_crowding_backtest.py outputs."""

from pathlib import Path

import nbformat as nbf


REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "alpha-search" / "notebooks" / "CrowdingBacktestWorkbench.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        r"""
# Crowding backtest workbench

An implementable version of the Step 4 forecasts. Each hour the frozen OOS
forecast becomes a unit signal (`tail`: ±1 when |forecast| is in the training
top 20%; `continuous`: forecast / that threshold, clipped). The position is the
mean of the last H hourly signals, i.e. H staggered tranches each held H hours,
so overlapping trades net into one position.

Hourly P&L = position × next-hour perp return − position × funding settled that
hour − fee × |position change|. Fees per side: maker 2 bps (assumes passive
fills), taker 5 bps. Test period January 2023 to May 2026.

The primary variant (`all`, 24h, tail, 12-month window) was named before the run.
`baseline` and `baseline_positioning` were added afterwards for attribution.
"""
    ),
    code(
        r"""
from pathlib import Path
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/alpha-search-matplotlib')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')

REPO = next(path for path in (Path.cwd(), *Path.cwd().parents) if (path / 'alpha-search').is_dir() and (path / 'shared-methodology').is_dir())
RESULTS = REPO / 'alpha-search' / 'reports' / 'crowding_backtest_v1'
config = json.loads(
    (REPO / 'alpha-search' / 'configs' / 'crowding_backtest_v1.json').read_text()
)
summary = pl.read_csv(RESULTS / 'backtest_summary.csv')
by_year = pl.read_csv(RESULTS / 'backtest_by_year.csv')
attribution = pl.read_csv(RESULTS / 'positioning_attribution.csv')
hourly = pl.read_parquet(RESULTS / 'backtest_hourly.parquet')

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
INK, MUTED = '#2b2b2b', '#8a8a85'
MODELS = ['baseline', 'baseline_funding', 'baseline_basis', 'baseline_positioning', 'all']
MODEL_COLOUR = dict(zip(MODELS, SERIES))

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 115,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.color': '#e6e6e3',
    'lines.linewidth': 1.8,
})
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_float_precision(2)
"""
    ),
    md("## Manual controls"),
    code(
        r"""
HORIZON = 24            # 8 or 24 hours
RULE = 'tail'           # 'tail' or 'continuous'
WINDOW = 12             # training window in months: 12 or 6
COST = 'maker'          # 'gross', 'maker' or 'taker'

assert HORIZON in config['horizons_hours']
assert RULE in config['position_rules']
assert WINDOW in config['training_windows_months']
assert COST in config['costs_bps_per_side']
"""
    ),
    md("## 1. Summary of all variants"),
    code(
        r"""
summary.select(
    'model', 'horizon_hours', 'position_rule', 'training_window_months', 'primary',
    'mean_abs_position', 'turnover_per_day',
    f'{COST}_return_pct_per_year', f'{COST}_sharpe', f'{COST}_max_drawdown_pct',
    'recent_maker_return_pct_per_year', 'recent_maker_sharpe',
).sort(f'{COST}_sharpe', descending=True)
"""
    ),
    md(
        r"""
## 2. Equity curves

Cumulative net P&L (percent of notional, not compounded) for the selected
horizon, rule and window. Buy-and-hold is on its own panel because its scale is
much larger.
"""
    ),
    code(
        r"""
view = hourly.filter(
    (pl.col('horizon_hours') == HORIZON) & (pl.col('position_rule') == RULE)
    & (pl.col('training_window_months') == WINDOW)
)
fig, ax = plt.subplots(figsize=(10.5, 4.6))
for model in MODELS:
    series = view.filter(pl.col('model') == model).sort('time_s')
    if series.is_empty():
        continue
    equity = series[f'net_{COST}_bps'].cum_sum() / 100
    ax.plot(series.select(pl.from_epoch('time_s'))['time_s'], equity,
            color=MODEL_COLOUR[model], label=model)
ax.axhline(0, color=MUTED, linewidth=1)
ax.axvline(np.datetime64('2025-06-01'), color=MUTED, linestyle=':', linewidth=1)
ax.text(np.datetime64('2025-06-05'), ax.get_ylim()[1] * 0.95, 'last 12 months',
        fontsize=9, color=INK, va='top')
ax.set_ylabel(f'cumulative {COST} net P&L (%)')
ax.set_title(f'{HORIZON}h horizon, {RULE} rule, {WINDOW}-month training window', loc='left')
ax.legend(frameon=False, loc='upper left')
plt.show()
"""
    ),
    code(
        r"""
panel = pl.read_parquet(
    REPO / 'alpha-search' / 'reports' / 'funding_positioning_v1' / 'hourly_panel.parquet'
).filter(pl.col('time_s') >= view['time_s'].min()).sort('time_s')
long = (panel['F1h'].fill_nan(0.0) - panel['carry1h'].fill_nan(0.0)).cum_sum() / 100
fig, ax = plt.subplots(figsize=(10.5, 2.8))
ax.plot(panel.select(pl.from_epoch('time_s'))['time_s'], long, color=MUTED)
ax.set_ylabel('cumulative %')
ax.set_title('Reference: one unit long BTC perp, paying funding', loc='left')
plt.show()
"""
    ),
    md("## 3. Stability by year"),
    code(
        r"""
by_year.filter(
    (pl.col('horizon_hours') == HORIZON) & (pl.col('position_rule') == RULE)
    & (pl.col('training_window_months') == WINDOW)
).pivot(on='year', index='model', values=f'{COST}_return_pct_per_year')
"""
    ),
    md(
        r"""
## 4. What drives it: drop-one attribution of the positioning group

`baseline_positioning`, tail rule, 12-month window. Each row removes one
positioning feature; the last removes all three long/short ratio features.
"""
    ),
    code("attribution"),
    md(
        r"""
## Reading

See `alpha-search/reports/crowding-backtest-v1-initial-results.md`.
"""
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Alpha Search (.venv)",
            "language": "python",
            "name": "alpha-search-venv",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
