"""Build the residual-decomposition workbench from run_residual_decomposition.py outputs."""

from pathlib import Path

import nbformat as nbf


REPO = Path("/Users/petermillington/Research/market-impact-research")
OUTPUT = REPO / "alpha-search" / "notebooks" / "ResidualDecompositionWorkbench.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        r"""
# Residual decomposition workbench

This notebook answers the Step 1 gate: **does the P&B impact residual carry
predictive information beyond its own components, pretrend and a naive flow term,
and does anything clear realistic costs?**

All features are signed in price direction and known at the completed bar endpoint:

| Symbol | Definition |
|---|---|
| $R_t$ | bar return / trailing 24h volatility |
| $P_t$ | $s_t\widehat y_t$: `pb_activity` prediction, frozen fit calibrated before the month |
| $X_t$ | $R_t-P_t$: the signed residual used so far |
| $q_t$ | $s_t\,|Q_t|/V_{24h}$: linear signed flow, a naive benchmark for $P$ |
| `pre15`, `pre60` | clock-time return ending at bar start / trailing volatility |

Because $X$ is the fixed combination $R-P$, it can only beat free weights on $R$
and $P$ by luck. The informative questions are whether the 1:−1 restriction holds,
and whether $P$ adds anything once $R$, pretrend and $q$ are present.

Forecast regressions are fitted on the preceding 12 months (with an embargo) and
frozen for each test month, January 2023 to May 2026. June–August 2026 are not
used, including as target prices.
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
from matplotlib.ticker import NullFormatter
import numpy as np
import polars as pl
from IPython.display import Markdown, display
get_ipython().run_line_magic('matplotlib', 'inline')

REPO = Path('/Users/petermillington/Research/market-impact-research')
RESULTS = REPO / 'alpha-search' / 'reports' / 'residual_decomposition_v1'
config = json.loads(
    (REPO / 'alpha-search' / 'configs' / 'residual_decomposition_v1.json').read_text()
)
univariate = pl.read_csv(RESULTS / 'univariate_summary.csv')
fm = pl.read_csv(RESULTS / 'fama_macbeth_summary.csv', infer_schema_length=None)
oos = pl.read_csv(RESULTS / 'oos_summary.csv')
oos_monthly = pl.read_csv(RESULTS / 'oos_monthly.csv')
increments = pl.read_csv(RESULTS / 'oos_increments.csv')
tails = pl.read_csv(RESULTS / 'tail_breakeven.csv')
tail_monthly = pl.read_csv(RESULTS / 'tail_monthly.csv')
entry_delay = pl.read_csv(RESULTS / 'entry_delay.csv')

TAKER = 2 * config['costs_bps_per_side']['taker']
MAKER = 2 * config['costs_bps_per_side']['maker']
SIZES = config['event_sizes']

# Fixed categorical order (never cycled); ink colours for text and reference lines.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
INK, MUTED = '#2b2b2b', '#8a8a85'
KEY_MODELS = ['residual', 'bar_return', 'return_pretrend', 'full_impact_flow']
MODEL_COLOUR = dict(zip(KEY_MODELS, SERIES))

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 115,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.color': '#e6e6e3',
    'lines.linewidth': 2,
})
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)
"""
    ),
    md("## Manual controls"),
    code(
        r"""
N_EVENTS = 8000          # 200, 500, 1000, 2000, 4000, 8000
HORIZON = 15             # 5, 10, 15, 30 minutes
TAIL_FRACTION = 0.01     # 0.01, 0.05, 0.2
TAIL_MODEL = 'return_pretrend'

assert N_EVENTS in SIZES
assert HORIZON in config['future_horizons_minutes']
assert TAIL_FRACTION in config['tail_fractions']
assert TAIL_MODEL in config['models']
display(entry_delay)
"""
    ),
    md(
        r"""
## 1. Univariate information by feature

Mean monthly Spearman IC against the signed future return, 2023–May 2026.
Negative means reversal. The key reading is the sign of $P$: if flow-implied
impact *continued*, $P$ would be positive and subtracting it would sharpen the
reversal signal. If $P$ also reverses, the residual cancels information.
"""
    ),
    code(
        r"""
features = ['R', 'P', 'X', 'q', 'pre15', 'pre60']
view = univariate.filter(
    (pl.col('horizon_minutes') == HORIZON) & (pl.col('period') == '2023-01..2026-05')
)
grid = np.array([
    [view.filter((pl.col('n_events') == n) & (pl.col('feature') == f))['mean_spearman_ic'][0]
     for f in features]
    for n in SIZES
])
limit = np.abs(grid).max()
fig, ax = plt.subplots(figsize=(7.5, 4.2))
image = ax.imshow(grid, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
ax.set_xticks(range(len(features)), features)
ax.set_yticks(range(len(SIZES)), [f'{n:,}' for n in SIZES])
ax.set_ylabel('events per bar')
ax.grid(False)
for i in range(len(SIZES)):
    for j in range(len(features)):
        ax.text(j, i, f'{grid[i, j]:+.3f}', ha='center', va='center', fontsize=8.5,
                color='white' if abs(grid[i, j]) > 0.6 * limit else INK)
fig.colorbar(image, ax=ax, label='mean monthly Spearman IC')
ax.set_title(f'Univariate IC at {HORIZON} minutes, 2023–May 2026', loc='left')
plt.show()
view.select('n_events', 'feature', 'mean_spearman_ic', 'spearman_ic_t', 'negative_month_fraction')
"""
    ),
    md(
        r"""
## 2. The 1:−1 restriction

Within-month OLS of the winsorised future return on $R$ and $P$ (raw volatility
units). The residual imposes $b_R+b_P=0$. Its t-statistic is across months.
"""
    ),
    code(
        r"""
fm.filter(pl.col('model') == 'components').select(
    'n_events', 'horizon_minutes', 'months', 'b_R_mean', 'b_R_t', 'b_P_mean', 'b_P_t',
    'b_R_plus_b_P_mean', 'b_R_plus_b_P_t'
).sort('n_events', 'horizon_minutes')
"""
    ),
    md(
        r"""
## 3. Out-of-sample forecast comparison

Each model is fitted on the preceding 12 months and frozen. The chart shows mean
monthly out-of-sample Spearman IC of each model's forecast (positive = useful)
across bar sizes.
"""
    ),
    code(
        r"""
selected = oos.filter(pl.col('horizon_minutes') == HORIZON)
fig, ax = plt.subplots(figsize=(8, 4.2))
for model in KEY_MODELS:
    series = selected.filter(pl.col('model') == model).sort('n_events')
    ax.plot(series['n_events'], series['mean_spearman_ic'], marker='o', markersize=6,
            color=MODEL_COLOUR[model], label=model)
ax.set_xscale('log')
ax.set_xticks(SIZES, [f'{n:,}' for n in SIZES])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.axhline(0, color=MUTED, linewidth=1)
ax.set_xlabel('events per bar')
ax.set_ylabel('mean monthly OOS Spearman IC')
ax.set_title(f'Out-of-sample forecast IC at {HORIZON} minutes', loc='left')
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1.0))
plt.tight_layout()
plt.show()
selected.filter(pl.col('n_events') == N_EVENTS).sort('mean_spearman_ic', descending=True)
"""
    ),
    md(
        r"""
### Paired increments

Mean monthly difference in OOS IC between nested models, with a t-statistic over
test months. "Adding P" rows are the direct test of whether the impact model is
needed for alpha.
"""
    ),
    code(
        r"""
increments.filter(pl.col('horizon_minutes') == HORIZON).select(
    'n_events', 'comparison', 'mean_ic_difference', 'difference_t',
    'candidate_better_month_fraction'
).sort('comparison', 'n_events')
"""
    ),
    code(
        r"""
monthly = oos_monthly.filter(
    (pl.col('n_events') == N_EVENTS) & (pl.col('horizon_minutes') == HORIZON)
)
fig, ax = plt.subplots(figsize=(10, 3.8))
for model in KEY_MODELS:
    series = monthly.filter(pl.col('model') == model).sort('test_month')
    ax.plot(range(series.height), series['spearman_ic'], color=MODEL_COLOUR[model],
            label=model, linewidth=1.6)
ticks = series['test_month'].to_list()
ax.set_xticks(range(0, len(ticks), 6), ticks[::6])
ax.axhline(0, color=MUTED, linewidth=1)
ax.set_ylabel('OOS Spearman IC')
ax.set_title(f'Monthly OOS IC, {N_EVENTS:,}-event bars, {HORIZON} minutes', loc='left')
ax.legend(frameon=False, ncol=4, loc='upper left')
plt.show()
"""
    ),
    md(
        r"""
## 4. Costed break-even

Trades are taken only when the frozen forecast's magnitude exceeds the training
quantile for the selected tail fraction. Edge is the signed return after a delayed
entry at the first 200-event endpoint at least one second after the decision
(the entry-delay table above gives the realised delay). Trades overlap; the
daily-block t-statistic treats each UTC day as one observation.

Reference lines are round-trip costs at the configured base-tier fees (by default
taker 10 bps and maker 4 bps; edit `costs_bps_per_side` in the config to match
the actual fee tier). Maker fills are not guaranteed and
carry adverse selection, so the maker line is optimistic.
"""
    ),
    code(
        r"""
view = tails.filter(
    (pl.col('n_events') == N_EVENTS) & (pl.col('horizon_minutes') == HORIZON)
)
fig, ax = plt.subplots(figsize=(8, 4.2))
fractions = config['tail_fractions']
for model in KEY_MODELS:
    series = view.filter(pl.col('model') == model).sort('tail_fraction')
    ax.plot(series['tail_fraction'] * 100, series['mean_edge_delayed_bps'], marker='o',
            markersize=6, color=MODEL_COLOUR[model], label=model)
for level, label in [(TAKER, 'taker round trip'), (MAKER, 'maker round trip')]:
    ax.axhline(level, color=MUTED, linestyle='--', linewidth=1.2)
    ax.text(fractions[-1] * 100, level, f' {label} ({level:g} bps)', va='bottom',
            ha='right', fontsize=9, color=INK)
ax.set_xscale('log')
ax.set_xticks([f * 100 for f in fractions], [f'top {f:.0%}' for f in fractions])
ax.axhline(0, color=MUTED, linewidth=1)
ax.set_ylabel('mean gross edge per trade (bps)')
ax.set_title(f'Tail edge vs costs, {N_EVENTS:,}-event bars, {HORIZON} minutes',
             loc='left')
ax.legend(frameon=False, loc='upper right')
plt.show()
view.select(
    'model', 'tail_fraction', 'trades_per_day', 'mean_edge_immediate_bps',
    'mean_edge_delayed_bps', 'daily_block_t', 'positive_month_fraction',
    'net_taker_bps', 'net_maker_bps'
).sort('tail_fraction', 'mean_edge_delayed_bps', descending=[False, True])
"""
    ),
    md(
        r"""
### Concentration: is the tail edge a steady signal or a few stress days?

Tail trades overlap and cluster, so a daily-block t-statistic can look strong
while almost all the edge comes from a handful of crash or cascade days. The
table reports the share of total edge from the best ten days, the edge on all
other days, and the edge by calendar year. A share above one means the remaining
days lose money in aggregate.
"""
    ),
    code(
        r"""
year_columns = [c for c in tails.columns if c.startswith('edge_20')]
tails.filter(pl.col('net_taker_bps') > 0).sort('net_taker_bps', descending=True).select(
    'n_events', 'horizon_minutes', 'model', 'tail_fraction', 'trades_per_day',
    'mean_edge_delayed_bps', 'daily_block_t', 'best_10_day_edge_share',
    'edge_excluding_best_10_days_bps', *year_columns
).head(25)
"""
    ),
    code(
        r"""
view = tails.filter(
    (pl.col('horizon_minutes') == HORIZON) & (pl.col('tail_fraction') == TAIL_FRACTION)
    & pl.col('model').is_in(KEY_MODELS)
)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, column, title in [
    (axes[0], 'mean_edge_delayed_bps', 'All days'),
    (axes[1], 'edge_excluding_best_10_days_bps', 'Excluding the best 10 days'),
]:
    for model in KEY_MODELS:
        series = view.filter(pl.col('model') == model).sort('n_events')
        ax.plot(series['n_events'], series[column], marker='o', markersize=6,
                color=MODEL_COLOUR[model], label=model)
    for level, label in [(TAKER, 'taker'), (MAKER, 'maker')]:
        ax.axhline(level, color=MUTED, linestyle='--', linewidth=1.2)
        ax.text(SIZES[0], level, f'{label} ', va='bottom', ha='left', fontsize=9,
                color=INK)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xscale('log')
    ax.set_xticks(SIZES, [f'{n:,}' for n in SIZES])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('events per bar')
    ax.set_title(title, loc='left')
axes[0].set_ylabel('mean gross edge per trade (bps)')
axes[1].legend(frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1.0))
fig.suptitle(f'Top {TAIL_FRACTION:.0%} tail edge at {HORIZON} minutes', x=0.01,
             ha='left')
plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
series = tail_monthly.filter(
    (pl.col('n_events') == N_EVENTS) & (pl.col('horizon_minutes') == HORIZON)
    & (pl.col('model') == TAIL_MODEL) & (pl.col('tail_fraction') == TAIL_FRACTION)
).sort('test_month').with_columns(
    (pl.col('sum_edge_delayed_bps') / pl.col('trades')).alias('edge')
)
fig, (ax, count_ax) = plt.subplots(
    2, 1, figsize=(10, 5.4), sharex=True, gridspec_kw={'height_ratios': [3, 1.2]}
)
colours = [SERIES[0] if value >= 0 else SERIES[1] for value in series['edge']]
ax.bar(range(series.height), series['edge'], color=colours, width=0.8)
ax.axhline(TAKER, color=MUTED, linestyle='--', linewidth=1.2)
ax.text(-0.5, TAKER, f'taker round trip {TAKER:g} bps', va='bottom', ha='left',
        fontsize=9, color=INK)
ax.axhline(0, color=MUTED, linewidth=1)
ax.set_ylabel('mean gross edge\nper trade (bps)')
ax.set_title(
    f'Monthly tail edge: {TAIL_MODEL}, top {TAIL_FRACTION:.0%}, '
    f'{N_EVENTS:,} events, {HORIZON} min', loc='left'
)
count_ax.bar(range(series.height), series['trades'], color=MUTED, width=0.8)
count_ax.set_ylabel('trades')
ticks = series['test_month'].to_list()
count_ax.set_xticks(range(0, len(ticks), 6), ticks[::6])
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
## 5. Pretrend refresh

The earlier aligned-pretrend evidence came from 2020–21. In signed-price terms
the aligned pretrend is simply the pretrend return, so its univariate IC is shown
for 2022 and for the 2023–May 2026 test period.
"""
    ),
    code(
        r"""
univariate.filter(pl.col('feature').is_in(['pre15', 'pre60'])).filter(
    pl.col('horizon_minutes') == HORIZON
).select('n_events', 'feature', 'period', 'months', 'mean_spearman_ic',
         'spearman_ic_t', 'negative_month_fraction').sort('feature', 'n_events', 'period')
"""
    ),
    md(
        r"""
## Reading

See `alpha-search/reports/residual-decomposition-v1-initial-results.md` for the
written interpretation and gate decision.
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
