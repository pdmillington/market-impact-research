"""Generate the sweep-conditioned event-decay workbench."""

from pathlib import Path

import nbformat as nbf


REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "alpha-search" / "notebooks" / "EventSweepDecayWorkbenchV2.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        r"""
# Event decay v2: conditioning on realized execution sweep

This workbench replaces fill-count/price-level classes with the economically direct measure of event aggression: the side-aligned difference between the first and last execution prices.

Fill count and price-level count remain visible as secondary diagnostics, but they do not define the primary curves.
"""
    ),
    md(
        r"""
## Definitions

For reconstructed event $$i$$ with aggressor side $$\epsilon_i$$:

$$
S_i=10^4\epsilon_i\log\left(\frac{P_i^{last}}{P_i^{first}}\right)
$$

is the realized endpoint sweep. Immediate impact is decomposed as:

$$
I_i^{immediate}
=10^4\epsilon_i\log\left(\frac{P_i^{first}}{P_{i-1}^{last}}\right)
+S_i.
$$

Subsequent total and post-event responses are:

$$
R_i(h)=10^4\epsilon_i\log\left(\frac{P_{i,h}}{P_{i-1}^{last}}\right),
\qquad
R_i^{post}(h)=10^4\epsilon_i\log\left(\frac{P_{i,h}}{P_i^{last}}\right).
$$

Positive sweeps are divided by quartile boundaries calibrated through December 2022 and then frozen. A negative side-aligned endpoint movement is retained as an anomaly rather than treated as aggressive execution.
"""
    ),
    code(
        """
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
RESULT_ROOT = REPO / 'alpha-search' / 'reports' / 'event_decay_sweep_v2'
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 115, 'axes.spines.top': False, 'axes.spines.right': False})
"""
    ),
    md("## Manual controls"),
    code(
        """
PERIOD = 'validation'       # development, validation, later_exposed
SIDE_VIEW = 'equal_side'    # equal_side, buy, sell
SHOW_CONFIDENCE_BANDS = True
PRIMARY_CLASSES = ['no_endpoint_sweep', 'sweep_q1', 'sweep_q2', 'sweep_q3', 'sweep_q4']

assert PERIOD in {'development', 'validation', 'later_exposed'}
assert SIDE_VIEW in {'equal_side', 'buy', 'sell'}
"""
    ),
    md("## Load compact daily summaries"),
    code(
        """
required = ['run_manifest.json', 'sweep_quantile_cuts.csv',
            'event_sweep_class_counts.csv', 'decay_sample_summary.csv', 'decay_daily.csv']
missing = [name for name in required if not (RESULT_ROOT / name).exists()]
if missing:
    raise FileNotFoundError(
        f'Missing {missing}. Run alpha-search/scripts/run_event_decay_by_execution.py first.'
    )

manifest = json.loads((RESULT_ROOT / 'run_manifest.json').read_text())
cuts = pl.read_csv(RESULT_ROOT / 'sweep_quantile_cuts.csv')
counts = pl.read_csv(RESULT_ROOT / 'event_sweep_class_counts.csv')
sample = pl.read_csv(RESULT_ROOT / 'decay_sample_summary.csv')
daily = pl.read_csv(RESULT_ROOT / 'decay_daily.csv').with_columns(
    pl.when(pl.col('source_month') <= '2022-12').then(pl.lit('development'))
    .when(pl.col('source_month') <= '2024-12').then(pl.lit('validation'))
    .otherwise(pl.lit('later_exposed')).alias('period')
)
display(manifest)
display(cuts)
print(f"Sampled {sample['sample_events'].sum():,} anchors from {sample['source_events'].sum():,} events")
"""
    ),
    md("## Exact sweep-class frequencies and economic size"),
    code(
        """
count_summary = counts.group_by('category').agg(
    pl.col('events').sum(),
    pl.col('total_quantity').sum(),
    (pl.col('mean_endpoint_sweep_bps') * pl.col('events')).sum().alias('_weighted_sweep'),
    (pl.col('mean_quantity') * pl.col('events')).sum().alias('_weighted_quantity'),
).with_columns(
    (pl.col('events') / pl.col('events').sum()).alias('event_share'),
    (pl.col('total_quantity') / pl.col('total_quantity').sum()).alias('quantity_share'),
    (pl.col('_weighted_sweep') / pl.col('events')).alias('mean_endpoint_sweep_bps'),
    (pl.col('_weighted_quantity') / pl.col('events')).alias('mean_quantity'),
).drop('_weighted_sweep', '_weighted_quantity').sort('mean_endpoint_sweep_bps')
display(count_summary)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
labels = [value.replace('_', ' ') for value in count_summary['category']]
x = np.arange(len(labels))
axes[0].bar(x, 100 * count_summary['event_share'], color='#64748b')
axes[1].bar(x, count_summary['mean_quantity'], color='#2563eb')
for ax in axes:
    ax.set_xticks(x, labels, rotation=35, ha='right')
axes[0].set_ylabel('share of events (%)')
axes[1].set_ylabel('mean quantity (BTC)')
axes[0].set_title('frequency')
axes[1].set_title('size composition')
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
The quantity plot is essential: a difference between sweep classes is not automatically a sweep effect if the large-sweep events are also much larger. Matching or regression controls for quantity and volatility are the next causal-identification layer.
"""
    ),
    md("## Construct raw and equal-side daily paths"),
    code(
        """
buy = daily.filter(pl.col('side') == 'buy').select(
    'period', 'category', 'utc_day', 'horizon_ms', 'horizon_seconds',
    pl.col('mean_immediate_response_bps').alias('buy_immediate'),
    pl.col('mean_total_response_bps').alias('buy_total'),
    pl.col('mean_post_event_response_bps').alias('buy_post'),
)
sell = daily.filter(pl.col('side') == 'sell').select(
    'period', 'category', 'utc_day', 'horizon_ms',
    pl.col('mean_immediate_response_bps').alias('sell_immediate'),
    pl.col('mean_total_response_bps').alias('sell_total'),
    pl.col('mean_post_event_response_bps').alias('sell_post'),
)
paired = buy.join(sell, on=['period', 'category', 'utc_day', 'horizon_ms'], how='inner').with_columns(
    ((pl.col('buy_immediate') + pl.col('sell_immediate')) / 2).alias('immediate'),
    ((pl.col('buy_total') + pl.col('sell_total')) / 2).alias('total'),
    ((pl.col('buy_post') + pl.col('sell_post')) / 2).alias('post'),
)

if SIDE_VIEW == 'equal_side':
    path_daily = paired.select('period', 'category', 'utc_day', 'horizon_ms',
                               'horizon_seconds', 'immediate', 'total', 'post')
else:
    path_daily = daily.filter(pl.col('side') == SIDE_VIEW).select(
        'period', 'category', 'utc_day', 'horizon_ms', 'horizon_seconds',
        pl.col('mean_immediate_response_bps').alias('immediate'),
        pl.col('mean_total_response_bps').alias('total'),
        pl.col('mean_post_event_response_bps').alias('post'),
    )

curve = path_daily.group_by('period', 'category', 'horizon_ms', 'horizon_seconds').agg(
    pl.len().alias('days'),
    pl.col('immediate').mean(),
    pl.col('total').mean(),
    pl.col('post').mean(),
    (pl.col('total').std() / pl.len().sqrt()).alias('total_se'),
    (pl.col('post').std() / pl.len().sqrt()).alias('post_se'),
).sort('category', 'horizon_ms')
"""
    ),
    md("## Total response by realized sweep"),
    code(
        """
colours = {'no_endpoint_sweep': '#64748b', 'sweep_q1': '#38bdf8',
           'sweep_q2': '#2563eb', 'sweep_q3': '#7c3aed', 'sweep_q4': '#dc2626'}
fig, ax = plt.subplots(figsize=(11, 6))
for category in PRIMARY_CLASSES:
    part = curve.filter((pl.col('period') == PERIOD) & (pl.col('category') == category)).sort('horizon_ms')
    if part.is_empty():
        continue
    x = np.r_[0.0, part['horizon_seconds'].to_numpy()]
    y = np.r_[part['immediate'][0], part['total'].to_numpy()]
    ax.plot(x, y, marker='o', ms=3, color=colours[category], label=category.replace('_', ' '))
    if SHOW_CONFIDENCE_BANDS:
        se = np.r_[0.0, part['total_se'].to_numpy()]
        ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=colours[category], alpha=.08)
ax.axhline(0, color='black', lw=.8)
ax.set_xscale('symlog', linthresh=.1)
ax.set_xlabel('seconds after final execution (0 = immediate response)')
ax.set_ylabel('side-aligned total response (bps)')
ax.set_title(f'{PERIOD}; {SIDE_VIEW.replace("_", " ")}')
ax.legend(fontsize=8)
plt.show()
"""
    ),
    md("## Post-event continuation or reversal"),
    code(
        """
fig, ax = plt.subplots(figsize=(11, 6))
for category in PRIMARY_CLASSES:
    part = curve.filter((pl.col('period') == PERIOD) & (pl.col('category') == category)).sort('horizon_ms')
    if part.is_empty():
        continue
    x = np.r_[0.0, part['horizon_seconds'].to_numpy()]
    y = np.r_[0.0, part['post'].to_numpy()]
    ax.plot(x, y, marker='o', ms=3, color=colours[category], label=category.replace('_', ' '))
ax.axhline(0, color='black', lw=.8)
ax.set_xscale('symlog', linthresh=.1)
ax.set_xlabel('seconds after final execution')
ax.set_ylabel('movement from final execution price (bps)')
ax.set_title('Positive = continuation; negative = reversal')
ax.legend(fontsize=8)
plt.show()
"""
    ),
    md("## Incremental reaction and counterreaction"),
    code(
        """
base = path_daily.select(
    'period', 'category', 'utc_day',
    pl.lit(0, dtype=pl.Int64).alias('horizon_ms'),
    pl.lit(0.0).alias('horizon_seconds'),
    pl.col('immediate').alias('total'),
)
increments = pl.concat([
    base,
    path_daily.select('period', 'category', 'utc_day', 'horizon_ms', 'horizon_seconds', 'total')
]).sort('period', 'category', 'utc_day', 'horizon_ms').with_columns(
    pl.col('total').diff().over('period', 'category', 'utc_day').alias('increment')
).filter(pl.col('horizon_ms') > 0).group_by(
    'period', 'category', 'horizon_ms', 'horizon_seconds'
).agg(
    pl.col('increment').mean().alias('mean_increment'),
    (pl.col('increment').std() / pl.len().sqrt()).alias('standard_error'),
    pl.len().alias('days'),
).sort('category', 'horizon_ms')

fig, ax = plt.subplots(figsize=(11, 6))
for category in PRIMARY_CLASSES:
    part = increments.filter((pl.col('period') == PERIOD) & (pl.col('category') == category))
    ax.plot(part['horizon_seconds'], part['mean_increment'], marker='o', ms=3,
            color=colours[category], label=category.replace('_', ' '))
ax.axhline(0, color='black', lw=.8)
ax.set_xscale('log')
ax.set_xlabel('end of response interval (seconds)')
ax.set_ylabel('incremental side-aligned response (bps)')
ax.set_title('A wave requires reproducible alternating increments—not merely one peak')
ax.legend(fontsize=8)
plt.show()
"""
    ),
    md("## Reconstruction/path anomalies"),
    code(
        """
diagnostic_categories = ['opposite_endpoint_move', 'path_excursion_returned_to_start']
display(sample.select(
    'source_month', 'sample_events', 'positive_sweep_events',
    'no_endpoint_sweep_events', 'opposite_endpoint_move_events',
    'path_excursion_returned_to_start_events'
).tail(24))

diagnostic = daily.filter(
    (pl.col('side') == 'all') & pl.col('category').is_in(diagnostic_categories)
).group_by('source_month', 'category').agg(
    pl.col('sample_events').first(),
    pl.col('mean_endpoint_sweep_bps').first(),
    pl.col('mean_execution_span_bps').first(),
).sort('source_month', 'category')
display(diagnostic.tail(30))
"""
    ),
    md(
        r"""
## Interpretation limits and next model

- Endpoint sweep measures realized execution aggression more directly than fill multiplicity.
- Volume still confounds comparisons: large events are more likely to sweep. The next estimation stage should match sweep classes on quantity, volatility, side and activity, or include them in a local-projection regression.
- Raw response includes the effect of correlated subsequent order flow. It is not yet the isolated decay kernel of one hypothetical trade.
- A propagator/local-projection model controlling for lagged and subsequent signed flow is required before interpreting repeated sign changes as a reaction wave.
- Fill count and price-level count should only be promoted if they add predictive information conditional on quantity and realized sweep.
"""
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "shared-methodology (.venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
