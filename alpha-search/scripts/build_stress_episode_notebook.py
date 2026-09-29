"""Build the stress-episode workbench from run_stress_episodes.py outputs."""

from pathlib import Path

import nbformat as nbf


REPO = Path("/Users/petermillington/Research/market-impact-research")
OUTPUT = REPO / "alpha-search" / "notebooks" / "StressEpisodeWorkbench.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        r"""
# Stress-episode workbench

Step 3 of the alpha roadmap. Step 1 found that the only edge clearing costs came
from a handful of crash and cascade days. This notebook studies those days
directly as discrete episodes rather than as a continuous factor.

**Detection** runs on a 1-second grid built from `timestamp_direction_v1`
events. At each second, $z$ is the largest move over the trailing 30, 60 and 300
seconds, in units of trailing-24h one-minute volatility scaled by
$\sqrt{W/60}$. The episode threshold is calibrated on 2021–2022 to a target
episode rate and frozen for January 2023 to May 2026. Qualifying seconds less
than 30 minutes apart form one episode, and the trigger is its first qualifying
second.

**Returns are reported as reversion**: positive means price moved back against
the triggering move.

Open interest is used in two separate ways:

- `oi_change_1h_realtime_pct`: one-hour change up to the latest 5-minute
  snapshot at least five minutes before the trigger. It is known at decision time.
- `oi_change_episode_label_pct`: change from before the move to 15 minutes
  after the trigger. It is an **ex-post label** for understanding episode type,
  not a tradable feature.

June–August 2026 are excluded; late-May paths are truncated rather than
extended into holdout data.
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
from IPython.display import Markdown, display
get_ipython().run_line_magic('matplotlib', 'inline')

REPO = Path('/Users/petermillington/Research/market-impact-research')
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
INK, MUTED = '#2b2b2b', '#8a8a85'
RNG = np.random.default_rng(20_260_929)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 115,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.color': '#e6e6e3',
    'lines.linewidth': 2,
})
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_float_precision(2)


def day_bootstrap(values, days, reps=2000):
    # Mean and 95% interval resampling whole UTC days of episodes.
    values = np.asarray(values, dtype=float)
    days = np.asarray(days)
    keep = np.isfinite(values)
    values, days = values[keep], days[keep]
    if len(values) < 3:
        return np.nan, np.nan, np.nan
    unique, inverse = np.unique(days, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    draws = RNG.integers(0, len(unique), size=(reps, len(unique)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return values.mean(), *np.quantile(means, [0.025, 0.975])
"""
    ),
    md("## Manual controls"),
    code(
        r"""
EXPERIMENT = 'stress_episodes_v2'   # v1: 30s-5m spike triggers; v2: 15m/60m moves
TARGET_PER_YEAR = 100      # 50, 100 or 200 calibrated episodes per year
PERIOD = 'test'            # 'test' (2023-01..2026-05) or 'calibration' (2021-22)
HOLDING_SECONDS = 900      # 300, 900, 1800 or 3600
RULE = 'trigger'           # trigger, delay_Ns, exhaustion_30s or exhaustion_60s

RESULTS = REPO / 'alpha-search' / 'reports' / EXPERIMENT
config = json.loads(
    (REPO / 'alpha-search' / 'configs' / f'{EXPERIMENT}.json').read_text()
)
manifest = json.loads((RESULTS / 'run_manifest.json').read_text())
episodes_all = pl.read_parquet(RESULTS / 'episodes.parquet').with_columns(
    pl.from_epoch('trigger_second').dt.date().alias('day'),
    pl.from_epoch('trigger_second').dt.year().alias('year'),
)
paths_all = pl.read_parquet(RESULTS / 'paths.parquet')
trades_all = pl.read_parquet(RESULTS / 'trades.parquet')
TAKER = 2 * config['costs_bps_per_side']['taker']

assert TARGET_PER_YEAR in config['target_episodes_per_year']
assert PERIOD in {'test', 'calibration'}
assert HOLDING_SECONDS in config['holding_horizons_seconds']

episodes = episodes_all.filter(
    (pl.col('target_per_year') == TARGET_PER_YEAR) & (pl.col('period') == PERIOD)
)
keys = episodes.select('trigger_second', 'day', 'direction', 'scheduled_release_window',
                       'oi_change_episode_label_pct', 'oi_change_1h_realtime_pct',
                       'aggression_share', 'spread_proxy_bps', 'year')
paths = paths_all.filter(pl.col('target_per_year') == TARGET_PER_YEAR).join(
    keys, on='trigger_second'
)
trades = trades_all.filter(pl.col('target_per_year') == TARGET_PER_YEAR).join(
    keys, on='trigger_second'
).with_columns(
    (pl.col('gross_bps') - TAKER - pl.col('entry_spread_proxy_bps')).alias('net_bps')
)
display(Markdown(
    f"`{EXPERIMENT}`: windows {config['trigger_windows_seconds']} s; "
    f"threshold z = {manifest['thresholds_z'][str(TARGET_PER_YEAR)]:.2f}; "
    f"{episodes.height} {PERIOD} episodes on {episodes['day'].n_unique()} days."
))
"""
    ),
    md("## 1. Episode catalogue"),
    code(
        r"""
episodes_all.filter(pl.col('target_per_year') == TARGET_PER_YEAR).group_by('year').agg(
    pl.len().alias('episodes'),
    (pl.col('direction') < 0).sum().alias('down_moves'),
    pl.col('scheduled_release_window').sum().alias('scheduled_release'),
    pl.col('move_bps').median().alias('median_trigger_move_bps'),
    pl.col('oi_change_episode_label_pct').median().alias('median_oi_label_pct'),
).sort('year')
"""
    ),
    code(
        r"""
episodes.sort('peak_z', descending=True).select(
    'trigger_time_utc', 'direction', 'trigger_z', 'peak_z', 'move_bps',
    'aggression_share', 'deep_intensity_ratio', 'spread_proxy_bps',
    'scheduled_release_window', 'oi_change_1h_realtime_pct',
    'oi_change_episode_label_pct', 'adverse_excursion_300s_bps'
).head(20)
"""
    ),
    md(
        r"""
## 2. Average reversion path around the trigger

Offsets before zero show the aligned run-up (negative reversion = price moving in
the trigger direction). Shading is a 95% day-block bootstrap interval.
"""
    ),
    code(
        r"""
def plot_paths(frame, groups, title):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for colour, (label, condition) in zip(SERIES, groups):
        subset = frame.filter(condition)
        rows = []
        for offset in sorted(subset['offset_seconds'].unique()):
            chosen = subset.filter(pl.col('offset_seconds') == offset)
            mean, low, high = day_bootstrap(chosen['reversion_bps'], chosen['day'])
            rows.append((offset, mean, low, high))
        rows = np.array(rows, dtype=float)
        count = subset['trigger_second'].n_unique()
        ax.plot(rows[:, 0] / 60, rows[:, 1], color=colour, label=f'{label} (n={count})')
        ax.fill_between(rows[:, 0] / 60, rows[:, 2], rows[:, 3], color=colour, alpha=0.15,
                        linewidth=0)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle=':')
    ax.set_xlabel('minutes from trigger')
    ax.set_ylabel('mean reversion from trigger price (bps)')
    ax.set_title(title, loc='left')
    ax.legend(frameon=False, loc='lower right')
    plt.show()

plot_paths(paths, [('all episodes', pl.lit(True))], f'All {PERIOD} episodes')
plot_paths(paths, [
    ('down moves', pl.col('direction') < 0),
    ('up moves', pl.col('direction') > 0),
], 'By direction of the triggering move')
plot_paths(paths, [
    ('scheduled release window', pl.col('scheduled_release_window')),
    ('unscheduled', ~pl.col('scheduled_release_window')),
], 'Scheduled macro-release timing versus other episodes')
plot_paths(paths, [
    ('open interest fell (ex-post)', pl.col('oi_change_episode_label_pct') < 0),
    ('open interest rose (ex-post)', pl.col('oi_change_episode_label_pct') >= 0),
], 'Forced versus new positioning (ex-post open-interest label)')
"""
    ),
    md(
        r"""
## 3. How far the move runs after the trigger

Further movement in the trigger direction after the trigger. This is the risk
an early contrarian entry must survive.
"""
    ),
    code(
        r"""
columns = [f'adverse_excursion_{h}s_bps' for h in config['adverse_excursion_horizons_seconds']]
episodes.select(
    *[pl.col(c).median().alias(f'median_{c}') for c in columns],
    *[pl.col(c).quantile(0.9).alias(f'p90_{c}') for c in columns],
).transpose(include_header=True, header_name='statistic', column_names=['bps'])
"""
    ),
    md(
        r"""
## 4. Entry rules

Each rule enters in the reversion direction and holds for a fixed time. `net_bps`
subtracts the taker round trip and the trailing 30-second spread proxy at entry,
a rough allowance for stress-time spread. Intervals resample UTC days.
"""
    ),
    code(
        r"""
rows = []
for (rule, holding), frame in trades.group_by(['rule', 'holding_seconds']):
    gross = day_bootstrap(frame['gross_bps'], frame['day'])
    net = day_bootstrap(frame['net_bps'], frame['day'])
    rows.append({
        'rule': rule, 'holding_seconds': holding, 'trades': frame.height,
        'median_wait_s': frame['entry_wait_seconds'].median(),
        'mean_gross_bps': gross[0], 'mean_net_bps': net[0],
        'net_ci_low': net[1], 'net_ci_high': net[2],
        'median_net_bps': frame['net_bps'].median(),
        'hit_rate_net': (frame['net_bps'] > 0).mean(),
    })
rule_table = pl.DataFrame(rows).sort('holding_seconds', 'mean_net_bps',
                                      descending=[False, True])
rule_table
"""
    ),
    code(
        r"""
view = rule_table.filter(pl.col('holding_seconds') == HOLDING_SECONDS).sort('mean_net_bps')
fig, ax = plt.subplots(figsize=(8, 0.45 * view.height + 1.2))
positions = np.arange(view.height)
ax.barh(positions, view['mean_net_bps'], color=SERIES[0], height=0.6)
ax.errorbar(view['mean_net_bps'], positions,
            xerr=[view['mean_net_bps'] - view['net_ci_low'],
                  view['net_ci_high'] - view['mean_net_bps']],
            fmt='none', ecolor=INK, elinewidth=1, capsize=3)
ax.set_yticks(positions, view['rule'])
ax.axvline(0, color=MUTED, linewidth=1)
ax.set_xlabel('mean net reversion per trade (bps)')
ax.set_title(f'Entry rules, {HOLDING_SECONDS // 60}-minute hold, {PERIOD} episodes',
             loc='left')
ax.grid(axis='y', visible=False)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
## 5. Which episodes revert?

Mean net edge for the selected rule and holding period, split by features known
at the trigger (and, separately marked, the ex-post open-interest label).
Tercile cut points come from calibration episodes only.
"""
    ),
    code(
        r"""
selected = trades.filter(
    (pl.col('rule') == RULE) & (pl.col('holding_seconds') == HOLDING_SECONDS)
)
calibration = episodes_all.filter(
    (pl.col('target_per_year') == TARGET_PER_YEAR) & (pl.col('period') == 'calibration')
)

def split_rows(feature, label):
    cuts = calibration[feature].drop_nans().drop_nulls().quantile(1 / 3), \
        calibration[feature].drop_nans().drop_nulls().quantile(2 / 3)
    output = []
    for name, condition in [
        ('low', pl.col(feature) < cuts[0]),
        ('middle', (pl.col(feature) >= cuts[0]) & (pl.col(feature) < cuts[1])),
        ('high', pl.col(feature) >= cuts[1]),
    ]:
        frame = selected.filter(condition)
        mean, low, high = day_bootstrap(frame['net_bps'], frame['day'])
        output.append({'feature': label, 'group': name, 'trades': frame.height,
                       'mean_net_bps': mean, 'ci_low': low, 'ci_high': high})
    return output

rows = []
for feature, label in [
    ('oi_change_1h_realtime_pct', 'OI change 1h (real time)'),
    ('aggression_share', 'aggression share'),
    ('spread_proxy_bps', 'spread proxy'),
    ('oi_change_episode_label_pct', 'OI change over episode (EX-POST)'),
]:
    rows.extend(split_rows(feature, label))
for name, condition in [
    ('scheduled release', pl.col('scheduled_release_window')),
    ('unscheduled', ~pl.col('scheduled_release_window')),
    ('down move', pl.col('direction') < 0),
    ('up move', pl.col('direction') > 0),
]:
    frame = selected.filter(condition)
    mean, low, high = day_bootstrap(frame['net_bps'], frame['day'])
    rows.append({'feature': 'category', 'group': name, 'trades': frame.height,
                 'mean_net_bps': mean, 'ci_low': low, 'ci_high': high})
pl.DataFrame(rows)
"""
    ),
    md("## 6. Stability by year"),
    code(
        r"""
selected.group_by('year').agg(
    pl.len().alias('trades'),
    pl.col('gross_bps').mean().alias('mean_gross_bps'),
    pl.col('net_bps').mean().alias('mean_net_bps'),
    pl.col('net_bps').median().alias('median_net_bps'),
    (pl.col('net_bps') > 0).mean().alias('hit_rate'),
).sort('year')
"""
    ),
    md(
        r"""
## Reading

See `alpha-search/reports/stress-episodes-v1-initial-results.md` for the
written interpretation.
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
