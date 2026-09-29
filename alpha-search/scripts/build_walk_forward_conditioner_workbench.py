"""Generate a manual notebook for any walk-forward conditioner result directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result_root = args.result_root.expanduser().resolve()
    manifest = json.loads((result_root / "run_manifest.json").read_text())
    first_conditioner = manifest["conditioners"][0]
    first_size = manifest["event_bar_sizes"][0]
    first_horizon = manifest["clock_horizons_minutes"][0]

    cells = [
        md(
            """
# Walk-forward conditioner workbench

This notebook visualizes one completed generic conditioner screen. It does not
select a winning hypothesis automatically. Every fit, cut point, and test result
comes from chronological folds recorded in the run manifest.
"""
        ),
        md("## Manual controls"),
        code(
            f"""
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

RESULT_ROOT = Path({str(result_root)!r})
BAR_SIZE = {first_size}
HORIZON_MINUTES = {first_horizon}
SIDE = 'sell'
CONDITIONER = {first_conditioner!r}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({{'figure.dpi': 115, 'axes.spines.top': False, 'axes.spines.right': False}})
"""
        ),
        md("## Load the frozen run"),
        code(
            """
manifest = json.loads((RESULT_ROOT / 'run_manifest.json').read_text())
fits = pl.read_csv(RESULT_ROOT / 'impact_fits.csv')
signals = pl.read_csv(RESULT_ROOT / 'fold_signal_summary.csv')
increments = pl.read_csv(RESULT_ROOT / 'fold_conditioner_increment.csv')
grid = pl.read_csv(RESULT_ROOT / 'fold_conditioning_grid.csv')
cuts = pl.read_csv(RESULT_ROOT / 'fold_feature_cuts.csv')
aggregate_increments = pl.read_csv(RESULT_ROOT / 'aggregate_conditioner_increment.csv')

display({
    'data_fingerprint': manifest['data_fingerprint'],
    'event_bar_sizes': manifest['event_bar_sizes'],
    'clock_horizons_minutes': manifest['clock_horizons_minutes'],
    'primary_feature': manifest['primary_feature'],
    'conditioners': manifest['conditioners'],
    'folds': len(manifest['folds']),
})
display(pl.DataFrame(manifest['folds']))
"""
        ),
        md("## Impact-fit drift across folds"),
        code(
            """
part = fits.filter(pl.col('construction') == f'event_{BAR_SIZE}').sort('fold')
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for side, colour in [('sell', '#c2413b'), ('buy', '#13795b')]:
    chosen = part.filter(pl.col('side') == side)
    axes[0].plot(chosen['fold'], chosen['intercept'], marker='o', color=colour, label=side)
    axes[1].plot(chosen['fold'], chosen['slope'], marker='o', color=colour)
    axes[2].plot(chosen['fold'], chosen['imbalance_scale'], marker='o', color=colour)
for ax, title in zip(axes, ['intercept', 'log slope', 'imbalance scale']):
    ax.set_title(title)
    ax.set_xlabel('walk-forward fold')
axes[0].legend()
fig.suptitle(f'event_{BAR_SIZE}: training-only impact calibration')
plt.tight_layout()
plt.show()
"""
        ),
        md("## Training cut-point drift"),
        code(
            """
part = cuts.filter(
    (pl.col('construction') == f'event_{BAR_SIZE}') &
    (pl.col('feature') == CONDITIONER) &
    (pl.col('side') == (-1 if SIDE == 'sell' else 1))
).sort('fold', 'cut_number')
fig, ax = plt.subplots(figsize=(9, 4.5))
for cut_number in sorted(part['cut_number'].unique().to_list()):
    chosen = part.filter(pl.col('cut_number') == cut_number)
    ax.plot(chosen['fold'], chosen['cut_value'], marker='o', label=f'cut {cut_number}')
ax.set_xlabel('walk-forward fold')
ax.set_ylabel('training-fitted cut value')
ax.set_title(f'{CONDITIONER}: threshold drift')
ax.legend(ncol=2)
plt.show()
"""
        ),
        md("## Plain and conditioned rule by test fold"),
        code(
            """
part = signals.filter(
    (pl.col('construction') == f'event_{BAR_SIZE}') &
    (pl.col('horizon_minutes') == HORIZON_MINUTES) &
    (pl.col('side') == SIDE) &
    (pl.col('conditioner') == CONDITIONER)
).sort('fold')
fig, ax = plt.subplots(figsize=(10, 4.8))
for strategy, colour, marker in [
    ('primary_reversion_q_high', '#64748b', 'o'),
    ('primary_reversion_conditioner_q_high', '#7c3aed', 's'),
]:
    chosen = part.filter(pl.col('strategy') == strategy)
    ax.plot(chosen['fold'], chosen['day_mean_bps'], marker=marker, color=colour,
            label=('conditioned' if 'conditioner' in strategy else 'plain'))
ax.axhline(0, color='black', lw=.8)
ax.set_xlabel('out-of-sample fold')
ax.set_ylabel('equal-weight daily conditional return (bps)')
ax.set_title(f'event_{BAR_SIZE}; {SIDE}; {HORIZON_MINUTES} minutes; {CONDITIONER}')
ax.legend()
plt.show()
display(part.select('fold', 'test_start', 'test_end', 'strategy', 'signal_rate',
                    'day_mean_bps', 'day_t_stat'))
"""
        ),
        md("## Conditioner comparison across folds"),
        code(
            """
summary = aggregate_increments.filter(
    (pl.col('construction') == f'event_{BAR_SIZE}') &
    (pl.col('horizon_minutes') == HORIZON_MINUTES) &
    (pl.col('side') == SIDE) &
    (pl.col('comparison') == 'high_conditioner_minus_unconditioned')
).sort('median_day_mean_difference_bps')
display(summary)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(summary['conditioner'], summary['median_day_mean_difference_bps'],
        color=np.where(summary['median_day_mean_difference_bps'].to_numpy() >= 0,
                       '#13795b', '#c2413b'))
ax.axvline(0, color='black', lw=.8)
ax.set_xlabel('median conditioned-minus-plain return across folds (bps)')
ax.set_title('Prespecified conditioner comparison')
plt.tight_layout()
plt.show()
"""
        ),
        md("## Selected fold surfaces"),
        code(
            """
folds = sorted(grid['fold'].unique().to_list())
columns = min(3, len(folds))
rows = int(np.ceil(len(folds) / columns))
fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 4.2 * rows), squeeze=False)
matrices = []
for fold in folds:
    chosen = grid.filter(
        (pl.col('construction') == f'event_{BAR_SIZE}') &
        (pl.col('horizon_minutes') == HORIZON_MINUTES) &
        (pl.col('side') == SIDE) &
        (pl.col('conditioner') == CONDITIONER) &
        (pl.col('fold') == fold)
    )
    matrix = np.full((5, 5), np.nan)
    for row in chosen.iter_rows(named=True):
        matrix[row['conditioner_quantile'] - 1, row['primary_quantile'] - 1] = row['day_mean_bps']
    matrices.append(matrix)
limit = max(float(np.nanmax(np.abs(matrix))) for matrix in matrices)
for ax, fold, matrix in zip(axes.flat, folds, matrices):
    image = ax.imshow(matrix, origin='lower', cmap='RdBu_r', vmin=-limit, vmax=limit)
    ax.set_title(f'fold {fold}')
    ax.set_xlabel('primary quantile')
    ax.set_ylabel('conditioner quantile')
    ax.set_xticks(range(5), range(1, 6))
    ax.set_yticks(range(5), range(1, 6))
for ax in axes.flat[len(folds):]:
    ax.axis('off')
fig.colorbar(image, ax=axes, label='return in reversion direction (bps)')
fig.suptitle(f'{CONDITIONER}: fold-by-fold conditioning surface')
plt.show()
"""
        ),
        md(
            """
## Interpretation discipline

- Treat nearby event-bar sizes and horizons as correlated robustness checks, not independent discoveries.
- Inspect threshold drift and signal frequency before interpreting return differences.
- Do not promote a conditioner on pooled means alone; require directionally consistent chronological folds.
- Profitability, overlap, execution costs and position sizing remain a separate stage after a rule is frozen.
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
