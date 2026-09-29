# Alpha-search tooling workflow

The alpha layer is designed to move with `shared-data`. The source, canonical and
event layers may live on an external SSD; code, experiment manifests, notebooks
and compact result tables may remain inside the repository.

## 1. Audit a data root

The audit is read-only. It checks coverage, calendar gaps, source-archive presence,
the first and last event schemas, disk space and a portable data fingerprint.

```bash
./shared-methodology/.venv/bin/python \
  alpha-search/scripts/audit_alpha_data.py \
  --shared-data-root "/Volumes/YOUR_SSD/shared-data" \
  --require-contiguous
```

The fingerprint excludes absolute paths. It should therefore remain identical
when an unchanged dataset is copied from the internal drive to the SSD.

## 2. Continue ingestion with the shared parser

Do not create an alpha-specific downloader or parser. Continue using the shared
command, changing only the root:

```bash
./shared-methodology/.venv/bin/crypto-market-data download-and-parse \
  --root "/Volumes/YOUR_SSD/shared-data" \
  --symbol BTCUSDT \
  --start-month YYYY-MM \
  --end-month YYYY-MM
```

Run `status` and the alpha audit after ingestion. Preserve the downloaded ZIPs as
audit evidence.

## 3. Build portable short-event caches

```bash
./shared-methodology/.venv/bin/python \
  alpha-search/scripts/build_short_event_bars.py \
  --shared-data-root "/Volumes/YOUR_SSD/shared-data" \
  --sizes 200 500 1000 \
  --overwrite
```

By default the caches are written under the selected data root at
`features/alpha/symbol=BTCUSDT/short_event_screen_v1`. Use `--output-root` only
when a different cache location is deliberate.

The builder records the input-data fingerprint in `cache_manifest.json`. It will
not silently reuse a legacy cache or a cache built before additional months were
ingested. After extending coverage, rebuild all existing bar sizes together with
`--overwrite`. The current cache format is consolidated rather than incrementally
appendable; safe incremental construction would require explicit state for the
incomplete event bar at the old dataset boundary.

## 4. Freeze an experiment manifest

Copy or edit `configs/full_history_template.json`, then create a manifest from the
actual coverage found on disk:

```bash
./shared-methodology/.venv/bin/python \
  alpha-search/scripts/prepare_alpha_experiment.py \
  --config alpha-search/configs/full_history_template.json \
  --shared-data-root "/Volumes/YOUR_SSD/shared-data" \
  --output alpha-search/experiments/full_history_v1/manifest.json
```

The manifest records the data fingerprint, coverage, exact configuration and
half-open chronological folds. Rolling or expanding training windows are selected
in the JSON configuration. An embargo is recorded separately and applied before
each fit.

## 5. Run a generic conditioner screen

```bash
./shared-methodology/.venv/bin/python \
  alpha-search/scripts/run_walk_forward_conditioners.py \
  --manifest alpha-search/experiments/full_history_v1/manifest.json \
  --output-root alpha-search/experiments/full_history_v1/conditioner_results
```

The runner fits buy and sell impact curves separately within every training fold,
freezes primary-feature and conditioner cut points, and evaluates only the next
test window. Conditioner names are set in the configuration rather than in the
script.

The runner verifies that the bar-cache fingerprint matches the experiment
manifest. `--allow-unverified-cache` exists only for deliberate checks of older
caches created before manifests were introduced; it should not be used for the new
full-history experiment.

Outputs include fold-level fits, cut points, conditional surfaces, signal summaries,
paired conditioner increments and compact across-fold summaries. They are research
diagnostics, not portfolio returns.

## 6. Generate a manual workbench

```bash
./shared-methodology/.venv/bin/python \
  alpha-search/scripts/build_walk_forward_conditioner_workbench.py \
  --result-root alpha-search/experiments/full_history_v1/conditioner_results \
  --output alpha-search/notebooks/tools/FullHistoryConditionerWorkbench.ipynb
```

The notebook exposes bar size, horizon, side and conditioner controls. It shows
impact-fit drift, training-threshold drift, fold-by-fold plain versus conditioned
returns, conditioner comparisons and two-dimensional fold surfaces.
It is a generic diagnostic generated under `notebooks/tools`, not a replacement
for the canonical scale-free impact workbench.

## 7. Detailed event decay

The event-level decay tool accepts the same portable root and writes compact
daily summaries:

```bash
./shared-methodology/.venv/bin/python \
  alpha-search/scripts/run_event_decay_by_execution.py \
  --shared-data-root "/Volumes/YOUR_SSD/shared-data" \
  --output-root alpha-search/reports/event_decay_sweep_v2
```

The current v2 runner uses realized endpoint sweep as the primary aggression
classification and includes 100/250/500-millisecond horizons. Its
default calibration ends in December 2022 and its default analysis ends in May
2026, preserving June--August 2026 as a holdout:

The earlier fill/price-level output in `reports/event_decay_v1` is retained as
a versioned historical result, not overwritten.

## 7a. Short-bar impact-function cache

Build only the bar constructions needed for the next manual comparison. The
default cache excludes June--August 2026:

```bash
./shared-methodology/.venv/bin/python \
  alpha-search/scripts/build_impact_research_bars.py \
  --shared-data-root "/Volumes/YOUR_SSD/shared-data" \
  --event-sizes 200 500 1000 \
  --time-bars 30s 1m 2m 5m
```

Fifty- and 100-event bars can be added after inspecting the duration atlas.
They are supported but omitted from the default build because they create much
larger caches.

## 8. Multiscale impact residuals

The multiscale experiment separates the decision-bar clock from the trailing event
window over which impact is measured. Start with
`configs/sandbox_impact_residual_multiscale_v1.json`. Its `same_bar`, 1,000-event
and 4,000-event windows are resolved separately for every decision-bar size.

Generate the experiment manifest in the usual way, then run:

```bash
./shared-methodology/.venv/bin/python \
  alpha-search/scripts/run_multiscale_impact_residual.py \
  --manifest alpha-search/experiments/EXPERIMENT/manifest.json \
  --output-root alpha-search/experiments/EXPERIMENT/results
```

Each measurement scale receives its own buy/sell impact fit inside every training
fold. The fit sample contains globally aligned, non-overlapping blocks at that
measurement scale. The frozen curve is then applied to rolling measurements at
every decision-bar close. Longer-window residuals are recomputed from aggregate
signed flow and price response; smaller-window residuals are never added together.

For the fit-family comparison, run each configured decision/measurement pair as a
separate bounded diagnostic job. For example:

```bash
./shared-methodology/.venv/bin/python \
  alpha-search/scripts/run_impact_fit_diagnostics.py \
  --manifest alpha-search/experiments/EXPERIMENT/manifest.json \
  --output-root alpha-search/experiments/EXPERIMENT/results \
  --decision-events 200 \
  --measurement-events 4000
```

The diagnostic compares the raw-observation affine log curve, a zero-origin log
curve fitted to equal-count bin means, a binned power law and a monotonic benchmark.
Test-period calibration uses the next chronological fold with no refitting and with
the training imbalance bins frozen.

Generate the visual workbench with:

```bash
./shared-methodology/.venv/bin/python \
  alpha-search/scripts/build_multiscale_impact_residual_workbench.py \
  --result-root alpha-search/experiments/EXPERIMENT/results \
  --output alpha-search/notebooks/archive/MultiscaleImpactResidualWorkbench.ipynb
```

This is a historical benchmark path. New residual work should use the rolling
activity-adjusted scale-free model documented in
`notebooks/PBScaleFreeImpactWorkbench.ipynb`.
