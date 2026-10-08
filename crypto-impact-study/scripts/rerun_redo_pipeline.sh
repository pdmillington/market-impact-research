#!/usr/bin/env bash
# Rerun every redo analysis, in order, one heavy job at a time (24 GB machine).
# Assumes the shared data layer (events, paper_events_v2, pb_blocks_v1) is current.
# A failing step is logged and the pipeline continues; check the log at the end.
#
#   bash crypto-impact-study/scripts/rerun_redo_pipeline.sh
#
# Log: crypto-impact-study/reports/redo/rerun.log

set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO/shared-methodology/.venv/bin/python"
S="$REPO/crypto-impact-study/scripts"
NB="$REPO/crypto-impact-study/notebooks"
LOG="$REPO/crypto-impact-study/reports/redo/rerun.log"
cd "$REPO"
echo "== rerun started $(date -u '+%Y-%m-%d %H:%M:%S') UTC" > "$LOG"

step() {  # step <label> <command...>
  local label="$1"; shift
  local start=$SECONDS
  echo "-- $label: started $(date -u '+%H:%M:%S')" >> "$LOG"
  if "$@" >> "$LOG.detail" 2>&1; then
    echo "-- $label: ok ($((SECONDS - start)) s)" >> "$LOG"
  else
    echo "-- $label: FAILED ($((SECONDS - start)) s); see rerun.log.detail" >> "$LOG"
  fi
}
: > "$LOG.detail"

# 1. The paper's P&B collapse: bridge (paper period), pooled primary sample, by year.
step "P&B bridge 2025-01..2026-05" "$PY" -W ignore "$S/run_pb_collapse.py" --start 2025-01 --end 2026-05 --tag bridge_2025_01_2026_05
step "bridge comparison" "$PY" -W ignore "$S/bridge_comparison.py"
step "P&B pooled 2021-01..2026-08" "$PY" -W ignore "$S/run_pb_collapse.py" --start 2021-01 --end 2026-08 --tag pb_collapse_2021_01_2026_08
for year in 2021 2022 2023 2024 2025; do
  step "P&B $year" "$PY" -W ignore "$S/run_pb_collapse.py" --start "$year-01" --end "$year-12" --tag "pb_collapse_year_$year"
done

# 2. Activity surface, diagnostics, thesis figures 26-29.
step "activity surface" "$PY" -W ignore "$S/run_activity_surface.py"
step "surface diagnostics" "$PY" -W ignore "$S/diagnose_activity_surface.py"
step "surface figures" "$PY" -W ignore "$S/make_surface_figures.py"

# 3. Data audits.
step "event construction audit" "$PY" -W ignore "$S/event_construction_audit.py"
step "data description" "$PY" -W ignore "$S/data_description_stats.py"
step "trading gaps audit" "$PY" -W ignore "$S/trading_gaps_audit.py"

# 4. Paper notebooks not yet repointed: run through the runner (outputs not saved in the notebook).
step "runner notebooks" "$PY" -W ignore "$S/run_paper_notebooks.py" BarConstructionRobustness RegimeResponseCurves \
  StateDependentEffectiveLiquidity ImpactDecayAndLeadLag PatzeltBouchaudScaling

# 5. Repointed notebooks: execute in place so their saved outputs are current.
"$PY" "$S/build_activity_surface_notebook.py" >> "$LOG.detail" 2>&1
for name in PowerLawImpact ImbalanceResponseSymmetry VolatilityDayInfluence ActivitySurface; do
  step "notebook $name" env MPLBACKEND=Agg "$REPO/shared-methodology/.venv/bin/jupyter" nbconvert --to notebook \
    --execute --inplace --ExecutePreprocessor.timeout=-1 "$NB/$name.ipynb"
done

echo "== rerun finished $(date -u '+%Y-%m-%d %H:%M:%S') UTC" >> "$LOG"
