"""Re-run the market-impact paper notebooks on the corrected, extended data.

Each notebook's own code runs unchanged except that:
- DATA_DIR points at the corrected monthly event files (export_paper_events.py),
  2021-01 to 2026-08;
- figures and tables are written to reports/redo/paper/{figures,tables}, so the
  paper's original outputs are untouched;
- plots render off-screen (Agg) and IPython magics are skipped.

Each notebook runs in its own process (memory is released between notebooks).
A log of every cell (time, failure) is written to reports/redo/paper/run_log.json.

    python crypto-impact-study/scripts/run_paper_notebooks.py               # all chapter notebooks
    python crypto-impact-study/scripts/run_paper_notebooks.py PowerLawImpact
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback


REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "crypto-impact-study"
NOTEBOOKS = ("PowerLawImpact", "BarConstructionRobustness", "RegimeResponseCurves",
             "StateDependentEffectiveLiquidity", "ImpactDecayAndLeadLag", "PatzeltBouchaudScaling")
DATA_DIR = REPO / "shared-data" / "features" / "academic" / "symbol=BTCUSDT" / "paper_events_v2"
OUT = STUDY / "reports" / "redo" / "paper"
PB_CACHE = REPO / "shared-data" / "features" / "academic" / "symbol=BTCUSDT" / "pb_notebook_cache_v2"

DATA_DIR_PATTERN = re.compile(r"DATA_DIR\s*=\s*(?:\([^)]*?BTCUSDT_monthly['\"]\s*\)|[^\n]*BTCUSDT_monthly['\"]\)?)", re.S)
FIGURE_PATTERN = re.compile(r"PROJECT_DIR\s*/\s*['\"]reports['\"]\s*/\s*['\"]figures['\"]")
TABLE_PATTERN = re.compile(r"PROJECT_DIR\s*/\s*['\"]reports['\"]\s*/\s*['\"]tables['\"]")

# Cells skipped because they only work out of order in the saved notebook and
# produce nothing used later (checked 2026-10-05).
SKIP_CELLS = {
    "BarConstructionRobustness": {3},   # zero-range diagnostic using bars_n from a later cell
}

SETUP = """
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None
def display(obj, *args, **kwargs):
    print(getattr(obj, "shape", ""), type(obj).__name__)
"""


def environment() -> dict:
    env = dict(os.environ)
    env.update({
        "CRYPTO_IMPACT_PROJECT_DIR": str(STUDY),
        "CRYPTO_IMPACT_FIGURE_DIR": str(OUT / "figures"),
        "CRYPTO_IMPACT_TABLE_DIR": str(OUT / "tables"),
        "CRYPTO_IMPACT_PB_OUTPUT_ROOT": str(OUT),
        "CRYPTO_IMPACT_PB_CACHE_DIR": str(PB_CACHE),
        "MPLBACKEND": "Agg",
    })
    return env


def transformed_cells(name: str) -> list[str]:
    notebook = json.loads((STUDY / "notebooks" / f"{name}.ipynb").read_text())
    cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
        source = DATA_DIR_PATTERN.sub(f"DATA_DIR = Path({str(DATA_DIR)!r})", source)
        source = FIGURE_PATTERN.sub(f"Path({str(OUT / 'figures')!r})", source)
        source = TABLE_PATTERN.sub(f"Path({str(OUT / 'tables')!r})", source)
        cells.append(source)
    return cells


def run_one(name: str) -> dict:
    """Execute one notebook in this process (called in a child process)."""

    os.chdir(STUDY / "notebooks")
    sys.path[:0] = [str(STUDY), str(STUDY / "src")]
    namespace: dict = {"__name__": "__main__"}
    exec(SETUP, namespace)
    log = {"notebook": name, "cells": []}
    for index, source in enumerate(transformed_cells(name)):
        if index in SKIP_CELLS.get(name, set()):
            log["cells"].append({"cell": index, "seconds": 0.0, "status": "skipped"})
            print(f"{name} cell {index}: skipped (out-of-order diagnostic)", flush=True)
            continue
        start = time.time()
        try:
            exec(compile(source, f"{name}[{index}]", "exec"), namespace)
            status = "ok"
        except Exception:
            status = "failed"
            log["error"] = {"cell": index, "traceback": traceback.format_exc()[-3000:]}
        namespace["plt"].close("all")
        log["cells"].append({"cell": index, "seconds": round(time.time() - start, 1), "status": status})
        print(f"{name} cell {index}: {status} ({time.time() - start:.0f}s)", flush=True)
        if status == "failed":
            break
    log["completed"] = "error" not in log
    return log


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        print("CHILD-RESULT " + json.dumps(run_one(sys.argv[2])), flush=True)
        return
    names = sys.argv[1:] or list(NOTEBOOKS)
    for directory in (OUT / "figures", OUT / "tables", PB_CACHE):
        directory.mkdir(parents=True, exist_ok=True)
    if not any(DATA_DIR.glob("BTCUSDT-trades-*.parquet")):
        raise SystemExit(f"No corrected event files in {DATA_DIR}; run export_paper_events.py first.")
    log_path = OUT / "run_log.json"
    logs = json.loads(log_path.read_text()) if log_path.exists() else {}
    for name in names:
        process = subprocess.run([sys.executable, __file__, "--child", name], env=environment(),
                                 capture_output=True, text=True)
        sys.stdout.write(process.stdout[-4000:])
        result = next((json.loads(line[len("CHILD-RESULT "):]) for line in process.stdout.splitlines()
                       if line.startswith("CHILD-RESULT ")), None)
        if result is None:
            result = {"notebook": name, "completed": False, "error": {"stderr": process.stderr[-3000:]}}
        logs[name] = result
        log_path.write_text(json.dumps(logs, indent=2))
        print(f"== {name}: {'completed' if result.get('completed') else 'FAILED'}", flush=True)


if __name__ == "__main__":
    main()
