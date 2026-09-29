"""Freeze data coverage, configuration, and walk-forward folds in a manifest."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))

from alpha_search.research import (  # noqa: E402
    coverage_summary,
    data_fingerprint,
    discover_validated_partitions,
    walk_forward_folds,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shared-data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    configured_root = config.get("shared_data_root")
    if args.shared_data_root is None and not configured_root:
        raise ValueError("Supply --shared-data-root or set shared_data_root in the config.")
    root = (args.shared_data_root or Path(configured_root)).expanduser().resolve()
    symbol = config.get("symbol", "BTCUSDT")
    event_rule = config.get("event_rule", "timestamp_direction_v1")
    partitions = discover_validated_partitions(
        root, symbol=symbol, event_rule=event_rule
    )
    coverage = coverage_summary(partitions)
    if not partitions:
        raise ValueError("No validated partitions were found.")
    if coverage["missing_months"] and config.get("require_contiguous", True):
        raise ValueError(f"Missing calendar months: {coverage['missing_months']}")
    walk = config["walk_forward"]
    folds = walk_forward_folds(
        first_month=coverage["first_month"],
        last_month=coverage["last_month"],
        minimum_train_months=int(walk["minimum_train_months"]),
        test_months=int(walk.get("test_months", 1)),
        step_months=int(walk.get("step_months", 1)),
        rolling_train_months=(
            int(walk["rolling_train_months"])
            if walk.get("rolling_train_months") is not None
            else None
        ),
        embargo_days=int(walk.get("embargo_days", 0)),
    )
    manifest = {
        "manifest_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": config["experiment_name"],
        "shared_data_root": str(root),
        "symbol": symbol,
        "event_rule": event_rule,
        "data_fingerprint": data_fingerprint(partitions),
        "coverage": coverage,
        "config": config,
        "folds": [asdict(fold) for fold in folds],
    }
    text = json.dumps(manifest, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
