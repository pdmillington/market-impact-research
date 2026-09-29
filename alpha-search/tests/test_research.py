from __future__ import annotations

import json
from pathlib import Path

from alpha_search.research import (
    add_months,
    coverage_summary,
    data_fingerprint,
    discover_validated_partitions,
    walk_forward_folds,
)


def _write_report(root: Path, month: str, *, event_rows: int) -> None:
    year, number = month.split("-")
    canonical = root / "canonical" / "fills" / "symbol=BTCUSDT" / f"year={year}" / f"month={number}" / "part-000.parquet"
    event = root / "events" / "timestamp_direction_v1" / "symbol=BTCUSDT" / f"year={year}" / f"month={number}" / "part-000.parquet"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    event.parent.mkdir(parents=True, exist_ok=True)
    canonical.touch()
    event.touch()
    report = root / "metadata" / "BTCUSDT" / f"{month}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "symbol": "BTCUSDT", "month": month,
        "fill_rows": event_rows * 2, "event_rows": event_rows,
        "source_bytes": 10, "canonical_bytes": 20, "event_bytes": 30,
        "event_rule": "timestamp_direction_v1",
    }))


def test_discovery_uses_new_root_instead_of_metadata_paths(tmp_path: Path) -> None:
    _write_report(tmp_path, "2020-01", event_rows=7)
    partitions = discover_validated_partitions(tmp_path)

    assert partitions[0].event_path.is_relative_to(tmp_path)
    assert coverage_summary(partitions)["event_rows"] == 7
    assert len(data_fingerprint(partitions)) == 64


def test_coverage_reports_calendar_gaps(tmp_path: Path) -> None:
    _write_report(tmp_path, "2020-01", event_rows=7)
    _write_report(tmp_path, "2020-03", event_rows=9)

    summary = coverage_summary(discover_validated_partitions(tmp_path))

    assert summary["missing_months"] == ["2020-02"]
    assert summary["event_rows"] == 16


def test_walk_forward_folds_are_chronological_and_half_open() -> None:
    folds = walk_forward_folds(
        first_month="2020-01", last_month="2021-12",
        minimum_train_months=12, test_months=2, step_months=2,
        rolling_train_months=12, embargo_days=1,
    )

    assert add_months("2020-01", 12) == "2021-01"
    assert folds[0].train_start == "2020-01-01"
    assert folds[0].train_end == folds[0].test_start == "2021-01-01"
    assert folds[0].test_end == "2021-03-01"
    assert folds[-1].test_end == "2022-01-01"
