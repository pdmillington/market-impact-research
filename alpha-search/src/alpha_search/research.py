"""Data discovery, fingerprints, and chronological research folds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MonthPartition:
    month: str
    report_path: Path
    canonical_path: Path
    event_path: Path
    fill_rows: int
    event_rows: int
    source_bytes: int
    canonical_bytes: int
    event_bytes: int
    event_rule: str


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    embargo_days: int


def _parse_month(value: str) -> date:
    try:
        parsed = date.fromisoformat(f"{value}-01")
    except ValueError as exc:
        raise ValueError(f"Invalid month {value!r}; expected YYYY-MM.") from exc
    if parsed.strftime("%Y-%m") != value:
        raise ValueError(f"Invalid month {value!r}; expected YYYY-MM.")
    return parsed


def add_months(value: str, count: int) -> str:
    parsed = _parse_month(value)
    total = parsed.year * 12 + parsed.month - 1 + int(count)
    year, month_index = divmod(total, 12)
    return f"{year:04d}-{month_index + 1:02d}"


def iter_months(start: str, end: str) -> list[str]:
    """Return inclusive calendar months."""

    if _parse_month(start) > _parse_month(end):
        raise ValueError("start must not be after end.")
    values = []
    current = start
    while current <= end:
        values.append(current)
        current = add_months(current, 1)
    return values


def discover_validated_partitions(
    shared_data_root: Path,
    *,
    symbol: str = "BTCUSDT",
    event_rule: str = "timestamp_direction_v1",
    require_files: bool = True,
) -> list[MonthPartition]:
    """Discover validated months without trusting absolute paths in old metadata."""

    root = Path(shared_data_root).expanduser().resolve()
    reports = sorted((root / "metadata" / symbol).glob("????-??.json"))
    partitions: list[MonthPartition] = []
    for report_path in reports:
        report = json.loads(report_path.read_text())
        month = str(report["month"])
        year, month_number = month.split("-")
        canonical = (
            root
            / "canonical"
            / "fills"
            / f"symbol={symbol}"
            / f"year={year}"
            / f"month={month_number}"
            / "part-000.parquet"
        )
        event = (
            root
            / "events"
            / event_rule
            / f"symbol={symbol}"
            / f"year={year}"
            / f"month={month_number}"
            / "part-000.parquet"
        )
        if report.get("symbol") != symbol:
            raise ValueError(f"Symbol mismatch in {report_path}.")
        if report.get("event_rule") != event_rule:
            raise ValueError(f"Event-rule mismatch in {report_path}.")
        if require_files:
            missing = [path for path in (canonical, event) if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Validated partition {month} is missing: {missing}"
                )
        partitions.append(
            MonthPartition(
                month=month,
                report_path=report_path,
                canonical_path=canonical,
                event_path=event,
                fill_rows=int(report["fill_rows"]),
                event_rows=int(report["event_rows"]),
                source_bytes=int(report["source_bytes"]),
                canonical_bytes=int(report["canonical_bytes"]),
                event_bytes=int(report["event_bytes"]),
                event_rule=event_rule,
            )
        )
    return partitions


def coverage_summary(partitions: list[MonthPartition]) -> dict[str, Any]:
    if not partitions:
        return {
            "months": 0,
            "first_month": None,
            "last_month": None,
            "missing_months": [],
            "fill_rows": 0,
            "event_rows": 0,
            "source_bytes": 0,
            "canonical_bytes": 0,
            "event_bytes": 0,
        }
    months = [partition.month for partition in partitions]
    expected = iter_months(months[0], months[-1])
    return {
        "months": len(partitions),
        "first_month": months[0],
        "last_month": months[-1],
        "missing_months": sorted(set(expected).difference(months)),
        "fill_rows": sum(partition.fill_rows for partition in partitions),
        "event_rows": sum(partition.event_rows for partition in partitions),
        "source_bytes": sum(partition.source_bytes for partition in partitions),
        "canonical_bytes": sum(
            partition.canonical_bytes for partition in partitions
        ),
        "event_bytes": sum(partition.event_bytes for partition in partitions),
    }


def data_fingerprint(partitions: list[MonthPartition]) -> str:
    """Hash portable partition facts, excluding machine-specific root paths."""

    records = [
        {
            "month": p.month,
            "fill_rows": p.fill_rows,
            "event_rows": p.event_rows,
            "source_bytes": p.source_bytes,
            "canonical_bytes": p.canonical_bytes,
            "event_bytes": p.event_bytes,
            "event_rule": p.event_rule,
        }
        for p in partitions
    ]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def walk_forward_folds(
    *,
    first_month: str,
    last_month: str,
    minimum_train_months: int,
    test_months: int = 1,
    step_months: int = 1,
    rolling_train_months: int | None = None,
    embargo_days: int = 0,
) -> list[WalkForwardFold]:
    """Create expanding or rolling calendar-month folds.

    Dates are half-open boundaries.  ``train_end`` equals ``test_start``;
    ``embargo_days`` is stored explicitly for the evaluator to remove the final
    training observations before fitting.
    """

    for name, value in (
        ("minimum_train_months", minimum_train_months),
        ("test_months", test_months),
        ("step_months", step_months),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if rolling_train_months is not None and rolling_train_months < minimum_train_months:
        raise ValueError("rolling_train_months cannot be shorter than minimum_train_months.")
    if embargo_days < 0:
        raise ValueError("embargo_days cannot be negative.")

    final_exclusive = add_months(last_month, 1)
    test_start = add_months(first_month, minimum_train_months)
    folds: list[WalkForwardFold] = []
    while add_months(test_start, test_months) <= final_exclusive:
        if rolling_train_months is None:
            train_start = first_month
        else:
            train_start = add_months(test_start, -rolling_train_months)
            if train_start < first_month:
                train_start = first_month
        folds.append(
            WalkForwardFold(
                fold=len(folds),
                train_start=f"{train_start}-01",
                train_end=f"{test_start}-01",
                test_start=f"{test_start}-01",
                test_end=f"{add_months(test_start, test_months)}-01",
                embargo_days=embargo_days,
            )
        )
        test_start = add_months(test_start, step_months)
    return folds


def portable_partition_record(partition: MonthPartition) -> dict[str, Any]:
    record = asdict(partition)
    for key in ("report_path", "canonical_path", "event_path"):
        record[key] = str(record[key])
    return record
