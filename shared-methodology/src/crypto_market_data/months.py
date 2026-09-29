"""Calendar-month helpers."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date


def parse_month(value: str) -> date:
    """Parse a YYYY-MM value into the first day of that month."""

    try:
        year_text, month_text = value.split("-", maxsplit=1)
        parsed = date(int(year_text), int(month_text), 1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid month {value!r}; expected YYYY-MM.") from exc
    if value != parsed.strftime("%Y-%m"):
        raise ValueError(f"Invalid month {value!r}; expected YYYY-MM.")
    return parsed


def iter_months(start_month: str, end_month: str) -> Iterator[str]:
    """Yield inclusive YYYY-MM values in chronological order."""

    current = parse_month(start_month)
    end = parse_month(end_month)
    if current > end:
        raise ValueError("start_month must not be after end_month.")

    while current <= end:
        yield current.strftime("%Y-%m")
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
