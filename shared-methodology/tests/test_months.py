import pytest

from crypto_market_data.months import iter_months


def test_iter_months_crosses_year_boundary() -> None:
    assert list(iter_months("2025-11", "2026-02")) == [
        "2025-11",
        "2025-12",
        "2026-01",
        "2026-02",
    ]


def test_iter_months_rejects_reverse_range() -> None:
    with pytest.raises(ValueError, match="must not be after"):
        list(iter_months("2026-02", "2025-11"))
