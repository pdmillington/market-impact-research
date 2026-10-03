from __future__ import annotations

from datetime import date
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import p006_monthly_data as p  # noqa: E402


def test_latest_complete_month_handles_year_boundary():
    assert p.latest_complete_month(date(2026, 10, 1)) == "2026-09"
    assert p.latest_complete_month(date(2027, 1, 15)) == "2026-12"


def test_no_look_before_twelve_months_of_data():
    assert p.look_status(None)["looks_allowed"] == []
    assert p.look_status("2027-07")["looks_allowed"] == []
    assert p.look_status("2027-07")["next_look_needs_data_through"] == "2027-08"


def test_looks_open_only_at_registered_dates():
    assert p.look_status("2027-08")["looks_allowed"] == [1]
    assert p.look_status("2028-07")["looks_allowed"] == [1]
    assert p.look_status("2028-08")["looks_allowed"] == [1, 2]
    assert p.look_status("2028-08")["next_look"] is None
