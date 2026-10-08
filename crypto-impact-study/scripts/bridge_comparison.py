"""Compare the paper's published P&B estimates with the corrected-events bridge run.

Paper values: reports/tables/pb_master_curve_parameters.csv and pb_scaling_exponents.csv
(the paper's original outputs, 2025-01..2026-05, legacy event files).
Bridge values: reports/redo/bridge_2025_01_2026_05/ (run_pb_collapse.py on the same
period with corrected events).

Writes reports/redo/bridge_2025_01_2026_05/bridge_comparison.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


STUDY = Path(__file__).resolve().parents[1]
PAPER = STUDY / "reports" / "tables"
BRIDGE = STUDY / "reports" / "redo" / "bridge_2025_01_2026_05"
EXPONENTS = {  # label -> column in pb_scaling_exponents.csv
    "xi (Theil-Sen)": "fitted_q_scale",
    "psi (Theil-Sen)": "fitted_r_scale",
    "H_q,24h (Theil-Sen)": "trailing_volume_fraction_std",
    "H_q,raw (Theil-Sen)": "signed_volume_std",
    "H_epsilon (Theil-Sen)": "sign_sum_std",
    "H_r (Theil-Sen)": "return_std",
}


def main() -> None:
    paper_shape = pd.read_csv(PAPER / "pb_master_curve_parameters.csv").iloc[0]
    bridge_shape = json.loads((BRIDGE / "pb_master_curve_parameters.json").read_text())
    paper_exp = pd.read_csv(PAPER / "pb_scaling_exponents.csv").set_index("column")["robust_exponent"]
    bridge_exp = pd.read_csv(BRIDGE / "pb_scaling_exponents.csv").set_index("column")["robust_exponent"]
    rows = [{"quantity": f"{p} (master-curve shape)", "paper": paper_shape[p], "corrected_events": bridge_shape[p]}
            for p in ("alpha", "beta")]
    rows += [{"quantity": label, "paper": paper_exp[col], "corrected_events": bridge_exp[col]}
             for label, col in EXPONENTS.items()]
    table = pd.DataFrame(rows)
    table["difference"] = table["corrected_events"] - table["paper"]
    table.to_csv(BRIDGE / "bridge_comparison.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
