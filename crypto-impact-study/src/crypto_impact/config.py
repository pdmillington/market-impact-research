"""Project configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for one impact-estimation pipeline run."""

    symbol: str
    start_date: str
    end_date: str
    window: str
    raw_dir: Path = RAW_DATA_DIR
    processed_dir: Path = PROCESSED_DATA_DIR
    n_bins: int = 12

    @property
    def run_name(self) -> str:
        """Return a stable name for outputs from this run."""

        return f"{self.symbol}_{self.start_date}_{self.end_date}_{self.window}"
