"""Parse Binance futures trade ZIPs into canonical fills and events."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from .events import reconstruct_timestamp_direction_events
from .layout import DataLayout


RAW_COLUMNS = ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker"]
CANONICAL_COLUMNS = [
    "trade_id",
    "timestamp_ms",
    "price",
    "quantity",
    "aggressor_sign",
]


@dataclass(frozen=True)
class ParseReport:
    """Auditable summary of one parsed month."""

    symbol: str
    month: str
    source_archive: str
    canonical_path: str
    event_path: str
    source_bytes: int
    canonical_bytes: int
    event_bytes: int
    fill_rows: int
    event_rows: int
    mixed_direction_timestamps: int
    multi_price_events: int
    duplicate_trade_ids: int
    invalid_fill_rows: int
    fill_count_difference: int
    quantity_difference: float
    signed_quantity_difference: float
    first_timestamp_ms: int
    last_timestamp_ms: int
    event_rule: str = "timestamp_direction_v1"
    exact_duplicate_rows_removed: int = 0


def _csv_member(archive: zipfile.ZipFile) -> str:
    members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    if len(members) != 1:
        raise ValueError(
            f"Expected exactly one CSV in archive, found {len(members)}: {members}"
        )
    return members[0]


def _has_header(csv_path: Path) -> bool:
    with csv_path.open("rb") as stream:
        first_field = stream.readline().split(b",", maxsplit=1)[0]
    return first_field.strip().lower() == b"id"


def canonical_fills_from_csv(csv_path: Path) -> pl.LazyFrame:
    """Create the compact, one-row-per-exchange-trade canonical schema."""

    has_header = _has_header(csv_path)
    scan = pl.scan_csv(
        csv_path,
        has_header=has_header,
        new_columns=None if has_header else RAW_COLUMNS,
        schema_overrides={
            "id": pl.UInt64,
            "price": pl.Float64,
            "qty": pl.Float64,
            "quote_qty": pl.Float64,
            "time": pl.Int64,
            "is_buyer_maker": pl.Boolean,
        },
    )
    return (
        scan.select(
            pl.col("id").alias("trade_id"),
            pl.col("time").alias("timestamp_ms"),
            pl.col("price"),
            pl.col("qty").alias("quantity"),
            pl.when(pl.col("is_buyer_maker"))
            .then(pl.lit(-1, dtype=pl.Int8))
            .otherwise(pl.lit(1, dtype=pl.Int8))
            .alias("aggressor_sign"),
        )
        .select(CANONICAL_COLUMNS)
    )


def _write_canonical(csv_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    temporary_path.unlink(missing_ok=True)
    canonical_fills_from_csv(csv_path).sink_parquet(
        temporary_path,
        compression="zstd",
        compression_level=6,
        statistics=True,
        row_group_size=250_000,
        maintain_order=True,
    )
    temporary_path.replace(output_path)
    return output_path


def _remove_adjacent_exact_duplicates(canonical_path: Path) -> int:
    """Remove repeated source lines without masking conflicting trade IDs.

    Binance monthly archives occasionally contain immediately repeated rows. An
    exact repeat is not a second exchange fill and is removed. Rows that reuse a
    trade ID with any different canonical field remain in place so the standard
    uniqueness validation fails visibly.
    """

    fills = pl.scan_parquet(canonical_path)
    exact_repeat = pl.all_horizontal(
        [
            pl.col(column) == pl.col(column).shift(1)
            for column in CANONICAL_COLUMNS
        ]
    ).fill_null(False)
    removed = int(
        fills.select(exact_repeat.sum().alias("rows")).collect()["rows"][0]
    )
    if not removed:
        return 0

    temporary_path = canonical_path.with_suffix(".deduplicated.parquet.part")
    temporary_path.unlink(missing_ok=True)
    fills.filter(~exact_repeat).sink_parquet(
        temporary_path,
        compression="zstd",
        compression_level=6,
        statistics=True,
        row_group_size=250_000,
        maintain_order=True,
    )
    temporary_path.replace(canonical_path)
    return removed


def _build_report(
    *,
    symbol: str,
    month: str,
    source_path: Path,
    canonical_path: Path,
    event_path: Path,
    exact_duplicate_rows_removed: int = 0,
) -> ParseReport:
    fills = pl.scan_parquet(canonical_path)
    events = pl.scan_parquet(event_path)

    fill_summary = fills.select(
        pl.len().alias("rows"),
        pl.col("trade_id").n_unique().alias("unique_trade_ids"),
        (
            (pl.col("price") <= 0)
            | (pl.col("quantity") <= 0)
            | pl.col("price").is_null()
            | pl.col("quantity").is_null()
        ).sum().alias("invalid_rows"),
        pl.col("quantity").sum().alias("quantity"),
        (pl.col("quantity") * pl.col("aggressor_sign"))
        .sum()
        .alias("signed_quantity"),
        pl.col("timestamp_ms").min().alias("first_timestamp_ms"),
        pl.col("timestamp_ms").max().alias("last_timestamp_ms"),
    ).collect()
    event_summary = events.select(
        pl.len().alias("rows"),
        pl.col("fill_count").sum().alias("fill_count"),
        pl.col("quantity").sum().alias("quantity"),
        (pl.col("quantity") * pl.col("aggressor_sign"))
        .sum()
        .alias("signed_quantity"),
        (pl.col("price_level_count") > 1).sum().alias("multi_price_events"),
    ).collect()
    mixed = (
        fills.group_by("timestamp_ms")
        .agg(pl.col("aggressor_sign").n_unique().alias("directions"))
        .filter(pl.col("directions") > 1)
        .select(pl.len())
        .collect()
    )

    fill_rows = int(fill_summary["rows"][0])
    event_fill_count = int(event_summary["fill_count"][0])
    quantity_difference = float(
        event_summary["quantity"][0] - fill_summary["quantity"][0]
    )
    signed_quantity_difference = float(
        event_summary["signed_quantity"][0] - fill_summary["signed_quantity"][0]
    )
    duplicate_trade_ids = fill_rows - int(fill_summary["unique_trade_ids"][0])
    invalid_fill_rows = int(fill_summary["invalid_rows"][0])

    if duplicate_trade_ids or invalid_fill_rows or event_fill_count != fill_rows:
        raise ValueError(
            "Parsed data failed identity, positivity, or fill-count validation."
        )
    tolerance = max(1e-9, abs(float(fill_summary["quantity"][0])) * 1e-12)
    if (
        abs(quantity_difference) > tolerance
        or abs(signed_quantity_difference) > tolerance
    ):
        raise ValueError("Event reconstruction failed volume-conservation checks.")

    return ParseReport(
        symbol=symbol,
        month=month,
        source_archive=str(source_path),
        canonical_path=str(canonical_path),
        event_path=str(event_path),
        source_bytes=source_path.stat().st_size,
        canonical_bytes=canonical_path.stat().st_size,
        event_bytes=event_path.stat().st_size,
        fill_rows=fill_rows,
        event_rows=int(event_summary["rows"][0]),
        mixed_direction_timestamps=int(mixed.item()),
        multi_price_events=int(event_summary["multi_price_events"][0]),
        duplicate_trade_ids=duplicate_trade_ids,
        invalid_fill_rows=invalid_fill_rows,
        fill_count_difference=event_fill_count - fill_rows,
        quantity_difference=quantity_difference,
        signed_quantity_difference=signed_quantity_difference,
        first_timestamp_ms=int(fill_summary["first_timestamp_ms"][0]),
        last_timestamp_ms=int(fill_summary["last_timestamp_ms"][0]),
        exact_duplicate_rows_removed=exact_duplicate_rows_removed,
    )


def parse_month_archive(
    *,
    root: Path,
    symbol: str,
    month: str,
    overwrite: bool = False,
) -> ParseReport:
    """Parse one source archive and write canonical fills, events, and metadata."""

    layout = DataLayout(Path(root))
    source_path = layout.source_archive(symbol, month)
    canonical_path = layout.canonical_month(symbol, month)
    event_path = layout.event_month(symbol, month)
    report_path = layout.report_month(symbol, month)

    if not source_path.exists():
        raise FileNotFoundError(f"Source archive not found: {source_path}")
    if not overwrite:
        completed = (
            canonical_path.exists() and event_path.exists() and report_path.exists()
        )
        if completed:
            return ParseReport(**json.loads(report_path.read_text()))
        if canonical_path.exists() or event_path.exists() or report_path.exists():
            raise FileExistsError(
                f"Incomplete output exists for {symbol} {month}; inspect it or use "
                "overwrite=True."
            )

    temporary_root = layout.temporary_root()
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="month-", dir=temporary_root
    ) as temp_dir:
        with zipfile.ZipFile(source_path) as archive:
            member = _csv_member(archive)
            csv_path = Path(temp_dir) / Path(member).name
            with archive.open(member) as source, csv_path.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        _write_canonical(csv_path, canonical_path)

    exact_duplicate_rows_removed = _remove_adjacent_exact_duplicates(
        canonical_path
    )

    reconstruct_timestamp_direction_events(canonical_path, event_path)
    report = _build_report(
        symbol=symbol,
        month=month,
        source_path=source_path,
        canonical_path=canonical_path,
        event_path=event_path,
        exact_duplicate_rows_removed=exact_duplicate_rows_removed,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), indent=2) + "\n")
    return report
