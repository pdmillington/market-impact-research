# Shared crypto market-data methodology

This package contains the reusable ingestion and event-reconstruction layer used by
both the academic reaction-function work and the separate alpha-search project.

It deliberately keeps three concepts separate:

1. Original Binance monthly ZIP archives are immutable source evidence.
2. Canonical fills preserve one row per exchange trade with a compact schema.
3. Derived event datasets encode an explicit, versioned reconstruction rule.

The first event rule, `timestamp_direction_v1`, combines only **consecutive** fills
with the same millisecond timestamp and aggressor direction. Prices are not used as
a grouping key because one aggressive order can sweep several price levels. Each
event retains total quantity, VWAP, first/last/minimum/maximum price, fill count,
price-level count, and the first/last exchange trade identifiers.

## Installation

From this directory:

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## Example

```bash
crypto-market-data download \
  --root ../shared-data \
  --symbol BTCUSDT \
  --start-month 2019-09 \
  --end-month 2019-09

crypto-market-data parse \
  --root ../shared-data \
  --symbol BTCUSDT \
  --start-month 2019-09 \
  --end-month 2019-09
```

Monthly files are processed independently. The ZIP is extracted only under
`ROOT/.tmp/crypto-market-data` on the same data volume and is removed automatically
after the two Parquet outputs have been written. This prevents large transient CSV
files from consuming space on the internal system drive when `ROOT` is external.

Completed months are resumable. Re-running an overlapping range reads the saved
validation report for complete months and continues with the first missing month:

```bash
crypto-market-data download-and-parse \
  --root ../shared-data \
  --symbol BTCUSDT \
  --start-month 2019-09 \
  --end-month 2026-08
```

The parser removes only immediately adjacent records that are identical across
all canonical fields, and records the count in
`exact_duplicate_rows_removed`. Conflicting reuse of a trade ID remains a
validation failure, with one documented exception.

**Trade-ID reuse after an outage.** Binance has restarted its trade-ID counter a
few IDs early when trading resumed after a halt. In ETHUSDT on 2025-08-29,
trading stopped at 06:18:03 UTC, resumed at 06:37:04, and IDs 6299136398–6299136400
were issued twice, to six genuine, distinct fills. A duplicated ID is accepted as
*reused* only when it has exactly two records at least 60 seconds apart. When a
month contains reused IDs, its canonical fills and event reconstruction are ordered
by `(timestamp_ms, trade_id)` rather than `trade_id`, which restores the true
sequence. The count is recorded in `reused_trade_ids`. Months without reuse are
unaffected and keep trade-ID ordering. Any other duplicate still fails.

Check current coverage and storage without scanning the Parquet files:

```bash
crypto-market-data status --root ../shared-data --symbol BTCUSDT
```

The canonical layer is the permanent research input. Event outputs are caches and
may be deleted and reconstructed if storage becomes constrained. Do not delete the
source ZIPs unless another verified copy is retained elsewhere.
