# Trade-event construction audit (BTCUSDT perpetual)

Produced by `crypto-impact-study/scripts/event_construction_audit.py`
(run 2026-10-06). Every number below is taken from the CSVs in this folder.

## The rule now used: `timestamp_direction_v1`

Defined in `shared-methodology/src/crypto_market_data/events.py`.

- **Order.** Fills are processed in exchange trade-ID order.
- **When a new event starts.** A new event starts whenever the millisecond
  timestamp changes *or* the aggressor sign changes.
- **Price is not a grouping key.** A sweep that walks through several price
  levels is therefore one event.
- **Contents of each event:** sign, total quantity, VWAP, first, last, minimum
  and maximum price, fill count, number of price levels, and first and last
  trade ID.
- **Price used by the impact study:** the last fill price of the event, i.e.
  where the sweep ended.

## What the paper's original input files did (`old_inputs.csv`)

The files are in `data/raw/BTCUSDT_monthly`, 2025-01 to 2026-05. They were
checked against the raw fills:

| Check | Result |
|---|---|
| Rows | One per millisecond timestamp. 459,217,752 rows against 459,217,769 fill timestamps; the 17 missing are one per month. |
| Price | The **first** fill's price at the timestamp, in every row. It never equals the last fill's price where the two differ. |
| Sign | The **first** fill's sign, in every row. |
| Quantity | The gross quantity of all fills at the timestamp, in every row. |

**Consequences of the old construction:**

1. **Multi-price sweeps were already one event in volume.** But each event
   carried the price at which the sweep *started*, not where it ended. In the
   old files, 81.0 million timestamps (17.6%) have first ≠ last price.
2. **Mixed-sign timestamps were merged under the first fill's sign.** Volume
   from the opposite side was counted with the wrong sign.

## Event statistics, 2021-01 to 2026-08 (`summary.csv`)

**Counts:**

| Quantity | Value |
|---|---:|
| Fills | 7,676,258,461 |
| Millisecond timestamps with trading | 2,285,747,217 |
| Events (`timestamp_direction_v1`) | 2,335,198,845 |
| Fills per event | 3.29 |
| Extra events from splitting mixed-sign timestamps | 2.16% |

**Sweeps** (multi-fill and multi-price events):

| Quantity | Value |
|---|---:|
| Events with more than one fill | 35.0% of events, 86.8% of volume |
| Events spanning more than one price level | 15.1% of events, **55.7% of volume**, 52.7% of fills |

**Mixed-sign timestamps:**

| Quantity | Value |
|---|---:|
| Timestamps with both aggressor sides | 1.58% of timestamps, holding 3.66% of events |
| Volume mis-signed by the first-fill-sign rule | 1.77% of volume |

**Price path:**

| Quantity | Value |
|---|---:|
| Share of the absolute log-price path that occurs *inside* events (first → last fill) | **38.9%** |

**Price levels per event** (`price_levels.csv`):

| Levels | 1 | 2 | 3 | 4–5 | 6–10 | 11–20 | 21–50 | >50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Share of events | 84.9% | 5.2% | 2.9% | 3.0% | 2.5% | 1.1% | 0.5% | 0.1% |
| Share of volume | 44.3% | 5.9% | 4.4% | 6.4% | 9.5% | 9.8% | 11.7% | 8.0% |

**By year** (`monthly.csv`, monthly min–max):

- the multi-price event share ranges from 5.7% to 26.2%;
- the multi-price volume share ranges from 36% to 73%;
- the mis-signed volume share ranges from 1.0% to 2.5%;
- the within-event share of the price path ranges from 26% to 52%.

## Reading for the paper

- **Event structure.** Sweeps across price levels are a minority of events,
  but they carry over half of all volume and almost two-fifths of the price
  path.
- **Why the event price matters.** An event priced at its first fill moves
  that within-event price change into the *next* event. Within a bar this
  only shifts the move. At bar boundaries it moves the move into the
  following bar, misattributing it relative to the flow that caused it.
- **Mixed-sign timestamps.** These are rare (1.6% of timestamps), but they
  mis-sign 1.8% of volume. The old validation statement ("negligible") should
  be replaced by these numbers.
- **Effect on the paper's estimates (the bridge).** Re-estimating the paper's
  own period with corrected events changes ξ from 0.955 to 0.941 and ψ from
  0.608 to 0.606 (`../bridge_2025_01_2026_05/`). The correction matters for
  measurement but does not overturn the P&B results.
