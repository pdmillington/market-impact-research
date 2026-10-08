# Gaps in the Binance trade archives and their repair (2026-10-08)

## What was found (`scripts/trading_gaps_audit.py`)

- **Detection.** Every pair of consecutive trade events from 2019-10 to 2026-08
  was checked for time gaps of 60 s or more. Each gap was tested for
  continuity of exchange trade IDs:
  - continuous IDs mean a genuine pause (the exchange printed nothing);
  - an ID jump means trades exist that the archive lacks.
- **Missing data: 12 stretches.**
  - Where: 2021-02 to 2023-11.
  - Size: 4.11 million trades in total.
  - Pattern: almost all are the end of a UTC day missing from Binance's
    monthly archive, with the data resuming at 00:00:00.
  - Largest: 2022-02-14 07:28 UTC to midnight, 2,295,328 trades. This
    produced a 16.6-hour 4,000-event bar.
- **Genuine pauses: 16.**
  - The six longest: 2021-03-02 01:00–02:00 (60 min), 2022-05-28 (35 min),
    2022-05-01 (30 min), 2023-09-12 (20 min), 2025-08-29 (19 min) and
    2024-10-28 (15 min).
  - The rest last 1–3 minutes, in thin markets (2019–2020).
  - These are real market features and are kept.
- **No closures for holidays.** Christmas and New Year trade continuously, at
  roughly 40–80% of normal activity (`holiday_activity.csv`).

## Repair

**8 stretches filled from Binance's daily trade archives.**
- Checks:
  - checksums were verified;
  - every fill shared by the daily and monthly archives is identical in price,
    quantity, time and side (`daily_archive_gap_check.csv`).
- Amount added:
  - 3,269,892 fills, giving 1,185,760 events;
  - affected months: 2021-02, 2022-02, 2022-09, 2022-11, 2023-02, 2023-03.
- How it was done:
  - `parse_month_archive` (shared-methodology) now merges any stored daily
    archive for the month;
  - it adds only trade IDs absent from the monthly file, and refuses to merge
    if any shared fill disagrees;
  - the script that ran the repair is
    `shared-methodology/scripts/repair_from_daily_archives.py`.
- Superseded files: the previous versions of these months and every derived
  dataset are in `shared-data/superseded/2026-10-08_before_daily_supplement/`.

**4 stretches unrecoverable.**
- Which:
  - 2022-10-25 22:52–24:00 (100,865 trades);
  - 2023-03-14 22:33–24:00 (318,590);
  - 2023-11-21 02:43–08:51 (417,955);
  - 2023-11-21 10:29–10:38 (7,265).
- In total that is 844,675 trades, about 0.011% of all fills from 2021-01 to
  2026-08.
- Why they can't be filled:
  - both trade archives lack them;
  - Binance's aggTrades archive has them, but it merges same-price fills up to
    100 ms apart;
  - events rebuilt from aggTrades number only 58–65% of the true events, so it
    cannot reproduce `timestamp_direction_v1`.
- Handling:
  - the gaps are listed in `crypto-impact-study/config/btcusdt_data_gaps.csv`;
  - any bar or block that spans one is excluded (4 per bar size; a time bar
    overlapping one is also dropped);
  - its observed volume stays in the trailing 24-hour windows.

## Suggested wording for the paper (Data section or footnote)

> The data are Binance's public USD-M BTCUSDT trade archives. The monthly
> archives lack the closing hours of eight UTC days between February 2021 and
> March 2023. These were restored from Binance's daily archives, which agree
> exactly with the monthly archives wherever both are available: 3.27 million
> fills were added. Four further stretches, on 25 October 2022, 14 March 2023
> and 21 November 2023, are missing from both archives. They total 0.84
> million fills (0.011% of the sample), and the bar spanning each is
> excluded. Apart from these, the market traded continuously throughout the
> sample, including holidays. The exception is a small number of exchange
> pauses, the longest one hour (2 March 2021).
