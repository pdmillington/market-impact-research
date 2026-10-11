# Archived exploratory notebooks

These notebooks are kept as a record of early exploration (September 2026). They are
not part of the paper and are not maintained.

- **Data.** They read the legacy paper input files (one row per millisecond
  timestamp with the first fill's price and sign), 2025-01 to 2026-05. Those files
  are now in `shared-data/legacy/crypto-impact-study-data/raw/BTCUSDT_monthly`. The
  notebooks still point at the old `~/research/...` path and will not run as they are.
- **Contents.**
  - Descriptive bar statistics.
  - Regime-conditioned mean reversion.
  - A naive log–log impact regression and the binned-slope work that became
    `PowerLawImpact.ipynb`.
  - An EMA volatility-forecast model.
  - Mutual-information lag profiles.
  - Direction-history signals with in-sample filters and a walk-forward check.
  - Order-flow-imbalance quintiles.
- **Status of the trading ideas.** None was registered in
  `alpha-search/HYPOTHESIS_LEDGER.md`, and the backtests choose thresholds on the
  data they are evaluated on. They do not meet the alpha-search standard. An idea
  worth pursuing should be registered as a new hypothesis before any test.

`DataTest2.ipynb` is a later variant of `DataTest.ipynb`. It adds conditioned mean
reversion and a three-state direction-history analysis, and drops the
signal-filter cells.
