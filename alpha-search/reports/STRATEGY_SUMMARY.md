# Strategy summary in plain English (October 2026)

This is a short guide to what was tried, what survived, and what it means.
Full detail is in `HYPOTHESIS_LEDGER.md` and the per-experiment reports. To
explore the data hands-on, use `notebooks/TopStrategiesWorkbench.ipynb`.

## How to read the numbers

- **bps (basis points):** 1 bp = 0.01%. A trade that makes 10 bps on $100k
  makes $100.
- **Gross vs net:** gross is before costs. Net subtracts fees (assumed 2 bps
  per side, so 4 bps per round trip), and where stated, spread and
  slippage.
- **t-statistic (t):** roughly, how many standard errors the average sits
  above zero. Above 2 is the usual "probably not luck" line. We compute it
  day by day, so one busy day can't dominate.
- **Ordinary days:** the result with the 10 best days removed. Crypto edges
  often live in a few crash days; this shows what is left.
- **Registered / frozen:** we write the exact rule down *before* looking at
  the data it is tested on. Results found by browsing don't count until
  they pass such a test.
- **Locked months:** June–August 2026, kept unread as a final one-shot
  check.

## The scoreboard

| # | Idea | Status | One-line verdict |
|---|---|---|---|
| 1 | 5-minute snap-back after sharp moves (C-006) | **Confirmed** on new coins; locked months not rejected | The best result so far, but it has decayed to about break-even per trade at target fees. Concentrated in big days. |
| 2 | Rest your orders instead of crossing (H-018) | **Confirmed**, all 3 coins | Saves about 2 bps per side on every trade you make. A cost saver, not an alpha. |
| 3 | Fade rare extreme moves (C-007) | **Partly confirmed** (BTC and ETH, not SOL) | Similar to #1 but rarer. Kept as a possible add-on. |
| 4 | Big "sweeps" predict the next seconds (H-014/15) | **Confirmed but too small** | Rock-solid statistically, but worth fractions of a bp. Fees eat it. |
| 5 | Crowded funding and positioning reverse over a day (H-010/19/20) | **Did not hold up** | Worked on BTC in 2023, failed on other coins and venues. The funding income is real, but price swings swamp it. |

---

## 1. The 5-minute snap-back (C-006): the main finding

**The idea.** When price moves unusually hard over the last few minutes
(the sharpest 1% of moves, judged against recent volatility), it tends to
partly snap back over the next ~5 minutes. Fast traders overshoot, and
liquidity providers step in.

**What we trade.** A simple regression on the latest bar's move and the
15- and 60-minute trend before it gives a forecast. When the forecast is in
its most extreme 1%, we trade in its direction (in practice, against the
recent move), enter about a second later, and exit after 5 minutes.

**How it was tested.**

- It was found on Bitcoin (2023 to 2026) among many variations, so the BTC
  numbers alone could be a lucky pick.
- To check, we froze the exact rule and applied it unchanged to Ethereum and
  Solana. No previous test had looked at their 5–15 minute returns.
- The model is refitted each month using only the previous 12 months.

**The result.**

| | ETH | SOL | BTC (where it was found) |
|---|---:|---:|---:|
| Trades per day | ~28 | ~18 | ~28 |
| Average per trade, before costs | 12.7 bps | 29.1 bps | 15.5 bps |
| Ordinary days only | 5.6 | 11.0 | 7.5 |
| t-statistic | 13.5 | 12.9 | 14.3 |
| Last 12 months to May 2026 (per trade) | ~4.7 bps | ~3.1 bps | ~5.8 bps |

It passed every pre-set test on both new coins. Re-pricing trades with real
bid/ask quotes (May 2023 to March 2024) showed the spread costs only about
0.5–3.5 bps, so it isn't a price-quote artefact.

**The catches.**

- **Concentration:** about 60% of the profit comes from the 10 best days
  (crashes and squeezes).
- **Decay:** the edge is shrinking. Over the last 12 months of the test
  (per trade, which is what you earn) it averaged about 4–6 bps before
  costs, roughly the 4 bp round trip. **At your target fees it is
  currently about break-even.**
- **Thin books:** these trades happen when the order book is thin. The best
  price typically holds only $1k–30k. Rough slippage at $100k per trade is
  about 1–2 bps for BTC/ETH and about 9 bps for SOL. On today's thin margin,
  that matters. This is indicative only: our order-book data is top-of-book
  and short.
- **Clustering:** many trades fire together in fast markets, so exposure
  and impact add up.

**Locked months (June–August 2026), the one-shot check.**

- Not rejected under the rule set beforehand, but weak per trade: BTC
  0.2 bps, ETH 4.9, SOL 28.3 (mostly one day).
- The two big-move days in that window lost money on all three coins.

**What's next.** A prospective test (P-006) on new data from September 2026,
judged per trade after fees. It is low-powered: it will most likely tell us
within a year whether the effect has gone, but would need it to strengthen to
confirm. A proper depth (L2) study and your fee route also matter, because
the margin is now thin.

---

## 2. Rest, don't cross (H-018): cheaper execution

**The idea.** Paying the taker fee (5 bps at base tier) is expensive. If
you can wait up to 30 minutes, placing a limit order at the best price
usually gets filled. You pay the maker fee instead, and lose only a little
to price drift while waiting.

**How it was tested.** We simulated about 2,900 random $10k orders per
coin over 8 months against real quotes and trades, with a conservative
queue model (we are last in line at our price).

**The result.** Cost per order, in bps (lower is better):

| | Take immediately | Rest at best price | Rest + signal timing |
|---|---:|---:|---:|
| BTC | 5.0 | 3.1 | 3.7 |
| ETH | 5.0 | 3.1 | 3.7 |
| SOL | 5.3 | 3.3 | 3.6 |

- Resting saves about 1.9–2.0 bps per side, and every order filled within
  30 minutes.
- Adding our short-term signals to decide when to cross made it *worse*:
  each early cross costs more in fees than it saves.

**What it means.** Use patient limit orders wherever a strategy can wait.
For the 5-minute snap-back, entry must be fast, so mostly taker, but exits
could rest.

---

## 3. Fading extreme moves (C-007)

**The idea.** After a rare, violent move (about 100 a year), fade it for
5 minutes.

**The result.**

- ETH passed: about 12 bps per trade, positive every year.
- SOL failed: its profit came entirely from a few 2024 crash days.
- BTC (where it was found): about 10 bps.

It is not confirmed overall, but kept as a possible add-on for BTC and ETH.
It probably overlaps heavily with the snap-back (#1).

---

## 4. Sweeps and order-book "footprints" (H-014 to H-016)

**The idea.** When a large order sweeps through several price levels, price
keeps moving the same way for a few seconds.

**The result.**

- One of the most statistically robust findings: positive in 38 of 38
  months, on BTC, ETH and SOL.
- But the effect is about 0.2–0.8 bps over seconds, far below fees.
- Trying to use it to protect market-making quotes or to time execution
  didn't help (H-017, H-018).
- Aggregated over 8–24 hours it carries no information (E-005).

**What it means.** This is real market microstructure that is too small to
trade at our costs. It may still matter for understanding impact (the
paper).

---

## 5. Crowded funding and positioning (H-009, H-010, H-019, H-020)

**The idea.** When too many traders lean one way (high funding, a rich
premium, a strong run-up), the market tends to reverse over a day.

**The result.**

- It looked good on Bitcoin (2023–26 Sharpe about 0.7–0.9, earned mostly in 2023).
- It failed on ETH and SOL, and on a broad panel of ~180 alt coins.
- A market-neutral version that collects funding earned that funding very
  reliably, but price swings in the alts wiped it out. Hedging that properly
  (cash-and-carry) needs too much capital, so it was dropped.

**What it means.** The crowding signal looks like a 2021–2023 effect that
has faded.

---

## Lessons from the process

- **Test before trusting:** most things that looked good on Bitcoin did not
  survive a frozen test on fresh data. The one that did (#1) is the
  strongest candidate.
- **Fees decide everything:** many real effects are smaller than fees.
  Cheaper execution (#2 and your fee route) changes which ideas are viable.
- **Watch the best days:** crypto edges often come from a handful of crashes.
  Always look at the "ordinary days" number.
- **Measure P&L properly:** returns on perpetual futures must be simple
  returns, not log returns. A correction caught during this work changed how
  some results split between long and short legs.
