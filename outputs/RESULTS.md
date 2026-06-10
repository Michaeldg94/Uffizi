# Uffizi project results

Two separate streams. The crowd simulator produces the **museum-wide** results; the
RL agents are the **optimal individual** built on top of those crowds.

Stored data:
- `outputs/results_museum_wide.json` (== `12_museum_wide.json`) — crowd-simulator stream
- `outputs/results_rl.json` — RL stream (booking grid + algorithm matrix)
- `outputs/results_rl_booking.json` — richer reproducible booking eval (pipeline 10)

Population mix (both streams): **60% Instagram tourist / 30% normal tourist / 10% art lover**.

---

## Stream 1 — Museum-wide (crowd simulator): base crowd vs intervened crowd (all 11 levers)

The gate: a director will only consider interventions that do **not collapse revenue**.

| crowd | total welfare (completed) | revenue (completed) | peak Botticelli density |
|---|---|---|---|
| 500 | 277,256 -> 343,507 (**+24%**) | 11,542 -> 11,317 | 0.41 -> 0.33 |
| 2500 | 1,106,175 -> 1,274,045 (**+15%**) | 56,604 -> 56,544 | 1.16 -> 1.00 |
| max | 1,514,458 -> 1,986,425 (**+31%**) | 96,502 -> 94,792 | 1.82 -> 1.00 |

Per-segment welfare per completed visitor (base -> intervened):
- Instagram tourist: ~flat per visit, but **more get through** (throughput up)
- Normal tourist: **+49% to +58%**
- Art lover: **+9% to +33%**

Welfare rises across the board and masterpiece congestion drops sharply.

**Revenue (resolved).** The canonical status-quo baseline is **~94-95k/day** (current
pricing read from the revenue model: EUR 29 pre-booked / EUR 25 walk-in, 15% free,
10% reduced at EUR 2, ~75% paying x ~5000). The old ~EUR 62.5k figure assumed a wrong
3000 x EUR 27 and is discarded.

Against that baseline:
- **All 11 (kitchen sink)**: revenue **preserved** (~flat). RAMA general entry (EUR 15,
  below the EUR 29 baseline) + resident annual pass + group-surcharge/quiet-hours/
  timed-entry demand suppression offset dynamic pricing. The director gate (no revenue
  collapse) passes at **+31% welfare, revenue intact**.
- **Optimized portfolio** (drops the heavy suppressors, keeps RAMA + enrichment +
  extended hours + dynamic pricing + group cap + resident pass): welfare **+~16%** and
  revenue **+~15%**, both clearly up. The win-win.

The RL agents (Stream 2) are trained on the all-11 intervened world, so all-11 is the
consistent "intervened crowd"; the portfolio is the economics-optimization bonus.

---

## Stream 2 — RL (optimal agent on top): booking grid

Deterministic eval, seeds 900000-5. Baseline = matched no-intervention walk; intervened
= the RAMA booking agent. 3/3 masterpieces secured at every cell; entry A1; no teleports.

| profile | crowd | baseline | intervened | vs base | lead chosen |
|---|---|---|---|---|---|
| Art lover | 500 | 5112 | 7113 | +39% | 7 days |
| Art lover | 2500 | 4780 | 6297 | +32% | 35 days |
| Art lover | max | 4009 | 5646 | +41% | 35 days |
| Normal tourist | 500 | 2694 | 3686 | +37% | 7 days |
| Normal tourist | 2500 | 2708 | 3695 | +36% | 7 days |
| Normal tourist | max | 2680 | 3685 | +38% | 35 days |

Book-early-when-busier emerges: lead time rises with crowd (7 -> 35 days).

### Algorithm comparison (points, deterministic)

| | art 500 | art 2500 | art max | tour 500 | tour 2500 | tour max |
|---|---|---|---|---|---|---|
| PPO-baseline | 5112 | 4780 | 4009 | 2694 | 2708 | 2680 |
| DQN-baseline | 5230 | 3683 | 1756 | 2694 | 2708 | 2680 |
| PPO-intervened | 7113 | 5646 | 4833 | 3686 | 3695 | 2093 |
| MaskablePPO-intervened | 7113 | 6297 | 5646 | 3686 | 3695 | 3685 |
| DQN-intervened | 2842 | 6290 | 4324 | 3686 | 3695 | 3685 |

MaskablePPO is the clear winner; masking is what lets PPO discover book-early; DQN is
crowd-fragile and high-variance on the art lover.
