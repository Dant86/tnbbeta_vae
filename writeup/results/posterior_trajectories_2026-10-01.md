# Real training trajectories of p, q, epsilon and kappa, 2026-10-01

- **Date:** 2026-10-01
- **Master commit:** `da47de8` ("Add apps.eval.posterior_trajectories: real
  (p,q,epsilon)/kappa plots")
- **Data:** `runs/<run_id>/metrics.jsonl` for 301 runs, rsynced from the DSI
  cluster (`/home/vpathak/tnbbeta_vae/runs/`) into `cluster_runs/` (local,
  untracked -- not the figures' data source for anyone reproducing this, just
  this run's provenance).
- **Command:**
  ```
  uv run python -m apps.eval.posterior_trajectories --runs-dir cluster_runs --output-dir writeup/results
  ```

Four PNGs, answering `week_2_plan.md`'s "use the real training data, not just
synthetic limits" note directly: what do $(p,q,\varepsilon)$ or $\kappa$
actually converge to during real MNIST training, not just under a guessed
limit path.

## `tnbbeta_posterior_trajectories.png`

$p,q,\varepsilon$ vs. training step, `mnist_tnbs_d{2,5,10,20,40}`, 5 seeds
each.

- $p\to1$ and $q\to0$ essentially immediately at every dimension, as already
  established -- the new information here is *how fast* (within the first
  few thousand steps, well before the 100-epoch KL warm-up finishes) and
  *how tight* across seeds (no visible seed-to-seed spread once settled,
  except the note below).
- At $d=20$ and $d=40$, a faint second band is visible in the $p$ row near
  $p\approx0$ with no corresponding second band in $q$ (which stays
  cleanly at $\approx0$ for all 5 seeds). Most likely explanation: one seed
  landed on the $(\mu,p)\sim(-\mu,1-p)$ alias of the same physical cap
  (CLAUDE.md), not a qualitatively different outcome -- not independently
  confirmed per-seed here, worth checking directly if it matters later.
- $\varepsilon$ climbs with dimension and roughly tracks it at low-to-mid
  $d$ (settling around 2.5-3 at $d=2$, 5-6 at $d=5$, 10-11 at $d=10$,
  20-23 at $d=20$) before clearly overshooting it at $d=40$ (settling
  around 55-60, roughly $1.4d$). Relevant to the joint $(p,\varepsilon)$
  limit-path question in the Theory section: whatever the right scaling
  ansatz is, it needs to explain epsilon tracking $d$ at low dimension and
  then growing faster than $d$ by $d=40$.
- **MNIST never leverages TNBBeta's bimodal capacity -- confirmed precisely
  against the proven threshold, not just eyeballed from the chart above.**
  These runs use ambient dimension $d+1$ (`tnbs`'s convention, CLAUDE.md), so
  the bimodality boundary (`tnbbeta_vs_power_spherical_expressivity.md`,
  Theorem 5.1/Corollary 5.3) is $m=\varepsilon-\tfrac{(d+1)-1}{2}=\varepsilon-\tfrac{d}{2}<0$.
  Averaging each dimension's final `posterior_epsilon_mean` over its 5 seeds
  (same `cluster_runs/` data as above) and comparing against $d/2$:

  | $d$ | $\varepsilon$ (mean, final) | threshold $d/2$ | $m$ | $\varepsilon/(d/2)$ |
  |---|---|---|---|---|
  | 2 | 2.69 | 1.0 | +1.69 | 2.7x |
  | 5 | 5.63 | 2.5 | +3.13 | 2.3x |
  | 10 | 10.56 | 5.0 | +5.56 | 2.1x |
  | 20 | 22.19 | 10.0 | +12.19 | 2.2x |
  | 40 | 57.51 | 20.0 | +37.51 | 2.9x |

  $m$ is strongly positive at every dimension tested -- $\varepsilon$ doesn't
  just clear the unimodal threshold, it settles at roughly **2-3x** it,
  consistently, regardless of whether $\varepsilon$ is tracking $d$ linearly
  (low/mid $d$) or growing faster than $d$ ($d=40$). This is the exact
  opposite regime from `dblp_bridge_diagnostic_2026-10-01.md`'s corrected
  shape diagnostic, where `frac_bimodal` $=1.000$ for every com-DBLP node,
  bridge or not: MNIST's digits apparently never need TNBBeta's extra
  (bimodal) capacity at all, while com-DBLP's featureless co-authorship graph
  always does, independent of a node's own community count. Whatever
  determines which regime a dataset lands in looks like a property of the
  task/data (a 10-class image classification likelihood vs. a featureless
  graph's structural link-prediction likelihood), not of per-node structure
  within one dataset -- community count only modulates *how far* into the
  already-universal bimodal regime a com-DBLP node sits (`q`, `m`), not
  *whether* it's there, mirroring how dimension here only modulates how far
  into the unimodal regime MNIST sits, not whether it's there.

## `kappa_trajectories.png`

$\kappa$ vs. training step, `mnist_{vmfs,pss,vmfks}_d{2,5,10,20,40}`, 5
seeds each, overlaid per dimension.

- All three show the same qualitative shape: a sharp overshoot in the first
  few thousand steps (plausibly while the KL weight is still ramping up,
  so concentrating is nearly free), then a decay down to a lower, stable
  plateau once the full KL weight engages.
- **Power Spherical's $\kappa$ settles at roughly 1.5-2x vMF's/vMF-kappa-
  init's absolute value at $d=2,5,10$** (e.g. $d=5$: `pss` $\approx$
  2200-2500, `vmfs`/`vmfks` $\approx$ 1000-1200), despite the three being
  statistically tied on every Table 1 metric at these dimensions
  (`mnist_table1_with_power_spherical_2026-10-01.md`). Consistent with the
  two families' $\kappa$ not being the same unit even though they share a
  name -- Power Spherical's density is $\propto(1+\mu^\top x)^\kappa$ vs.
  vMF's $\propto e^{\kappa\mu^\top x}$, different functions of $\kappa$ near
  the pole. At $d=20,40$ the three converge much closer together.

## `fixed_epsilon_ablation_trajectories.png`

$q$ vs. training step, `mnistfix_eps{0.5,1.0,1.5}_d5`, 3 seeds each, with
each line's actual endpoint marked (see below for why that matters).

Matches the previously-reported final values exactly, now as full
trajectories: $\varepsilon=0.5$'s three seeds all race to $q\approx1$ and
crash within the first $\sim$2,000-10,000 steps (marked endpoints); $
\varepsilon=1.0$ settles around 0.95-0.97; $\varepsilon=1.5$ around 0.85-0.88.

## `fixed_mu_ablation_trajectories.png`

$p$ and $q$ vs. training step, `mnistfixmu_tnb_d5`, 3 seeds.

**Building the endpoint markers surfaced a real finding the first draft of
this chart silently hid.** `mnistfixmu_tnb_d5_seed1` diverges catastrophically
at step 3,119 (`epsilon`$\to33{,}045$, `kl`$\to-125{,}926$ -- numerically
broken) to the *opposite* corner ($p\to\sim10^{-6}$, $q\to\sim0.999999$),
while seeds 0 and 2 converge "safely" to $p\to1,q\to0$ and train to
completion ($\sim$150,000-170,000 steps). Without the endpoint marker, seed
1's entire trajectory is compressed into under 2% of the x-axis against the
other two seeds' much longer runs and is visually indistinguishable from
their shared initial transient -- the chart would have quietly shown "all 3
seeds agree" when they don't. This is exactly the previously-reported
"2 of 3 seeds land at one corner, 1 of 3 at the other, numerically
catastrophic" finding, now directly visualized with the actual crash point
identified (step 3,119, not just "early").

**Takeaway for chart-building generally, not just this one:** always mark
or otherwise surface where a line's data actually ends when overlaying runs
of very different lengths on a shared step axis -- an early-terminated run
is easy to construct invisibly otherwise.
