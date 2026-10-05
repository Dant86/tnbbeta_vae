# Week 2 plan: TNBBeta/S-VAE convergence, TNBBeta/Power-Spherical divergence

Two headline goals for the week, both anchored in the week 1 finding that
TNBBeta-VAE and S-VAE posteriors converge to the same cap-shaped geometry on
MNIST:

1. Show, theoretically **and** empirically, that in this MNIST setting the
   S-VAE's and TNBBeta-VAE's posteriors genuinely converge to each other (not
   just "tie on aggregate metrics").
2. Identify concrete points of *divergence* between the TNBBeta-VAE and the
   Power Spherical VAE (De Cao & Aziz, 2020) — the natural comparison once
   convergence-to-vMF is established, since Power Spherical is also
   rejection-free and also has a closed-form KL, so the TNBBeta's case has to
   rest on something other than "it's not vMF."

## Status board

| # | Item | Avenue | Kind | Status |
|---|---|---|---|---|
| 1 | Implement Power Spherical distribution + VAE baseline | 2 | implementation | **done** — PR #39/#40, branch `feat/power-spherical-distribution`, merged |
| 2 | Shared $(W,V)$-decomposition + $p\to1$ local limit theorem | 1 | theory | in progress (user) — see the real empirical constraint below, found this week |
| 3 | Two-sample tests + trained-posterior overlay | 1 | empirical | **done (Part 1)** — [PR #43](https://github.com/Dant86/tnbbeta_vae/pull/43), merged. Part 2 (trained-posterior overlay) still pending, see below |
| 4 | Non-containment + provable expressivity gap | 2 | theory | **done, fully proved** — `tnbbeta_vs_power_spherical_expressivity.md` in this directory |
| 6 | Rerun MNIST suite with Power Spherical as a 4th model | 2 | empirical | **done** — `writeup/results/mnist_table1_with_power_spherical_2026-10-01.md` |
| 8 | Fix `GraphVAE`'s dense-feature bottleneck + com-DBLP loader | 2 | engineering | **done** — merged into PR #42 |
| 7 | com-DBLP link prediction (real overlapping communities) | 2 | empirical | pipeline done and merged (PR #42, PR #44) — **not yet actually run**, see commands below |
| ~~5~~ | ~~Closed-form $\mathrm{KL}(\mathrm{TNBBeta}(p,0,\varepsilon)\|\text{Uniform})$~~ | 2 | theory | **dropped** — no clear payoff over the existing MC-KL |

## Avenue 1 — TNBBeta $\leftrightarrow$ S-VAE convergence

### Theory

Both vMF and SphericalTNBBeta are axially-symmetric about their mean
direction, so both admit the same decomposition Algorithm 1 already uses:
draw a scalar height $W\in[-1,1]$ from some family-specific marginal, draw
$\mathbf{v}$ uniformly on $S^{d-2}$, form $\mathbf{x}=(W,\sqrt{1-W^2}\mathbf{v})$,
Householder-reflect to $\boldsymbol\mu$. That means "are the two posteriors
similar" reduces to a 1-D question: how close are their two $W$-marginals?

- vMF's $W$-marginal is the standard closed form,
  $f_W^{\mathrm{vMF}}(w) \propto e^{\kappa w}(1-w^2)^{(d-3)/2}$.
- TNBBeta's $W$-marginal follows directly from the closed-form density
  (`tnbbeta_from_beta_derivation.md` in this same directory) via $w=2y-1$.

**The limit to derive:** at $q=0$ and fixed $\varepsilon$, sending $p\to1$
degenerates $T_p$ into a step function and concentrates $Y$ near $1$ (i.e.
$W$ near the pole) — this is the correct knob for *location*, not
$\varepsilon\to\infty$, which concentrates $Y$ at $y=p$ wherever $p$ happens
to sit, not necessarily at the pole.

**Update, found this week (PR #43): $p\to1$ at *fixed* $\varepsilon$ is not
enough to also match *shape*.** Moment-matching TNBBeta($p,0,\varepsilon$) to
vMF($\kappa$) via $\bar r$ and comparing $w$-marginals: energy distance
shrinks ~80% as $\kappa:5\to1000$ (both families collapsing toward the same
point), but the KS statistic *grows* 125–165% over the same range, and
doesn't improve after standardizing each distribution by its own std. Direct
check on $1-W$: vMF's std $\approx$ its own mean at every $\kappa$ (exactly
the known $\mathrm{Gamma}(\tfrac{d-1}2,\theta)$/exponential local limit for
$\kappa\to\infty$); TNBBeta's std is **16–20$\times$ its own mean** at
$\kappa=1000$, fixed $\varepsilon=1$ — not Gamma-shaped at all. So fixed-$\varepsilon$
$p\to1$ gets the right *point* but lands in a different universality class
for the *tail shape*. Working hypothesis, unverified: the correct limit is
**joint**, $p\to1$ and $\varepsilon\to\infty$ together at some matched rate
(e.g. $p=1-c/N,\ \varepsilon=dN$ for constants $c,d$ to solve for as
$N\to\infty$), not $p\to1$ alone — worth treating "what joint scaling
recovers a Gamma-shaped limit" as the concrete target rather than guessing a
single-parameter path.

**Also use the real training data, not just synthetic limits.** Every
TNBBeta/vMF/Power-Spherical training run already logs its posterior
parameters at *every step* (not just the final checkpoint) to
`runs/<run_id>/metrics.jsonl` via `RunLogger` — see
`src/tnbbeta_vae/models/diagnostics.py` (`tnbbeta_spherical_posterior_diagnostics`:
`posterior_p_mean/min/max`, `posterior_q_mean/min/max`, `posterior_epsilon_mean`)
and the `posterior_kappa_mean` field vMF/Power Spherical's own `training_step`s
log. This is sitting on the cluster for every completed MNIST sweep run,
unanalyzed beyond final-checkpoint snapshots so far — pulling the full
trajectories (what $(p,\varepsilon)$ pair, or what $\kappa$, training actually
converges to at each dimension, and how fast) gives a real empirical anchor
for whatever joint scaling path the theorem above needs, instead of guessing
one abstractly. `runs/` is gitignored/local to wherever training ran (the
cluster), not something this repo's checked-out code can read directly.

Once a candidate limiting shape is in hand, the right empirical test is to
redo the two-sample comparison on the **correctly rescaled** tail variable
(whatever the derivation says to divide $(1-W)$ by), not on raw $w$ — the raw
KS-growing result above only shows the two finite-parameter distributions
aren't already identical, which was never the claim; a limit theorem is a
statement about the rescaled shape, not the unscaled one.

### Empirical

**Part 1, done:** [PR #43](https://github.com/Dant86/tnbbeta_vae/pull/43) —
the two-sample test above (energy distance + KS on the $w$-marginal, 4
dimension/$\varepsilon$ combinations, $\kappa\in\{5,...,1000\}$), with a real
review pass that caught and fixed 4 correctness bugs in the first version
(a silent high-$\kappa$ ceiling on the bisection, non-reproducible/noisy
matching that produced spurious non-monotonic rows, too-coarse a tolerance
that collapsed distinct $\kappa$'s onto the same matched $p$, and a silent
tensor-axis bug). The corrected result is the energy-distance-shrinks/KS-grows
finding written into the Theory section above, not the original version's
unqualified "convergence confirmed" — that claim didn't survive the fixes.

**Part 2, still pending:** overlay the **trained** MNIST posteriors'
$w$-marginals at each dimension already in the sweep (TNBBeta vs. vMF vs. now
also Power Spherical, now that the `pss` sweep is done too), and report the
KL/JS divergence between the fitted densities directly. The script for the
TNBBeta/vMF half already exists (`apps/distributions/vmf_tnbbeta_convergence.py`,
Part 2) but auto-skips when it can't find checkpoints — it needs to actually
run somewhere with `$TNBBETA_CHECKPOINT_DIR` access (the cluster), and should
be extended to a three-way comparison while at it.

## Avenue 2 — TNBBeta vs. Power Spherical

### Implementation — done

PR #39/#40 (`feat/power-spherical-distribution`, merged; GitHub shows the
merge as "#40" for the same branch):

- `src/tnbbeta_vae/distributions/power_spherical.py` — `PowerSpherical`,
  verified against the actual paper (arXiv:2006.04437, Theorems 12/13/15/17):
  density $\propto(1+\boldsymbol\mu^\top\mathbf{x})^\kappa$, sampler via
  $z\sim\mathrm{Beta}(\tfrac{d-1}2+\kappa,\tfrac{d-1}2)$ then the same
  Householder reflection `TNBBetaSpherical` already uses, closed-form KL to
  Uniform($S^{d-1}$). $\kappa=0$ confirmed (algebraically and numerically)
  exactly uniform.
- `ConvPowerSphericalVAE`, wired into `heads.py`'s `family` pattern and the
  shared `posterior_and_prior`/`log_likelihood` interface the Table 1
  pipeline uses.
- `apps/eval/svae_knn.py`, `confidence_probe.py`, `svae_concentration.py`
  updated to recognize `conv_power_spherical_vae` (previously would have
  crashed or silently skipped it).
- `scripts/slurm/mnist_sweep.sbatch` extended with an 8th model row (`pss`,
  ambient $d+1$, matching the paper-convention sphere models already in the
  table).
- 382 tests passing, ruff/pyright clean.

### Theory — non-containment and a provable expressivity gap — done, fully proved

Full derivation, proofs, and numerical verification:
`tnbbeta_vs_power_spherical_expressivity.md` in this directory. Both families
agree exactly at one point: TNBBeta's $q=0,p=0.5,\varepsilon=\tfrac{d-1}2$
special case and Power Spherical's $\kappa=0$ are both exactly
$\mathrm{Uniform}(S^{d-1})$ (see `uniform_prior_params`'s docstring). Summary
of what's now proved, not just conjectured:

**Non-containment: proved exactly**, not just numerically suggested
(Theorems 1.1–2.1, Corollary 2.3). $\mathrm{TNBBeta}(p,q,\varepsilon) =
\mathrm{Beta}(a,b)$ **iff** $p=\tfrac12,\ q=0,\ a=b=\varepsilon$ — a clean
normal-form argument (verified to $2.7\times10^{-13}$ relative error against
`TNBBetaUnivariate.log_prob`), extended to the spherical families
intersecting in **exactly** $\{\mathrm{Uniform}(S^{d-1})\}$, non-nested in
both directions, covering $q>0$ too.

**Expressivity gap: proved (Theorem 6.1), but the mechanism is $\varepsilon$,
not $q$ — correcting an error from earlier this week.** $q$ only sharpens or
relocates a single mode; it does **not** produce bimodality. The real
trigger is a trichotomy in $m=\varepsilon-\tfrac{d-1}2$ (Theorem 5.1):
$m>0$, no pole modes; $m=0$, at most one; **$m<0$, the density diverges at
*both* poles**, genuinely bimodal at $\{\mu,-\mu\}$. `pq_ablation.png` fixes
$\varepsilon=1$ on $S^2$ — exactly $m=0$ — so **every cell of that grid is
provably unimodal**; cite `p_epsilon_ablation.png`'s $\varepsilon\in\{0.5,0.8\}$
block instead, which is the one that's actually bimodal. A striking certified
example worth featuring visually: $(p,q,\varepsilon)=(0.85,0.95,0.3)$, $d=3$,
is **trimodal** — a pole mode, an interior ring at $43.6°$, and an antipodal
mode simultaneously (lobe masses $0.103/0.723/0.173$, exact quadrature and
4M-point Monte Carlo agreeing to 4 decimals).

The Power Spherical side also needed a fix: the original log-concavity
argument (unimodal for $a,b>1$) **fails at $d=2$** specifically — $d=2$ gives
$\mathrm{Beta}(\tfrac12+\kappa,\tfrac12)$, which is genuinely U-shaped for
$\kappa<\tfrac12$, and $d=2$ is a real sweep row, not a corner case. Fixed
with something better and dimension-free (Proposition 4.1): Power Spherical's
spherical density is directly $\propto(1+\mu^\top x)^\kappa$, monotone in the
cosine, hence strictly unimodal for *every* $\kappa$ and *every* $d\ge2$ — no
log-concavity argument needed at all.

**Action item before the com-DBLP run (#7) means anything:** `uniform_prior_params`
sits exactly at $m=0$ — the prior boundary. Any posterior head that keeps
$\varepsilon\ge\tfrac{d-1}2$ can never be multimodal regardless of $p,q$.
Checked `heads.py`: `epsilon = softplus(raw) + eps_floor` has no hard floor
at $\tfrac{d-1}2$, so it's not structurally blocked — but nothing currently
logs whether training actually pushes $\varepsilon$ below it.
`models/diagnostics.py`'s `tnbbeta_spherical_posterior_diagnostics` logs
`epsilon`'s *mean* only; add `epsilon_min` and `m = epsilon - (dim-1)/2`
there before the real cluster run, so the bridge-node result (if tied) can be
read as "the model never found the bimodal regime" vs. "it found it and it
didn't help" — two very different conclusions that look identical in the
accuracy numbers alone.

### Empirical — rerun the MNIST suite with Power Spherical — done

`sbatch --array=210-234 scripts/slurm/mnist_sweep.sbatch` run; results in
`writeup/results/mnist_table1_with_power_spherical_2026-10-01.md`. Headline:
Power Spherical ties TNBBeta/vMF everywhere, *including* $d=40$ (the one
dimension reference vMF collapses at) — zero aggregate Table-1
differentiation anywhere, which is exactly why the theory section above and
the com-DBLP run below are now the entire remaining case for TNBBeta over
Power Spherical, not a bonus on top of a win.

### Empirical — a real (non-synthetic) task that needs TNBBeta's expressivity

MNIST can't show the expressivity gap because it has no genuine decoder-side
ambiguity: no image needs two far-apart, mutually exclusive latent
explanations averaged badly by a single cap, so a wider unimodal cap is
always at least as good as splitting mass, and training never pays the KL
cost of a ring/bimodal posterior. The property we actually need in a dataset
is a genuine identifiability ambiguity or mixed-membership structure.

Considered and set aside (see "Discarded directions" below for the reasoning
on each): a hue-symmetry variant of `gaussian_blob_batch`, rotated MNIST on
rotationally-symmetric digits, protein side-chain dihedral angles. Settled
on, because it stays VAE-centric and reuses the existing link-prediction
pipeline almost as-is:

**com-DBLP, a real co-authorship network with documented ground-truth
overlapping communities** (SNAP; verified against the live dataset page, not
recollection: 317,080 nodes, 1,049,866 edges; communities are defined by
publication venue, so an author who publishes in multiple venues is a
genuine, externally-validated multi-community node — not something we have
to argue should exist).

**Why link prediction, as currently run on Cora/Citeseer/Pubmed, wouldn't
show this even if it mattered:** those are strongly assortative citation
graphs (most nodes sit in one dominant topical community), and AUC/AP is an
aggregate metric dominated by the many unambiguous same-community pairs — the
same "diluted aggregate" pattern that hid the expressivity gap in Table 1 and
in far_side_pct/antipodal_pct. A real effect on a minority of bridging nodes
wouldn't move the aggregate number regardless of whether it exists.

**Three real blockers found this week, all now fixed and merged:**

1. `GraphVAE`'s first layer was a dense `nn.Linear` applied directly to
   `batch.features`. A 317,080-node one-hot identity feature matrix (the
   standard fix for a featureless graph) would have materialized as a dense
   $317{,}080\times317{,}080$ float32 tensor — about 402GB, which OOMs on an
   A100 (80GB) or an H200 (141GB) identically, since this is a representation
   problem, not a compute-speed one. Fixed: the first layer routes through
   `torch.sparse.mm` when `batch.features` is sparse, the same trick already
   used for the adjacency multiplication two lines below (PR #42).
2. No node features at all in the raw SNAP data (unlike Planetoid's
   bag-of-words) — a sparse identity matrix is the loader's feature
   substitute (`src/tnbbeta_vae/data/snap_community.py`).
3. **Found later, while preparing the actual run commands:** `heads.py` has
   supported `power_spherical` since PR #39/#40, but
   `apps/link_prediction/main.py --family` was never updated to accept it —
   the two PRs landed independently and neither referenced the other. Fixed
   in PR #44 (one line). Without this, the 3-way comparison below would have
   silently only run vMF and TNBBeta.

The bridge-node diagnostic itself also shipped with two real placeholder bugs
in its first pass (a random 50/50 primary/secondary community split instead
of real edge-counting, and a node-ID remapping that assumed sorted order
instead of the loader's actual first-seen-in-edge-list order) — both caught
and fixed with regression tests before merging (PR #42's final state).

**Decided: train on the full graph, not an induced subgraph.** The original
plan was to induce a subgraph from the top-quality community file to keep
things Pubmed-scale, but that's solving a problem that mostly goes away once
#8's sparse-matmul fix lands: once features are handled sparsely, com-DBLP's
~2.4M-nonzero normalized adjacency is small by modern GNN standards, and a
full-batch epoch should cost roughly 0.1-1 second on an A100 (dominated by
Python/dispatch overhead, not FLOPs) — a few minutes of GPU time for a full
training run, not hours. Training on the full graph also avoids a subtler
problem with subsetting: SNAP's "top 5,000 by quality" communities are
selected for good separation/low conductance, which likely *anti-correlates*
with overlap — a subgraph built from them could easily end up with less
overlap than Cora, undermining the whole point.

**Bridge-node identification:** use `com-dblp.all.cmty.txt.gz` (every
community SNAP has, not the curated top-5000-by-quality file) to count how
many communities each node belongs to. Nodes with multiplicity $\geq2$ are
the genuine multi-venue authors — the targeted population for the diagnostic
below.

**Targeted diagnostic (the actual point, not just an AUC number):**
analogous to `svae_concentration.py`'s per-class breakdown, but keyed on
community-membership count instead of MNIST class label.

- Does a bridge node's TNBBeta posterior show more spread/bimodal structure
  than a single-community node's, in a way vMF/Power Spherical structurally
  cannot?
- Does link-prediction accuracy *specifically on a bridge node's edges into
  its secondary community* improve for TNBBeta over vMF/Power Spherical? This
  is the metric that actually tests the hypothesis — the aggregate AUC/AP is
  expected to stay tied, same as it did on Cora/Citeseer/Pubmed.

**Scope control:** don't replicate the full Planetoid hyperparameter grid
(`lrs \times dropouts \times latent\_dims \times 5` seeds $=135$
configurations) — this is a targeted follow-up experiment, not a core
baseline reproduction. A much smaller grid (1 learning rate, 1-2 dropout
settings, 2 latent dims, 3-5 seeds; $\sim$10-20 runs) is enough, and should
be smoke-tested for per-epoch timing on the cluster before committing to a
full run, to confirm the 0.1-1s/epoch estimate above.

**Not yet run.** The pipeline is merged (PR #42, #44) but nobody has actually
executed it on real data yet — see the commands in the main conversation
for the download, timing smoke test, and the actual 3-family $\times$ 2
latent-dim $\times$ 5-seed grid. Consider adding the `epsilon_min`/`m`
diagnostic logging flagged in the Theory section above before committing to
the full run, not after.

## Discarded directions (kept for the record)

- **Closed-form $\mathrm{KL}(\mathrm{TNBBeta}(p,0,\varepsilon)\|\text{Uniform})$
  via the entropy-transform trick.** No clear payoff identified over just
  using the existing Monte Carlo KL — dropped rather than pursued as a
  "nice to have."
- **A hue-symmetry variant of `gaussian_blob_batch`** (render color invariant
  to hue $\to$ hue+$\pi$, forcing a provably antipodal true posterior).
  Clean and minimal-engineering, but synthetic/constructed-to-order; set
  aside once a real dataset with the same property was in reach.
- **Rotated MNIST, restricted to rotationally-symmetric digits (0, 8, maybe
  1).** A real within-MNIST contrast (genuinely bimodal for symmetric digits,
  still unimodal for the rest) would have been a nice selective-activation
  result, but needs the rotation angle itself as the encoded latent factor
  rather than reusing the existing classification-style pipeline — more
  structural work than the SNAP-graph route for a comparable payoff.
- **Protein side-chain dihedral angles (rotamer distributions).** The
  strongest "externally-validated ground truth" option of everything
  considered (rotamer states are an established structural-biology fact, not
  something we have to argue for) and doesn't need a VAE at all in its
  simplest form (direct MLE fit of TNBBeta/vMF/Power Spherical to observed
  angles). Set aside only because the paper's framing is VAE-centric and the
  SNAP-graph route stays inside the existing link-prediction pipeline;
  worth revisiting if the SNAP result turns out weak or ambiguous.

## Open decisions

- **com-DBLP vs. com-Amazon.** Defaulted to DBLP for thematic continuity with
  the citation-network baselines (co-authorship sits next to citation
  naturally); com-Amazon's product-category communities are the fallback if
  DBLP's overlap structure turns out to be weaker than expected once actually
  measured.
