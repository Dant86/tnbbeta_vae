# TNBBeta's posterior geometry converges to vMF's, 2026-09-23

Synthesis of every geometry-adjacent result gathered so far, plus the fixed-epsilon
ablation's numbers (not previously written up). Written to answer one question directly:
is "TNBBeta's trained posterior geometry converges to something vMF-equivalent, so the
task never exploits its extra expressivity" a claim the evidence actually supports?

## The claim

A strictly more expressive family (TNBBeta: mean direction + p + q + epsilon) than vMF
(mean direction + kappa), optimized end-to-end on real tasks, converges onto the exact
sub-family that is shape-equivalent to vMF -- and keeps converging there even when the
easiest route to it is deliberately blocked.

## Evidence, in order of how direct it is

**1. Direct measurement of the posterior's own shape parameters (most direct).** The
latitude diagnostic (`apps/eval/svae_latitude.py`) reads `p` and `q` straight off trained
MNIST posteriors, no downstream task in between. Free (unconstrained) training:

| run | p_mean | q_mean |
|---|---|---|
| free epsilon (baseline, `tnbs_d5_*`) | ~0.999 | ~1e-6 (clamp floor) |

`q -> 0` is not "small" in any loose sense -- it is the exact special case
(`uniform_prior_params`'s own derivation uses it) at which the univariate TNBBeta
collapses to a plain `Beta(epsilon, epsilon)`, and `p -> 1` (a pole) is where that Beta's
mass concentrates into a single tight cap around the mean direction. That shape -- one
direction, one concentration -- is exactly vMF's parameterization. This is a first-hand
look at the object in question, not an inference from behavior.

**2. Downstream metrics tie vMF everywhere this shape would predict they should.** Table
1 (density), Table 2 (direction k-NN), the confidence-only probe, semi-supervised, and
link prediction all show TNBBeta matching vMF/vmfk at every healthy dimension. Consistent
with claim 1, but on its own this is only suggestive -- tied outcomes don't by themselves
prove a shared mechanism.

**3. The fixed-epsilon ablation (causal, not just observational) -- new results below.**
This is the one experiment that actively tried to force the posterior away from the
cap shape rather than just observing where it settles by default.

### Fixed-epsilon ablation results

Setup: `fixed_epsilon in {0.5, 1.0, 1.5}`, MNIST, d=5 manifold (ambient 6), 3 seeds,
`scripts/slurm/fixed_epsilon_sweep.sbatch`. Rationale for testing exactly these three
values: the univariate TNBBeta's boundary density has exponent `(epsilon - 1)` near
`y -> 0, 1`, giving three qualitatively distinct regimes (see
`docs/` derivation / PR #29's commit message): `epsilon < 1` unbounded/U-shaped at the
poles, `epsilon = 1` flat baseline, `epsilon > 1` unimodal and vanishing at the poles.

| epsilon | outcome | p_mean | q_mean |
|---|---|---|---|
| 0.5 | **crashed** (`FloatingPointError`, all 3 seeds) | -- | -- |
| 1.0 | trained cleanly | 0.9989 | 0.9449 |
| 1.5 | trained cleanly | 0.9990 | 0.8707 |

**The eps=0.5 crash is not a bug.** At `epsilon < 1` the boundary term behaves like
`y^(epsilon - 1)` near `y=0` (and symmetrically near `y=1`), which diverges as `y -> 0`
since the exponent is negative -- the density has no finite supremum there. The training
objective is trying to maximize an unbounded log-density, so gradient ascent has no
optimum to converge to; it simply runs off toward the singularity until floats overflow.
Gradient clipping was tried and rejected here (see "What we decided not to do," below) --
it would only slow the walk toward a point that doesn't exist, not produce a meaningful
result.

**At eps=1.0 and eps=1.5, `q` substitutes for the epsilon it lost.** With epsilon no
longer free, `q` moves off its near-zero floor (0.94, 0.87) to supply the "how
concentrated" signal epsilon used to provide. This is exactly the substitution effect the
ablation was designed to detect. The open question was *what shape* that substitution
produces -- does `q > 0` under fixed epsilon build a genuinely different (ring/bimodal)
posterior, or does it just rebuild the same cap by another route?

Checked numerically (`TNBBetaUnivariate` sampled directly at the fitted parameters, 500k
draws, `w = 2y - 1` = cosine to the mean direction):

| condition | P(w > 0.9) | P(w < -0.9) | mean(w) |
|---|---|---|---|
| eps=1.0 fitted (p=0.999, q=0.945) | 0.9989 | 0.0000 | 0.9971 |
| eps=1.5 fitted (p=0.999, q=0.871) | 0.9998 | 0.0000 | 0.9975 |
| free-epsilon baseline (p=0.999, q~0, eps=2.0) | 0.9998 | 0.0000 | 0.9967 |

All three put >99.8% of the posterior's mass within `cos > 0.9` of a single pole, with
*zero* mass near the antipode. `q` does not rebuild a ring or an antipodal pair of blobs
when epsilon is denied it -- it rebuilds the same single tight cap, just via a different
one of the family's three extra parameters. This refutes my own earlier hypothesis (that
the eps=1.0 `q_mean=0.945` result might be the bimodal "cap vs ring" regime documented in
week-0 CIFAR d=3 findings); I ran the check specifically because that guess needed
verification before being stated as a finding, and it did not survive it.

## Reading it together

Claims 1-3 triangulate on the same conclusion from three independent angles (a direct
parameter reading, tied downstream outcomes, and a targeted intervention), which is why I
think "the geometry is convergent" is a well-supported, worth-pursuing framing rather than
a post-hoc gloss on tied numbers:

- The model is not merely *near* vMF's shape by default -- it is *steered back* to it when
  the cheapest path there (epsilon) is removed, actively reallocating onto `q` to rebuild
  the identical cap rather than exploring the qualitatively different geometry (ring,
  bimodal) the family can represent in principle.
- The one region where TNBBeta doesn't just tie vMF -- epsilon carrying real class signal
  at d=40 that beats vMF's kappa (`mnist_confidence_epsilon_full_2026-09-23.md`) --
  doesn't contradict this. It's still one scalar doing kappa's job, just a different named
  scalar at higher dimension; it does not involve `p`/`q` producing ring/bimodal geometry
  either.
- The one regime where a qualitatively different shape *is* mathematically available
  (`epsilon < 1`) turns out to be untrainable, because the objective has no finite
  supremum there -- so even the family's one genuine escape route from cap-shaped
  geometry is not a route gradient descent on this task could ever take.

## Update 2026-09-26: quantified, not just visualized

`mnist_concentration_ring_check_2026-09-26.md` turns the Hammer figure's visual
impression into numbers, at every dimension in the sweep (not just d=2), in two parts:
a ring-exclusion check (does either family's per-class spread show any antipodal/ring
mass -- answer: no, for both, at every dimension from 5 upward, exactly 0.0% across
all 50 runs) and a concentration comparison (`r_bar` and its vMF-equivalent kappa --
tied at every dimension once a known `vmfs` d=40 partial-collapse artifact is
accounted for). LaTeX-formatted tables there, ready to paste into the paper.

**Caveats worth stating plainly:** all of this is MNIST-specific and only tested at d=5
(and, for claims 1-2, d=2/5/10/20/40) with a conv encoder/decoder -- it is not yet known
whether the same convergence holds on a different dataset/architecture, or whether it is
in some way an artifact of MNIST's class structure being well suited to single-mode
per-class encoding.

## What we decided not to do

Gradient clipping (`torch.nn.utils.clip_grad_norm_`) was implemented and tested as a
possible fix for the eps=0.5 crash, then shelved once the crash was understood to be an
expected consequence of an unbounded objective rather than a bug: clipping would slow
divergence, not produce convergence to a real optimum. Not pursued further; the branch
this lived on was not merged.

## Next lever: fixing the mean direction

If direction is the escape hatch that lets `p`/`q`/`epsilon` stay uninformative (per
claim 1, direction alone already carries the discriminative signal at low/mid dimension),
removing that hatch is the sharpest remaining test of whether TNBBeta's extra parameters
are usable when they're the *only* thing available. See
`scripts/slurm/fixed_mean_direction_sweep.sbatch` (TNBBeta only, d=5, 3 seeds) for the
first pass at this.
