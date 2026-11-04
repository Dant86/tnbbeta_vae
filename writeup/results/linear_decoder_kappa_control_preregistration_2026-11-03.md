# Linear-decoder ring: tuned-kappa control for the vMF / PowerSpherical caps. Pre-registration

Status: **FROZEN** (2026-11-03, before any run of the new code). Later changes are appended under "Deviations" only.
Base commit: `dd8ede0` (branch `synthetic-graph-families`) plus the uncommitted working tree (new files only, listed under "Code").
Builds on: `linear_decoder_ring_preregistration_2026-11-02.md` and `linear_decoder_ring_results_2026-11-02.md` (decoder `L`, same DGP, 480 confirmatory runs; not redone).

## Why this control

In the parent study TNBBeta beat vMF / PowerSpherical (PS) in 19 of 24 confirmatory cells with a strictly linear learned decoder. But the caps
*collapsed toward the prior* (fitted kappa about 1.0-1.9, `sigma_hat` inflated up to 5x), and the parent's own caveat is that this may be an
**optimisation** outcome (kappa initialised at `d`, 300 epochs, Adam 1e-3, kappa trajectory not logged) rather than a limit of the cap *family*. If a
well-tuned cap closes the gap, "TNBBeta beats caps with a linear decoder" would be a statement about kappa optimisation. This study asks exactly that,
and null or negative outcomes (the advantage disappears) are legitimate and will be reported as such. No outcome is assumed.

## Primary question

Does TNBBeta's test-ELBO-gap advantage survive against the **best-tuned cap**, where the cap's concentration is tuned per `(d, sigma)` cell?

## Task, data, metrics (identical to the parent)

Same DGP (`z ~ Uniform(S^(d-1))`, `y = e_1 . z + N(0, sigma^2)`), decoder `L` only (`y_hat = u . z`, no bias, learned scalar `sigma_hat` started at 0.2),
`d in {3, 5, 9, 17}`, `sigma in {0.1, 0.2, 0.4}`, 12 cells; 20,000 training `y` (generator seed `100 + seed`), 2000 shared test `y`; seeds 0-9;
encoder MLP `[1,128,128]`, one linear posterior head; Adam, batch 256, 4 ELBO draws, KL warm-up over the first 10% of epochs, analytic KL for caps;
float32 training, float64 CPU evaluation; primary metric the exact test ELBO gap `mean_y[log p_true(y) - ELBO(y)]` (256 draws), plus IWAE gap (K = 64),
the exact decomposition (model misfit + KL to the model's ring posterior), `sigma_hat`, `|centre . u|`, ring statistic, fitted kappa. All of this is the
parent's `evaluate` function, imported unchanged. Encoder / decoder initial weights, data and shuffling are identical across arms for a seed (the head is
built last), so arms are paired by seed. The vMF arms use `ExactVonMisesFisher` (the exact-envelope sampler, imported from the parent's module where it
is the local copy) in training and evaluation; the library vMF sampler is never used.

## Arms (new runs)

Parent runs reused **as is** (not rerun; same code and seeds, so already paired): `tnbbeta` (library defaults), `vmf` and `power_spherical` (initial kappa `d`,
trainable, 300 epochs, lr 1e-3), read from `ring_recovery/linear_decoder/L/`.

| id | arms | description |
|---|---|---|
| (a) high initial kappa | `vmf_k5d`, `vmf_k20d`, `power_spherical_k5d`, `power_spherical_k20d` | kappa head bias initialised so kappa = `5d` or `20d`, still trainable; 300 epochs, lr 1e-3 |
| (b) pinned kappa grid | `{vmf,power_spherical}_pin{3,10,30,100,300}` | one constant kappa for every example (head outputs the mean direction only, `d` outputs); absolute grid kappa in {3, 10, 30, 100, 300} (a factor of about 3 apart, 1 up to 300; sigma = 0.1 needs kappa of order `1/sigma^2` = 100 for a thin cap) ; 300 epochs |
| (c) fast / long | `vmf_H`, `power_spherical_H`, `tnbbeta_H` | head learning rate 1e-2 (everything else 1e-3) and **600 epochs** (warm-up 60), initial kappa `d` (TNBBeta: library default). TNBBeta gets the same variant so the budget comparison is fair |

Per-epoch **trajectory** logged for every new run: median over a fixed 512-example training subset of kappa (caps) or epsilon, p, q (TNBBeta) at the end of each
epoch, `sigma_hat`, and the epoch's mean training loss (saved in the run JSON; no RNG consumed). Total new runs: 19 arms x 120 cells(seed x d x sigma) = 2280
(pinned 1200, (a) 480, (c) 360 at double length).

## Tuning rule and primary tests

**Cap pool** for a family (vMF or PS) in a `(d, sigma)` cell: all of the family's arms above plus the parent's initial-`d` arm (the pool does not include any
TNBBeta arm; `_H` caps are in the pool). **Best-tuned cap** = the pool member with the smallest *seed-mean test ELBO gap in that cell* (tuning on the
very metric and seeds that are then tested: deliberately generous to the cap, an in-sample selection, so a TNBBeta win against it is conservative).
Diverged runs are excluded; an arm is eligible only if at least 8 of 10 seeds are valid, the paired test uses seeds valid in both arms, fewer than 5 pairs => no test.

**Primary (confirmatory) tests:** for each `(d, sigma)` (12 cells) and each family (vMF, PS): one-sided paired Wilcoxon signed-rank over seeds of
`best-tuned-cap gap - tnbbeta gap` (parent `tnbbeta`, H1: TNBBeta smaller), 24 tests, Holm step-down at 0.05 over all 24, and a cell is **confirmed** only if
Holm rejects AND the mean paired difference is at least **0.05 nats**. Every p-value is verified by enumerating all `2^n` sign patterns (a mismatch is reported).
Power note: as in the parent, with `n = 10` the smallest one-sided p is `1/1024` and Holm's first threshold is `0.05/24 = 0.00208`; a rejection needs about
10/10 seeds, so a non-rejection is not evidence of no effect and win counts are reported with every test.

**Sensitivity analyses (pre-stated, descriptive, no extra inference family):**
(S1) *Leave-one-seed-out tuning* removes the in-sample selection bias: for each seed `s` the cap arm is the pool member with the smallest mean gap over the
other 9 seeds, and the cap's gap for `s` is that arm's gap at `s`; the same 24 tests and rules are reported as a second table (labelled sensitivity; if its
conclusion differs from the primary table that is stated).
(S2) *Equal budget*: `tnbbeta_H` against the best-tuned cap in the pool restricted to `_H` arms and `{pin*, k5d, k20d}` (the best cap over all arms) — i.e. TNBBeta's best of {default, H} (per cell, same in-sample rule over both TNBBeta arms) against the best cap overall; mean paired differences and win counts only.
(S3) *Pooled families*: the best cap over both families in each cell versus TNBBeta (descriptive).
(S4) Per-arm gap tables; the **location of the best pinned kappa** (a best at the grid edge 3 or 300 is flagged as a grid insufficiency, not as a tuned optimum); fitted
`sigma_hat/sigma`; and the kappa trajectories of the trainable arms (does kappa move from its initial value, and to where).

## Pre-stated interpretation rules (so the reading is not chosen after the result)

* **Advantage survives** in a `(d, sigma)` cell if it is confirmed against both families' best-tuned cap (primary table). **Advantage is an optimisation artefact** in
  a cell if the best-tuned cap's seed-mean gap is within 0.05 nats of (or below) TNBBeta's. Cells in between (positive difference below the margin or not Holm-significant)
  are "inconclusive / small". The headline is the count of cells in each category, per sigma and per d.
* A pinned-kappa cap that wins shows the cap *family* is competitive when its kappa is set correctly; a cap that wins only with trainable high-initial kappa
  shows an optimisation effect (initialisation). Both are reported separately by arm.
* If caps with pinned kappa still show inflated `sigma_hat` and a worse gap, the parent's result is not explained by kappa optimisation.

## Risks and known caveats

* In-sample tuning over 14-15 pool members per family per cell inflates the cap's apparent performance (intended; S1 addresses the bias).
* Float32 training of vMF at large kappa and d = 17: non-finite losses are counted as divergence and reported per arm.
* The kappa grid is absolute (not scaled with d); at d = 17 kappa = 3 is almost the prior and at d = 3 kappa = 300 is very thin. Edge flags (S4) apply.
* Head lr 1e-2 may be unstable; that is a legitimate outcome of the variant.
* Tests are over 10 seeds only; the parent's power note applies. The control selects the cap on the same 2000 test `y` shared by all seeds; this is not an independent test sample.
* GPU contention among worker processes affects wall time only.

## Code (all new files; no existing file is edited)

`apps/synthetic/linear_decoder_kappa_control.py` (arms, trajectory-logging trainer, summarize with the tuning rule and tests; imports `evaluate`,
`LinearRingVAE`, `ExactVonMisesFisher`, tests helpers from `apps/synthetic/linear_decoder_ring.py`), `apps/synthetic/linear_decoder_kappa_control_plot.py`,
`scripts/slurm/linear_decoder_kappa_control.sbatch` (+ `_aux.sbatch`), tests `tests/apps/test_linear_decoder_kappa_control.py`, results note
`writeup/results/linear_decoder_kappa_control_results_2026-11-03.md`. Outputs under `ring_recovery/kappa_control/L/d<d>_s<sigma>/<arm>/s<seed>.{json,npz,pt}` (gitignored).

## Deviations

**D1 (2026-11-03, before any run of the new arms; found by the unit test of the grid size).** Counting error in the text above, not a design change: the arm list
(5 pins + 2 high-init per family = 14, plus 3 `_H`) has **17** new arms, so there are **2040** new runs (pinned 1200, (a) 480, (c) 360 at double length), not "19 arms /
2280". The cap pool for a family has 9 members (5 pins, `k5d`, `k20d`, `_H`, and the parent's initial-`d` arm), not "14-15"; the in-sample selection inflation is
correspondingly smaller than stated. Nothing else changes.

**D2 (2026-11-03, during execution, after about 1600 of 2040 runs had finished; partial results had been looked at).** The shared GPU queue slowed throughput to about
6-10 runs per 10 minutes. To finish in reasonable time the remaining queue was restricted to skip the pinned-kappa arms with kappa in {10, 30, 100, 300}: in every
cell seen so far (at least one seed each) those arms had gaps 0.9-20 nats, far above the trainable and pin-3 arms' 0.01-0.5, so they cannot be the selected best-tuned cap. They
therefore stay at whatever number of seeds had already finished (reported in the arm table as `valid` counts, 6-10 of 10; they are in the pool only if at least 8 seeds finished, and
none of them is ever the selected arm in the result tables, which is checked and stated in the results note). All other arms (`pin3`, `k5d`, `k20d`, `_H` and the parent arms) are run to the full 10 seeds. This is a deviation from the "all 17 arms x 10 seeds" plan, made with partial outcomes in view (disclosed here).
