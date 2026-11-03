# vMF sampler bias audit: pre-registration (2026-11-03)

Frozen before any result of this audit was read. Base commit `dd8ede0` (branch `synthetic-graph-families`), uncommitted working tree. Results: `vmf_sampler_bias_audit_2026-11-03.md`.

## Question
The library `VonMisesFisher` rejection sampler (port of the S-VAE reference) switches Wood's envelope parameter `b` to the Taylor value `(m-1)/(4 kappa)` over `kappa in (10, 11)` while `a`, `d` follow from the blended `b`. The ring dimension-scaling study saw biased draws at `m = 17, kappa = 17` (E[cos] 0.608 vs exact 0.627). Where exactly is it biased, and does it change the TNBBeta-vs-vMF conclusions of the metric-first graph search?

## (1) Grid test
- `m in {3,4,5,6,8,9,10,12,16,17,20,25,33,50,65}`, `kappa in {1,2,4,8,10,11,12,14,16,20,25,30,40,50,60,80,100,150,200,300,500}`; N = 400,000 draws per cell via the full `rsample` path (random mean direction); float32 and float64; library default vs `exact_envelope=True`.
- Exact references: `E[w] = I_{m/2}(k)/I_{m/2-1}(k)`, `E[w^2] = 1 - (m-1) E[w]/k`, exact marginal CDF by quadrature (KS).
- A cell is **flagged biased** if `|mean - exact| > 4 SE`, or `|second moment - exact| > 4 SE`, or `KS sqrt(N) > 1.95` (about a 0.1% test). With SE <= 0.0009 the smallest detectable mean bias is about 0.004. The expected sizes: `exact_envelope=True` has zero flagged cells up to multiplicity (315 cells x 2 moment tests at 4 SE: about 0.04 false flags expected per variant); any flagged cell for it is reported as a failure of the fix.
- Cause to be checked: the cell set flagged for the default should be confined to `kappa > 10` (where the Taylor blend `s > 0`) and `m` above a threshold; for `kappa >= 11` the bias at fixed `m` should decay as `kappa` grows (Taylor error shrinks), and the fix should remove it entirely.

## (2) Audit
Grep `writeup/results` and `apps` for every vMF use; record `m` (ambient) and the kappa regime (fixed or learned and typical value from the write-up) and mark each as inside / outside / unknown relative to the flagged region. Learned-kappa runs whose kappa is not in the write-up are marked "unknown".

## (3) Fix
Add `exact_envelope: bool = False` to `VonMisesFisher` (additive; default behaviour unchanged, so every earlier result reproduces). Tests in `tests/distributions/test_von_mises_fisher.py`. Whether to flip the default is a decision for the owner and is not made here.

## (4) Graph-search rerun
Subset: configs `-00, -05, -10, -15` of each of the 10 families of the `fresh` stage (40 configs), all 16 draws, conditions `vmf_init` and `vmf_pin`, same seed / learning rate / graph / edge split as the original (`DIM = 16`, so `m = 16`, `kappa_init = kappa_pin = 16`: inside the suspect region). Each is run with the library sampler (`lib`, control: shows what a rerun alone changes, e.g. device non-determinism) and with `exact_envelope=True` (`exact`). TNBBeta and PowerSpherical are not rerun (their `fresh` results are the comparison; some of those ran on GPU, the reruns run on CPU).
- Unit of analysis: config (mean over the 16 draws per config, 40 units).
- Metrics (advantage = comparator - TNBBeta for lower-is-better, TNBBeta - comparator otherwise): `truth_kl`, `sampled_log_loss`, `truth_kl_w`, `ece`, `cal_slope_dev`, `centre_auc`, `sampled_auc`, `rank_corr`.
- Test A (confirmatory): one-sided paired Wilcoxon signed-rank (TNBBeta better), over 2 comparators x 2 samplers x 8 metrics = 32 rows, Holm at 0.05. Report median / mean advantage, dz, config better/worse counts.
- Test B (secondary): two-sided paired Wilcoxon, vMF(lib) vs vMF(exact), 16 rows, Holm at 0.05, same effect sizes.
- Headline statement: TNBBeta's wins "change" if the set of Holm-confirmed TNBBeta-better rows differs between `lib` and `exact`. Nulls are reported as nulls; with 40 configs power for effects with dz < about 0.45 is low (80% power at dz 0.45, alpha 0.05 one-sided), which is stated, not hidden.
- Command: `uv run python -m apps.synthetic.vmf_exact_rerun manifest; TOTAL_TASKS=12 sbatch --array=0-11 scripts/slurm/vmf_exact_rerun.sbatch; AUX=analyze sbatch scripts/slurm/vmf_bias_aux.sbatch`.
