# Exposure of the finished graph experiments to the small-epsilon TNBBeta defect

Date: 2026-11-01. Code: base commit `ec2677e` (branch `synthetic-graph-families`), read-only (nothing in `src/` or
`distributions/` edited; no training). Question from `tnb_negative_kl_clamp_and_beta_underflow_2026-11-01.md`: `rsample` collapses to
`y = p` below eps ~ 3e-3 and `log_prob` clamps y, so the MC KL/ELBO is biased below eps ~ 0.3. Were the graph runs trained or measured in
that regime? Speculative hypothesis tested: "`m < 0` on every graph but `second_mode_mass = 0` (one cap)" is a sampler artefact of tiny eps.

## Verdict

**No.** Every TNBBeta graph posterior I could inspect has epsilon at or above about 0.5 (graph_search run means >= 0.52; per-node minimum
over all loaded checkpoints 0.68). The sampler defect (eps < 3e-3) is 100x or more away and the clamp bias (eps < 0.3) is not reached.
This clears the graph findings of this defect (except for what I could not check, below). `m < 0` at d = 16 is large-d geometry (the
threshold is (d-1)/2 = 7.5), not small eps. The hypothesis is also refuted on its own terms: `second_mode_mass` is *not* 0 in graph_search
(see below), and its sampler is in a regime where it is exact.

## Commands

Scratch scripts (session scratchpad `/tmp/claude-14853/-home-vpathak-tnbbeta-vae/39a83aae-c9e5-4e62-8b02-10e6fc87ddf7/scratchpad/`):

    # run-level logged stats (post_epsilon_mean/std, post_m_mean, post_frac_bimodal, second_mode_mass) for every *_tnbbeta* run JSON
    python3 scratchpad/scan.py {pilot,discovery,confirmation,axis_diagnostic,kappa_grid}   # reads $TNBBETA_CHECKPOINT_DIR/graph_search/<stage>/runs
    uv run python scratchpad/an.py pilot confirmation axis_diagnostic ; uv run python scratchpad/an2.py
    # per-node epsilon from final.pt (CPU forward of GraphVAE.posterior_and_prior on the whole graph, like apps.eval.graph_posterior_shape)
    PYTHONPATH=. uv run python scratchpad/node_eps.py out.json planetoid_tnbbeta_{cora,citeseer,pubmed}_d16_seed{0,1,2}:<ds> \
        fixedmg_{cora,citeseer,pubmed,mag_cs}_free_seed{0,1}:<ds> mag_cs_tnbbeta_seed{0,1,2}:mag_cs:0

(Whole-graph forward, not the training adjacency: fine for an epsilon distribution; see the CLAUDE.md warning about per-node statistics tied
to a trained checkpoint.) No SLURM was needed; the loads ran on CPU in under two minutes.

## Results

### graph_search (all stages; run-level logged mean/std of per-node epsilon; per-node values and checkpoints are NOT saved there)

| stage | TNBBeta runs | eps_mean min / 1st pct / median | runs with eps_mean < 0.6 | `m_mean < 0` | `second_mode_mass` == 0 / > 0.01 |
|---|---|---|---|---|---|
| pilot | 240 | 0.678 / 0.69 / 3.26 | 0 | 0.91 | 0.33 / 0.34 |
| discovery | 3840 | 0.571 / 0.69 / 3.22 | 3 | 0.96 | 0.28 / 0.39 |
| confirmation | 1088 | 0.523 / 0.69 / 3.48 | 1 | 0.96 | 0.32 / 0.37 |
| axis_diagnostic | 36 | 0.693 / 0.69 / 0.73 | 0 | 0.97 | 0.08 / 0.86 |
| kappa_grid | 960 | 0.585 / 0.69 / 3.44 | 1 | 0.96 | 0.33 / 0.38 |

Only 4 of 6164 runs have eps_mean < 0.6; none < 0.5 (min 0.523, degree_corrected-03, with std 0.04, so no node near 0.3). By family
(confirmation) the median eps_mean is 0.73 (bipartite), 1.28 (mixed_membership), 2.3-4.3 (the rest); bipartite and the ring/DC axis runs
sit at 0.69-0.73 with std ~0.00, i.e. at the head's initial value `softplus(0) = 0.693` (931 of 6164 runs are within 0.68-0.72 with std <
0.02: the plateau of `tnb_plateau_mechanism_2026-11-01.md`, not the 1e-4 floor). Within-run std is up to 9.4, so for runs with large std the
logged mean alone cannot rule out a thin lower tail; the checkpoints below show what the per-node tail actually looks like (right-skewed,
floor about 0.7), and no checkpoint-level node falls below 0.68. A Cantelli bound from the logged moments alone is loose (up to 0.35
for the worst run) and is not evidence either way. I could not load graph_search checkpoints (none saved), so per-node eps < 0.3 for
graph_search is **inferred, not measured**.

### Per-node epsilon from saved checkpoints (all loadable ones tested; `final.pt`, whole-graph forward)

| run family | runs | nodes | min eps | 1st pct | median | frac < 0.3 / < 0.5 |
|---|---|---|---|---|---|---|
| planetoid_tnbbeta Cora d16 (features) | 3 seeds | 2708 | 1.28-1.38 | 2.3-2.4 | 4.5-5.3 | 0 / 0 |
| planetoid_tnbbeta Citeseer d16 | 3 | 3327 | 2.02-2.35 | 3.0-3.3 | 5.2-5.6 | 0 / 0 |
| planetoid_tnbbeta Pubmed d16 | 3 | 19717 | 1.88-2.02 | 2.3-2.5 | 3.0-3.4 | 0 / 0 |
| fixedmg_*_free Cora (featureless) | 2 | 2708 | 0.93-0.95 | 1.13-1.22 | 2.9 | 0 / 0 |
| fixedmg_*_free Citeseer | 2 | 3327 | 0.79-0.80 | 0.90-0.95 | 2.6 | 0 / 0 |
| fixedmg_*_free Pubmed | 2 | 19717 | 0.90-1.18 | 1.57-1.58 | 2.8 | 0 / 0 |
| fixedmg_*_free MAG-CS | 2 | 21957 | 0.91-0.93 | 1.55 | 3.0-3.1 | 0 / 0 |
| mag_cs_tnbbeta (features) | 3 | 21957 | 0.68-0.77 | 0.71-0.88 | 0.80 / 2.07 / 2.21 | 0 / 0 |

(`mag_cs_tnbbeta_seed0` is at its initial plateau: median 0.80, q99 1.18, matching its logged `posterior_shape_final.json` eps_mean 0.82.)
Also logged (not re-loaded): synthetic_graphs `gfp_*_tnbbeta_free` (8 presets x 5 seeds; eps_mean 1.3-4.1, std 0.1-1.1, three
mixed_membership seeds at 0.70 plateau) and DBLP d16 (`bridge_diagnostic_final.json`: bridge eps_mean 1.3/1.2, std 0.4/0.3, over 317k nodes).

### Fixed-m / fixed-epsilon sweeps

By construction eps = threshold * (1 + f) with f >= -0.9 gives eps >= 0.1 * threshold = 0.75-0.8 for the d = 17/16 datasets and 3.2 at
the 33/65-dimensional ones. The one raw-m grid point that reaches the boundary is `link_prediction/*_m-7.2_tnbbeta` (d = 16:
eps = 7.5 - 7.2 = 0.3, where the clamped density integrates to 0.99 and the d = 3 bias is -0.04 nats; effect on a link-prediction AUC
comparison is not expected to be material but I did not re-evaluate it). `mnistfix*` runs (d = 5, eps 0.5-1.5, m = -2 gives eps = 0,
rejected by `epsilon_from_m`) are out of scope.

## Which findings could be affected, and how

* `second_mode_mass = 0` and "m < 0 but one cap": unaffected by the sampler collapse (needs eps < 3e-3; observed >= 0.52). Note that in
  graph_search `second_mode_mass` is not uniformly 0: 28-33% of runs are exactly 0, 37-39% exceed 0.01, up to 0.43 (the antipodal-pair
  maximum is 0.5), with the largest values on the plateau runs (bipartite / DC / ring at eps ~0.69). I did not locate the screening
  file behind the "= 0" statement in CLAUDE.md (the `gfp_*` JSONs in `synthetic_graphs/` carry no `second_mode_mass`), so I cannot say
  whether it was measured on a different set of runs; the CLAUDE.md wording ("a single cap") is true for some runs and not for graph_search
  as a whole, which is a separate discrepancy worth checking, unrelated to this defect.
* `frac_bimodal` / `m_mean < 0` (0.91-0.97 of runs): arithmetic on eps at d = 16; unaffected.
* The 46 confirmed TNBBeta-beats-vMF/PS probability-metric pairs: the training KL uses the library MC with eps >= 0.5 where the clamp bias
  is below about 0.04 nats per sample (d = 3 numbers; at d = 16 the Jacobian term is also clamped, but the boundary mass there is
  ~ (1e-6)^eps, about 1e-4 at eps = 0.7, so I estimate an effect of order 1e-3 nats; not computed with the honest evaluator, which is d = 3
  only). There is no sign of the exploit regime (eps driven to 1e-4) in any run. Direction if any bias existed: optimistic KL, i.e. it
  would favour TNBBeta slightly; magnitude is far below the effect sizes in that note, but not formally bounded. Not re-run.
* I did not use `apps/synthetic/tnb_low_epsilon_check.py` on graph runs because no run has a sizeable fraction (indeed any measured node)
  with eps < 0.3; it is d = 3 only, so it could not evaluate d = 16 posteriors anyway.

## What I could not check

* Per-node eps for graph_search, graph_families `gfp_*`, SBM and DBLP runs (no checkpoints loaded: graph_search saves none; I did not load the
  317k-node DBLP checkpoints or the ~25 `fixedm_*` SBM checkpoints). Inferred from logged mean/std only.
* The `lk`/`lk2` link-prediction JSONs, `dose_response_*`, `kdyn_*`, `matched_tnbbeta_*`, `fixedmg_*_{m<0}` TNBBeta checkpoints other than the `free`
  ones above (fixed-epsilon ones pin eps by construction, see above).
* Conv-VAE/MNIST/semi-supervised/MLP TNBBeta runs (not graph experiments; the stage-3 ring note already found one collapsed seed).
* A d = 16 honest KL (no such evaluator exists; the d = 3 evaluator would need the Jacobian term).
