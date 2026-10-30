# Why does TNBBeta go bimodal on GraphVAE but not on reconstruction tasks? Summary, 2026-10-08

- **Date:** 2026-10-08
- **Branch/commit:** `dtd-oriented-textures`, `98c6ff4` at time of writing.
  This summary ties together results spanning that branch and `master`
  (via PR #59); each underlying result is cited with its own commit below
  so none of this depends on this document alone for reproducibility.

This is a synthesis, not a new experiment: it connects five separate
dated results (three on `master`, three on this branch) into one
narrative, because the individual write-ups were produced in sequence as
hypotheses were ruled in or out, and the overall arc isn't visible from any
one of them alone.

## The question

`TNBBetaSpherical` can represent antipodal bimodality (`m = epsilon -
(latent_dim-1)/2 < 0`, proven necessary and sufficient by
`writeup/weekly_markdown_summaries/week_2/tnbbeta_vs_power_spherical_expressivity.md`'s
Theorem 5.1/Corollary 5.3) -- a shape vMF and Power Spherical are
structurally incapable of. The open question this investigation chases:
when does training actually find and use that capacity, and why?

## 1. The starting observation (pre-existing, `master`)

`writeup/results/dblp_bridge_diagnostic_2026-10-01.md` (master commit
`7a41567`) found TNBBeta beats vMF/Power Spherical significantly on
com-DBLP link prediction, with the margin growing monotonically with a
node's community count. Its corrected shape diagnostic found **every**
com-DBLP node, bridge or not, sits in the bimodal regime (`epsilon approx
1.1-1.2` against a threshold of `7.5` at `d=16`) -- ruling out the original
hypothesis that bimodality is reserved for genuinely multi-community
("bridge") nodes. This is the opposite regime from MNIST training
(`posterior_trajectories_2026-10-01.md`), which never approaches `m<0`.

## 2. First hypothesis tried and ruled out: manufactured sign-ambiguity (`master`, PR #59)

If bimodality isn't about bridge nodes specifically, maybe it's about
*any* genuine per-example ambiguity a decoder can't resolve. `axial_mixture`
(`src/tnbbeta_vae/data/axial_mixture.py`, commit `a665b80`) built a
synthetic task where the true angle is drawn from a 50/50 antipodal
mixture, observed through an embedding exactly invariant under `phi ->
phi+pi` -- so the Bayes-optimal posterior is *provably* bimodal.

- Free training never finds it (`writeup/results/axial_mixture_recovery_2026-10-07.md`,
  commit `08b8f6a`): `m_mean = +26.5` at `latent_dim=2`.
- Forcing it directly via `fixed_epsilon` at `latent_dim=5` (`m=-0.5`
  exactly, commit `66921be`) still doesn't help: net ELBO is *worse* than
  the free-epsilon unimodal collapse, and `p` stays pinned at an extreme
  rather than the `0.5` a balanced bimodal posterior needs. The task's
  decoder is exactly sign-invariant, so there's no reconstruction reward
  for splitting mass across both modes -- only extra KL cost.
- A real-data analogue (DTD oriented textures,
  `src/tnbbeta_vae/data/dtd.py`, `dtd-oriented-textures` branch) found the
  same pattern: `m_mean = +67.9` training freely; a real, significant
  (`~5.8` SE) but far-too-small-to-flip-sign correlation between texture
  orientation strength and `m`.

**Conclusion: manufactured sign-ambiguity is the wrong mechanism.** It
doesn't reproduce com-DBLP's result even on a task built specifically to
need it.

## 3. Is it DBLP-specific, or does it generalize across `GraphVAE` runs? (`dtd-oriented-textures`)

`apps/eval/graph_posterior_shape.py` (commit `7eee92d`) generalized the
shape diagnostic (factored out of `dblp_bridge_diagnostic.py` into
`tnbbeta_vae.models.posterior_stats`, commit `3582cc4`) to any `GraphVAE`
checkpoint. Run against the existing Planetoid checkpoints (Cora,
Citeseer, Pubmed -- 100x smaller than com-DBLP, *with* real content
features):

| dataset | nodes | features | `m_mean` | `frac_bimodal` |
|---|---|---|---|---|
| Cora | 2,708 | real | -2.27 | 0.952 |
| Citeseer | 3,327 | real | -2.08 | 0.940 |
| Pubmed | 19,717 | real | -4.00 | 0.986 |
| com-DBLP | 317,080 | none | ~-6.3 | 1.000 |

**Scale and feature-presence are both ruled out as necessary conditions.**
Every real `GraphVAE` link-prediction run checked goes bimodal, regardless.

## 4. Three controlled ablations isolate the real mechanism (`dtd-oriented-textures`)

Every real example so far combines a GCN encoder, a real graph, no
decoder, and a pairwise dot-product+BCE loss. Three ablations varied one
factor at a time:

- **(b)** `pairwise_mlp_cluster_recovery_2026-10-08.md` (commits
  `74be4cd`..`6310424`): the exact loss, through a plain MLP, **no graph at
  all**, on synthetic Gaussian-blob clusters (10 clusters x 50 points).
  Stays unimodal (`m_mean = +8.655`, `frac_bimodal = 0.0`); `epsilon`
  drifts *further* unimodal with more training in all 3 seeds checked. The
  loss alone, without a graph, is not sufficient.
- **(a)** `graph_vae_feature_reconstruction_ablation_2026-10-08.md`
  (commits `4a9563f`..`4946ebf`): the real GCN, the real Cora graph, with a
  **reconstruction decoder added back** on top of the link loss.
  `m_mean = -2.21` (vs. `-2.27` without it) -- essentially unchanged.
  Absence of reconstruction is not what's doing the work.
- **(c)** `sbm_gcn_recovery_ablation_2026-10-08.md` (commits
  `93870ea`..`98c6ff4`): the real GCN on a **synthetic**, featureless,
  500-node stochastic-block-model graph (community-correlated edges, no
  real dataset involved at all). `m_mean = -4.55`, `frac_bimodal = 1.00` in
  all 3 seeds, pinned there across 2000 epochs -- *more* bimodal than real
  Cora, on a graph 634x smaller than com-DBLP, with zero real-world
  structure.

**Triangulated conclusion: a GCN encoder aggregating over a graph with
community-correlated structure is sufficient for bimodality, independent
of scale, real features, and decoder presence.** Neither the loss function
alone nor the absence of reconstruction explains it; the GCN+graph
combination does.

## 5. The live mechanistic theory (not yet tested directly)

Sphere-family embeddings are forced to unit norm, so (unlike the
unconstrained Gaussian baseline, where `binary_cross_entropy_with_logits`
on an unbounded inner product lets high-degree nodes grow `||z||` as a
free per-node confidence channel -- a known effect in the graph-autoencoder
literature) a single sampled `z_i . z_j` is capped in `[-1, 1]`, and the
only global sharpening lever (`GraphVAE.temperature()`) is shared across
every node, not per-node. The candidate per-node substitute: **posterior
shape**. A GCN's neighbor-averaging pulls a node's embedding toward some
aggregate of its neighborhood; if that neighborhood is angularly
heterogeneous, no single direction scores well against every neighbor.
Where a unimodal cap is stuck with one mediocre compromise every training
step, a bimodal posterior can align sharply with different neighbor
subsets on different stochastic draws -- never bound by the `[-1,1]`
per-sample cap, but no longer forced into one fixed compromise either.
Ablation (c)'s result is consistent with this (aggregation over
community-correlated structure is exactly where the compromise-position
tension would arise), but it has not been tested directly: (c) does not
yet distinguish "GCN aggregation specifically" from "a graph-correlated
training signal exists in some looser sense."

## 6. Open, and the next test

The next ablation (in progress) holds (c)'s exact graph, exact edges, and
exact loss fixed, and varies *only* whether the encoder aggregates over
neighbors at all (identity vs. the real normalized adjacency) -- the
sharpest remaining test of "GCN aggregation itself" as the mechanism. See
the next dated entry in `writeup/results/` once that completes.
