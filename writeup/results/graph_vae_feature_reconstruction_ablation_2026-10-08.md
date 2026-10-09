# Feature-reconstruction ablation on Cora/TNBBeta: does adding a decoder pull `m` back toward unimodal? 2026-10-08

- **Date:** 2026-10-08
- **Branch/commit:** `dtd-oriented-textures`, `2cf9b6f` ("Thread
  --feature-reconstruction-weight through apps.link_prediction.main") --
  `master` did not yet have this code at the time of this run. The feature
  decoder itself was added at `4a9563f`.
- **Commands:**
  ```
  uv run python -m apps.link_prediction.main --dataset cora --family tnbbeta \
      --lrs 0.01 --dropouts 0 --latent-dims 16 --epochs 200 --seeds 0 \
      --feature-reconstruction-weight 1.0 \
      --run-name cora_tnbbeta_featurerecon_d16

  uv run python -m apps.eval.graph_posterior_shape \
      --run-name cora_tnbbeta_featurerecon_d16_seed0 --dataset cora
  ```

## Background

This project's research has found `TNBBetaSpherical`'s posterior goes
genuinely bimodal (`m = epsilon - (latent_dim - 1) / 2 < 0`) specifically on
real `GraphVAE` link-prediction runs (Cora, Citeseer, Pubmed, com-DBLP -- all
four, any scale, with or without real features), and specifically NOT on
reconstruction-style VAE tasks (MNIST, `axial_mixture`, DTD). Ablation "(b)"
(`writeup/results/pairwise_mlp_cluster_recovery_2026-10-08.md`) isolated
`GraphVAE`'s pairwise-dot-product-+-BCE loss on synthetic clusters through a
plain MLP, with no graph and no decoder, and found it does NOT reproduce
bimodality on its own -- ruling out "the loss function alone" as sufficient.

This is ablation "(a)", the complement: keep `GraphVAE`'s real GCN
architecture and the real Cora graph exactly as already used, and add back a
reconstruction term (decoding node features) on top of the existing
link-prediction loss. If that pulls `m` back toward/above 0, it argues
"presence of a reconstruction term" matters even holding the GCN and real
graph fixed. If it doesn't, that's further evidence the effect is something
else about `GraphVAE`/real graphs specifically.

## Factual check: are Cora's features actually binary?

Confirmed directly before building a decoder around it: loading
`load_planetoid(data_dir() / "planetoid", "cora").features` and inspecting
the dense array gives exactly two unique values, `{0.0, 1.0}` (2708 x 1433,
row sums ranging 1-30, mean ~18.2 nonzero words/node). Cora's features are
strictly binary bag-of-words, as expected, so a Bernoulli (binary
cross-entropy) decoder is the right likelihood -- no deviation from the
task's suggested default was needed.

## Picking the reconstruction weight

At `feature_reconstruction_weight = 1.0`, with `latent_dim = 16` and the
same other hyperparameters as the existing Cora run, five early training
steps gave:

| step | link_loss | kl | feature_loss |
|---|---|---|---|
| 0 | 0.938 | 3.268 | 0.698 |
| 1 | 1.004 | 3.195 | 0.693 |
| 2 | 0.960 | 3.214 | 0.688 |
| 3 | 0.923 | 2.855 | 0.683 |
| 4 | 0.899 | 2.650 | 0.678 |

`link_loss` and `feature_loss` are the same order of magnitude throughout
(both ~0.7-1.0); `kl / num_nodes` (the term actually added to `"loss"`) is
~0.001-0.0012, already negligible exactly as it is in the existing
(non-ablated) Cora run. Neither term swamps the other at `weight = 1.0`, so
this weight was used as-is for the real run below -- no reason to pick
something else that would make the comparison harder to read.

## Result: does it move `m`?

| | `epsilon_mean` | `m_mean` | `frac_bimodal` | `p_mean` | test AUC | test AP |
|---|---|---|---|---|---|---|
| Cora/TNBBeta, `latent_dim=16`, **no** feature reconstruction (recorded earlier this session) | 5.23 | -2.27 | 0.952 | 0.978 | - | - |
| Cora/TNBBeta, `latent_dim=16`, **with** feature reconstruction (`weight=1.0`, this run) | 5.293 | -2.207 | 0.948 | 0.990 | 0.9233 | 0.9350 |

(Val AUC/AP for this run: 0.9449 / 0.9331; the selected single configuration
matched the only one run, as expected with one `(lr, dropout, latent_dim)`
point and one seed.)

Full `posterior_stats` for the feature-reconstruction run:

```json
{
  "entropy_mean": NaN,
  "r_bar": 0.9792020916938782,
  "num_nodes": 2708,
  "p_mean": 0.9897773861885071,
  "p_std": 0.01502863597124815,
  "q_mean": 0.9690707921981812,
  "q_std": 0.028280967846512794,
  "epsilon_mean": 5.292879104614258,
  "epsilon_std": 1.4430692195892334,
  "m_mean": -2.2071211338043213,
  "frac_bimodal": 0.9483013153076172
}
```

## Reading it

**Adding a real reconstruction term, while holding the GCN and the real
Cora graph fixed, does not move `m` back toward/above 0.** `epsilon_mean`
(5.29 vs. 5.23), `m_mean` (-2.21 vs. -2.27) and `frac_bimodal` (0.948 vs.
0.952) are essentially unchanged -- within the kind of run-to-run noise a
single extra seed would plausibly produce on its own, not a directional
shift toward unimodal. `p_mean` even moved slightly further from 0.5 (0.990
vs. 0.978), the opposite of what "reconstruction pulls the posterior toward
the richer, less-collapsed shape reconstruction-style tasks show" would
predict. Link-prediction quality did not collapse from adding the second
loss term (test AUC 0.923, test AP 0.935 -- consistent with this codebase's
other Cora/TNBBeta results and with VGAE's typical Cora performance), so
this is a real, working combined objective, not a degenerate run where one
term dominated and the comparison is meaningless.

**Combined with ablation (b)'s result, this narrows things further.**
(b) showed the pairwise/BCE loss alone, on a plain MLP with no graph, does
not produce bimodality. (a) now shows that adding reconstruction back, on
the real GCN with the real graph, does not remove it either. Neither "the
loss function" nor "the absence of a reconstruction term" explains
bimodality by itself. What's left, and not yet ruled out by either
ablation: the GCN architecture's message-passing itself, or some property of
training on a real graph's scale/degree structure (a few thousand nodes,
power-law-ish degree distribution, multi-hop neighborhood reuse across
nodes) that a synthetic independent-item MLP encoder and a single decoder
term don't reproduce.

**Caveats:** one seed, one `(lr, dropout, latent_dim)` configuration,
`feature_reconstruction_weight = 1.0` only (not swept) -- the magnitude
check above suggests 1.0 is a reasonable, non-degenerate choice, but a
sweep over weight (e.g. 0.1, 1.0, 10.0) was out of scope for this ablation
and could in principle show a weight-dependent effect this single value
misses. The "without feature reconstruction" comparison row is also only a
single recorded run (quoted from this session, not independently re-run
here) rather than a seed-averaged baseline from this ablation's own code
path.
