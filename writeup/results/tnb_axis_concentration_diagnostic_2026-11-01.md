# TNBBeta axis-concentration diagnostic (2026-11-01)

Date: 2026-11-01. Base commit: `ea01fdedd2c12ec0d65c275d1c6927657f4231f0` (branch
`synthetic-graph-families`), uncommitted working tree on top of that commit.

## Question

Several TNBBeta graph-search configs end with mean `post_r_bar` = 0.10 and
`second_mode_mass` = 0.43 (bipartite-00, -02, -03, -11, degree_corrected-00, ring-01).
Both numbers are consistent with a near-uniform posterior and with an antipodal pair of
modes at +mu/-mu with 57/43 mass (`r_bar` cancels for antipodal mixtures). Which one is it?

## Decision rule (written before any run; not changed afterwards)

New gauge-free statistic `axis_abs_cos` = mean over nodes and samples of `|z . mu_hat|`,
using the same axis `mu_hat` (`mean_direction`/`loc`) and the same 32 samples as
`second_mode_mass`; `axis_abs_cos_node_spread` = std over nodes of each node's mean
`|z . mu_hat|`.

Near-uniform baseline at d = 16 (Monte Carlo, 4e6 uniform points on S^15):
E|z . mu| = 0.2026 (closed form sqrt(2/(pi d)) = 0.1995 is the large-d approximation).

- "near-uniform": mean `axis_abs_cos` within 0.05 of 0.2026 (i.e. in [0.153, 0.253]).
- "axis-concentrated (antipodal pair or cap)": mean `axis_abs_cos` > 0.5.
- "intermediate": otherwise.
- TNBBeta config is a **real antipodal pair** if axis-concentrated AND
  `second_mode_mass` > 0.1 (a cap has second-mode mass ~0).

Means are over the 4 draws of a config (min/max reported). Descriptive only.

## Commands

```
uv run python -m apps.synthetic.graph_search manifest --stage axis_diagnostic --configs discovery \
  --config-ids bipartite-00 bipartite-02 bipartite-03 bipartite-11 degree_corrected-00 ring-01 \
               bipartite-01 assortative-06 assortative-00 \
  --draws 4 --conditions tnbbeta vmf_init power_spherical_init vmf_k4 power_spherical_k4
STAGE=axis_diagnostic sbatch --array=0-5 scripts/slurm/graph_search.sbatch   # job 2288145
```

180 runs (9 configs x 4 draws x 5 conditions), seeds `40_000_000 + ...`, learning rates
reused from the pilot (fixed-kappa conditions use their family's `_init` rate). Raw run
files: `$TNBBETA_CHECKPOINT_DIR/graph_search/axis_diagnostic/runs/`. Table built by reading
those JSONs (mean over the 4 draws, `[min, max]`). Two TNBBeta runs were flagged
`diverged` (assortative-06 draw 3, bipartite-01 draw 0) and are included in the numbers.

Control config roles: bipartite-01 (earlier r_bar .45), assortative-06 (.29), assortative-00
(.63; TNB second-mode mass .128 vs 0 for matched vMF).

## Results (mean [min, max] over 4 draws, d = 16)

| config | condition | n | r_bar | second_mode_mass | axis_abs_cos | node spread |
|---|---|---|---|---|---|---|
| bipartite-00 | tnbbeta | 4 | 0.097 [0.096,0.097] | 0.430 [0.429,0.430] | 0.485 [0.478,0.491] | 0.052 [0.051,0.055] |
| bipartite-00 | vmf_init | 4 | 0.617 [0.613,0.627] | 0.000 [0.000,0.000] | 0.613 [0.609,0.622] | 0.026 [0.026,0.026] |
| bipartite-00 | power_spherical_init | 4 | 0.524 [0.523,0.526] | 0.001 [0.001,0.001] | 0.518 [0.516,0.520] | 0.027 [0.026,0.027] |
| bipartite-00 | vmf_k4 | 4 | 0.255 [0.255,0.256] | 0.157 [0.155,0.161] | 0.280 [0.278,0.281] | 0.032 [0.031,0.033] |
| bipartite-00 | power_spherical_k4 | 4 | 0.231 [0.231,0.232] | 0.172 [0.172,0.173] | 0.254 [0.252,0.255] | 0.030 [0.029,0.030] |
| bipartite-02 | tnbbeta | 4 | 0.098 [0.097,0.098] | 0.429 [0.426,0.431] | 0.472 [0.455,0.488] | 0.052 [0.052,0.054] |
| bipartite-02 | vmf_init | 4 | 0.615 [0.614,0.616] | 0.000 [0.000,0.001] | 0.611 [0.608,0.614] | 0.026 [0.025,0.026] |
| bipartite-02 | power_spherical_init | 4 | 0.525 [0.523,0.529] | 0.001 [0.001,0.002] | 0.517 [0.515,0.521] | 0.026 [0.026,0.027] |
| bipartite-02 | vmf_k4 | 4 | 0.256 [0.255,0.257] | 0.158 [0.155,0.162] | 0.280 [0.278,0.282] | 0.032 [0.031,0.032] |
| bipartite-02 | power_spherical_k4 | 4 | 0.231 [0.231,0.232] | 0.175 [0.173,0.177] | 0.253 [0.252,0.253] | 0.029 [0.028,0.031] |
| bipartite-03 | tnbbeta | 4 | 0.097 [0.096,0.098] | 0.432 [0.431,0.433] | 0.488 [0.482,0.490] | 0.054 [0.053,0.055] |
| bipartite-03 | vmf_init | 4 | 0.625 [0.614,0.637] | 0.000 [0.000,0.000] | 0.621 [0.609,0.631] | 0.026 [0.025,0.027] |
| bipartite-03 | power_spherical_init | 4 | 0.529 [0.524,0.536] | 0.001 [0.001,0.001] | 0.522 [0.517,0.530] | 0.026 [0.025,0.026] |
| bipartite-03 | vmf_k4 | 4 | 0.256 [0.255,0.257] | 0.159 [0.157,0.160] | 0.280 [0.278,0.281] | 0.031 [0.030,0.032] |
| bipartite-03 | power_spherical_k4 | 4 | 0.231 [0.230,0.233] | 0.173 [0.172,0.175] | 0.252 [0.251,0.253] | 0.029 [0.029,0.030] |
| bipartite-11 | tnbbeta | 4 | 0.098 [0.097,0.100] | 0.429 [0.427,0.432] | 0.469 [0.435,0.488] | 0.052 [0.048,0.056] |
| bipartite-11 | vmf_init | 4 | 0.622 [0.614,0.629] | 0.000 [0.000,0.000] | 0.618 [0.609,0.626] | 0.027 [0.026,0.027] |
| bipartite-11 | power_spherical_init | 4 | 0.523 [0.522,0.524] | 0.001 [0.001,0.002] | 0.516 [0.514,0.517] | 0.027 [0.026,0.027] |
| bipartite-11 | vmf_k4 | 4 | 0.256 [0.255,0.257] | 0.159 [0.156,0.162] | 0.280 [0.279,0.280] | 0.031 [0.030,0.033] |
| bipartite-11 | power_spherical_k4 | 4 | 0.231 [0.230,0.233] | 0.173 [0.172,0.174] | 0.253 [0.252,0.254] | 0.029 [0.029,0.030] |
| degree_corrected-00 | tnbbeta | 4 | 0.098 [0.097,0.098] | 0.431 [0.429,0.432] | 0.488 [0.482,0.490] | 0.053 [0.052,0.054] |
| degree_corrected-00 | vmf_init | 4 | 0.620 [0.616,0.624] | 0.000 [0.000,0.001] | 0.614 [0.611,0.619] | 0.027 [0.026,0.027] |
| degree_corrected-00 | power_spherical_init | 4 | 0.531 [0.523,0.538] | 0.001 [0.001,0.002] | 0.524 [0.516,0.532] | 0.026 [0.026,0.027] |
| degree_corrected-00 | vmf_k4 | 4 | 0.255 [0.254,0.255] | 0.158 [0.156,0.162] | 0.280 [0.278,0.282] | 0.031 [0.030,0.033] |
| degree_corrected-00 | power_spherical_k4 | 4 | 0.231 [0.230,0.232] | 0.171 [0.169,0.173] | 0.253 [0.252,0.254] | 0.029 [0.028,0.030] |
| ring-01 | tnbbeta | 4 | 0.097 [0.096,0.098] | 0.430 [0.427,0.432] | 0.473 [0.452,0.487] | 0.053 [0.051,0.054] |
| ring-01 | vmf_init | 4 | 0.623 [0.614,0.640] | 0.000 [0.000,0.000] | 0.618 [0.608,0.636] | 0.026 [0.025,0.026] |
| ring-01 | power_spherical_init | 4 | 0.530 [0.524,0.540] | 0.001 [0.001,0.001] | 0.523 [0.517,0.533] | 0.027 [0.026,0.027] |
| ring-01 | vmf_k4 | 4 | 0.256 [0.255,0.256] | 0.158 [0.156,0.159] | 0.280 [0.279,0.280] | 0.031 [0.031,0.032] |
| ring-01 | power_spherical_k4 | 4 | 0.231 [0.231,0.232] | 0.173 [0.172,0.175] | 0.253 [0.253,0.254] | 0.030 [0.030,0.031] |
| bipartite-01 | tnbbeta | 4 | 0.600 [0.565,0.651] | 0.010 [0.006,0.015] | 0.594 [0.558,0.647] | 0.192 [0.167,0.209] |
| bipartite-01 | vmf_init | 4 | 0.743 [0.612,0.807] | 0.000 [0.000,0.000] | 0.739 [0.606,0.804] | 0.029 [0.027,0.032] |
| bipartite-01 | power_spherical_init | 4 | 0.700 [0.617,0.764] | 0.000 [0.000,0.000] | 0.696 [0.611,0.762] | 0.035 [0.032,0.038] |
| bipartite-01 | vmf_k4 | 4 | 0.256 [0.255,0.257] | 0.158 [0.157,0.160] | 0.281 [0.279,0.282] | 0.031 [0.031,0.032] |
| bipartite-01 | power_spherical_k4 | 4 | 0.231 [0.230,0.231] | 0.172 [0.168,0.173] | 0.252 [0.252,0.254] | 0.029 [0.028,0.031] |
| assortative-06 | tnbbeta | 4 | 0.263 [0.102,0.727] | 0.313 [0.000,0.420] | 0.445 [0.291,0.724] | 0.061 [0.051,0.084] |
| assortative-06 | vmf_init | 4 | 0.620 [0.589,0.668] | 0.000 [0.000,0.001] | 0.615 [0.583,0.663] | 0.027 [0.026,0.029] |
| assortative-06 | power_spherical_init | 4 | 0.524 [0.498,0.567] | 0.002 [0.000,0.002] | 0.517 [0.491,0.560] | 0.028 [0.026,0.029] |
| assortative-06 | vmf_k4 | 4 | 0.256 [0.255,0.257] | 0.157 [0.154,0.162] | 0.281 [0.281,0.281] | 0.031 [0.030,0.032] |
| assortative-06 | power_spherical_k4 | 4 | 0.232 [0.232,0.232] | 0.174 [0.173,0.177] | 0.253 [0.252,0.255] | 0.030 [0.030,0.031] |
| assortative-00 | tnbbeta | 4 | 0.776 [0.543,0.909] | 0.021 [0.000,0.072] | 0.783 [0.572,0.909] | 0.096 [0.027,0.174] |
| assortative-00 | vmf_init | 4 | 0.615 [0.614,0.618] | 0.000 [0.000,0.000] | 0.611 [0.608,0.614] | 0.026 [0.025,0.027] |
| assortative-00 | power_spherical_init | 4 | 0.531 [0.522,0.552] | 0.001 [0.001,0.001] | 0.526 [0.515,0.547] | 0.027 [0.026,0.028] |
| assortative-00 | vmf_k4 | 4 | 0.254 [0.253,0.256] | 0.158 [0.156,0.159] | 0.282 [0.280,0.284] | 0.032 [0.031,0.033] |
| assortative-00 | power_spherical_k4 | 4 | 0.230 [0.229,0.231] | 0.171 [0.162,0.180] | 0.254 [0.252,0.255] | 0.031 [0.030,0.032] |

### Classification of TNBBeta by the pre-stated rule

| config | mean axis_abs_cos | mean second_mode_mass | class |
|---|---|---|---|
| bipartite-00 | 0.485 | 0.430 | intermediate |
| bipartite-02 | 0.472 | 0.429 | intermediate |
| bipartite-03 | 0.488 | 0.432 | intermediate |
| bipartite-11 | 0.469 | 0.429 | intermediate |
| degree_corrected-00 | 0.488 | 0.431 | intermediate |
| ring-01 | 0.473 | 0.430 | intermediate |
| bipartite-01 (control) | 0.594 | 0.010 | axis-concentrated, cap (second mass < 0.1) |
| assortative-06 (control) | 0.445 | 0.313 | intermediate (draws very heterogeneous: r_bar .10 to .73) |
| assortative-00 (control) | 0.783 | 0.021 | axis-concentrated, cap |

By the rule as written, none of the nine configs is a "real antipodal pair": the six
r_bar = 0.10 configs are all "intermediate" (0.47 to 0.49, just under the 0.5 cut-off), not
near-uniform (0.20 +/- 0.05) and not axis-concentrated.

Per-node spread: for the six group configs the std over nodes of the per-node mean
|z . mu_hat| is 0.05 (vMF/PS 0.03; baseline sampling noise with 32 samples is about
0.03), against a mean of 0.48. So the nodes are nearly homogeneous: there is no sign of a
mix of some concentrated and some uniform nodes. (The controls bipartite-01 and
assortative-00 do show a mix: spread 0.19 and 0.10.)

## Conclusion

1. The posteriors in the r_bar = 0.10 group are **not near-uniform**: |z . mu_hat| averages
   about 0.48, against 0.20 for uniform. Reading these as "near-uniform" is ruled out.
2. They are **not tight antipodal pairs** either, and the pre-stated rule does not label
   them real pairs (class: intermediate, 0.47-0.49 vs the 0.5 threshold, which I chose
   before seeing the data; the near-miss is a matter of that threshold, not a finding).
3. They are also hard to read as a single cap: a unimodal cap with 43% of its samples on
   the far side of the mean axis would have to be close to uniform (|cos| near 0.2), as
   the diffuse fixed-kappa=4 vMF/PS controls show (second mass 0.16-0.17 at |cos| 0.25-0.28),
   whereas a cap with |cos| ~0.5 (vMF/PS init, 0.52-0.62) has second-mode mass about 0.
   TNBBeta combines |cos| ~ 0.48 with second mass 0.43 and r_bar 0.10, which is what a
   fairly broad two-sided (axis-concentrated, roughly +/-mu) distribution produces. This
   is an inference from three summary statistics, not a direct observation of two modes.
4. Caveat on the rule: `second_mode_mass` > 0.1 does not by itself separate a pair from a
   diffuse cap (the kappa=4 controls exceed 0.1 with no second mode), so the rule's
   second clause is weak when `axis_abs_cos` is low; it only helps when |cos| is large.
5. The same fixed-seed numbers recur across the six configs to two decimals (r_bar .097-.098,
   second mass .429-.432), suggesting TNBBeta lands on the same posterior shape whatever
   the graph, i.e. possibly an optimisation attractor and not something fitted to each
   graph. This is observation, not tested.

Not shown: that the posterior has two separate modes (no density or per-node sample
histogram was inspected), that the two sides are used by the decoder (that +z vs -z
matters for edge probabilities), or that this helps predictive metrics. Only 4 draws per
config, descriptive, not pre-registered, no hypothesis tests.

## Code changes

- `apps/synthetic/graph_families.py::second_mode_mass` also returns `axis_abs_cos` and
  `axis_abs_cos_node_spread` (nan for Gaussian); `graph_search.evaluate` passes them
  through via its existing `**second_mode_mass(...)` spread, so they appear as metric keys.
- `apps/synthetic/graph_search.py`: `SEED_BASE["axis_diagnostic"] = 40_000_000`.
- Tests: `tests/apps/test_graph_families.py` (uniform and antipodal stubs),
  `tests/apps/test_graph_search.py` (stage manifest).
