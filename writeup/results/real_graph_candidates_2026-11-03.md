# Real-graph candidates matching (and contrasting) the TNBBeta "winning structure": 2026-11-03

- **Date:** 2026-11-03.
- **Type:** desk research (web + small downloads). No training, no code changes, no model results.
- **Code version:** `synthetic-graph-families` @ `dd8ede0` (working tree has unrelated uncommitted edits).
- **Target profile** (from `graph_search_metric_first_results_2026-11-02.md`, descriptive and within-family): config-level TNBBeta wins on probability-quality metrics co-occur with median realised mean degree 19.3 (vs 8.9 for losing configs), degree CV of expected degree 0.68 (vs 0.21) and clustering 0.084 (vs 0.038); bipartite is the only family with a confirmed family-level truth-KL advantage. **Matching degree/CV/clustering is necessary-looking but not shown sufficient**: these covariates co-occur with wins, no covariate rule transferred across families (leave-family-out R^2 = 0.000), and the within-family H1 is about log nominal degree inside a family. A real graph matching the profile is a test of the pattern, not a guaranteed win. Also, the synthetic CV is of *expected* degree under the generator, while the CVs below are of *realised* degree (std/mean over all nodes), which are not the same quantity; real heavy-tailed graphs give much larger CVs (1-2+) than the synthetic range.

## How numbers were obtained

"Source" = quoted from the dataset's own page (fetched 2026-11-03). "Computed" = I downloaded the file into a fresh scratchpad directory and computed with a short script (`python -I`, undirected, deduplicated, self-loops dropped; transitivity = 3 x triangles / wedges, *not* SNAP's average local clustering). Scripts and downloads were in the session scratchpad only, not committed. KONECT (konect.cc, konect.uni-koblenz.de) was unreachable from this machine, so nothing from KONECT is verified. PyG loader names are from the PyG docs page, which lists the names but gives few statistics.

## Candidates (winning-profile side)

| # | Dataset | Bipartite | Nodes | Edges | Mean deg | Degree CV | Clustering | Verified how |
|---|---|---|---|---|---|---|---|---|
| 1 | Gowalla user-location (LightGCN split) | yes | 70,839 (29,858 users + 40,981 locations) | 1,027,370 | 29.0 (users 34.4, items 25.1) | 1.58 overall (users 1.34, items 1.80) | 0 by construction (no odd cycles) | Computed from train+test files |
| 2 | ca-AstroPh (SNAP) | no | 18,772 (18,771 non-isolated) | 198,110 (SNAP count; I get 198,050 undirected after dedup/self-loop removal) | 21.1 | 1.45 | avg local 0.631 (SNAP); transitivity 0.318 (computed) | Source + computed |
| 3 | ca-HepPh (SNAP) | no | 12,008 (12,006 non-isolated) | 118,521 (SNAP; 118,489 after dedup) | 19.7 | 2.36 | avg local 0.612 (SNAP); transitivity 0.659 (computed) | Source + computed |
| 4 | MovieLens-100K | yes | 2,625 (943 users + 1,682 movies) | 100,000 | 76.2 (users 106, movies 59) | 1.20 (users 0.95, movies 1.35) | 0 | Source + computed |
| 5 | Yelp user-business (Network Repository) | yes | 57,520 (45,982 users + 11,538 businesses) | 229,906 | 8.0 (users 5.0, businesses 19.9) | 3.02 | 0 | Computed (the page's 50.4K nodes / 229.8K edges treat the two ID spaces as one) |
| 6 | Yelp2018 (LightGCN split) | yes | 69,716 | 1,561,406 | 44.8 | 1.30 | 0 | Computed |
| 7 | act-mooc (SNAP, Kumar et al. 2019) | yes (user -> course activity) | 7,144 (7,047 users + 97 targets) | 411,749 actions; 178,443 unique user-target pairs | 50.0 on unique pairs | 5.57 (targets average 1,840 neighbours) | 0 | Source + computed |

### Sources, licences, loaders

1. **Gowalla (LightGCN split).** Underlying data: Cho, Myers, Leskovec, "Friendship and Mobility", KDD 2011 (SNAP `loc-gowalla`); the user-item split used here is from He et al., "LightGCN", SIGIR 2020. Files: `https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/gowalla/{train,test}.txt` (each line: user id then item ids; train 4.6 MB, test 1.4 MB). The LightGCN repo states no licence for the data or code (checked); SNAP states none for the original either. Loader: none in this codebase; PyG has no Gowalla loader I could confirm. The 70k nodes are above "tens of thousands": use a user subsample or k-core filter, and report the induced degree stats afterwards (the subsampled graph will not keep these numbers).
2. **ca-AstroPh / ca-HepPh.** Leskovec, Kleinberg, Faloutsos, "Graph Evolution: Densification and Shrinking Diameters", ACM TKDD 1(1), 2007. `https://snap.stanford.edu/data/ca-AstroPh.txt.gz` (1.4 MB), `ca-HepPh.txt.gz` (0.8 MB). SNAP page states no licence. Stats on page: AstroPh 18,772 / 198,110 / clustering 0.6306 / 1,351,441 triangles; HepPh 12,008 / 118,521 / 0.6115 / 3,358,499 triangles. Both files list each edge in both directions, so the SNAP edge count is a directed-pair count and I recover a smaller undirected count; do not feed it to a loader that sums duplicates.
3. **MovieLens-100K.** Harper and Konstan, "The MovieLens Datasets", ACM TiiS 2015. `https://files.grouplens.org/datasets/movielens/ml-100k.zip` (4.9 MB). Terms (README, quoted in substance): research use allowed; must acknowledge; may not redistribute without permission; no commercial use without permission; may not imply endorsement. PyG: `MovieLens100K` (listed in PyG docs, stats not given there).
4. **Yelp user-business (Network Repository).** Rossi and Ahmed, AAAI 2015, plus Yelp as the data source. `https://nrvis.com/download/data/rec/rec-yelp-user-business.zip` (0.9 MB). Licence: Network Repository links a "Data License" policy page; the dataset page itself states none (not read). The third column is a rating; user and business IDs both start at 1, so the file must be read as bipartite with separate ID spaces.
5. **Yelp2018 (LightGCN split).** Same repo as Gowalla, `data/yelp2018/{train,test}.txt`. Same licence caveat (none stated; the underlying Yelp dataset has its own terms, not checked).
6. **act-mooc.** Kumar, Zhang, Leskovec, "Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks", KDD 2019; raw data from KDD Cup 2015. `https://snap.stanford.edu/data/act-mooc.tar.gz` (5.4 MB). No licence stated on the page. The SNAP index page labels it "bipartite", the dataset page does not say so itself (it follows from user-to-target edges).

**Existing codebase compatibility.** `load_snap_community` is hard-wired to `com-<name>.ungraph.txt.gz` with integer IDs and `identity_features`, and builds the CSR by summing symmetric entries, so `ca-*.txt.gz` (already double-listed edges) would produce entries of 2, not binary. `load_mag_coauthor` and `load_planetoid` read their own formats. Every candidate above therefore needs a small new loader (edge list -> deduplicated binary symmetric CSR -> `Graph` with identity features, plus bipartite ID offsetting for 1, 4-7). The downstream `split_edges` / `normalized_adjacency` pipeline takes a symmetric binary CSR, so a bipartite graph is just a block-structured adjacency of the same type; whether the existing evaluators (`apps.synthetic.graph_families`-style truth metrics) apply is another matter, since real graphs have no ground-truth edge probabilities. The truth-KL, ECE and calibration metrics that carry the headline would **not be computable** on real data; only the observable ones (log loss, ECE on held-out pairs, Hits@50, clustering/triangle error of sampled graphs, AUC, rank correlation) would.

## Candidates (contrast / losing-profile side)

| Dataset | Nodes | Edges | Mean deg | Degree CV | Transitivity | Verified how |
|---|---|---|---|---|---|---|
| roadNet-PA (SNAP) | 1,088,092 | 1,541,898 | 2.83 | 0.36 | 0.059 | Source (nodes/edges) + computed; max degree 9 |
| US power grid (Network Repository `power-US-Grid`, Watts and Strogatz 1998 data via Pajek/SuiteSparse) | 4,941 | 6,594 | 2.67 | 0.67 | 0.103 | Computed (file header names Pajek `USpowerGrid`) |

- **roadNet-PA**: `https://snap.stanford.edu/data/roadNet-PA.txt.gz` (9.9 MB). Cleanly sparse and homogeneous (the losing profile: mean degree under 3, CV 0.36), but 1.09M nodes is far too big as is; use a connected BFS patch of 5-20k nodes. SNAP's loc/road data lists edges once or twice (not checked), so dedupe.
- **US power grid**: `https://nrvis.com/download/data/power/power-US-Grid.zip` (24 KB). Useful because its CV (0.67) matches the winning profile's while its mean degree (2.7) does not: it separates the two covariates, which neither road nor collaboration graphs do. Licence not checked.
- Not downloaded: SNAP `ca-GrQc` (5,242 / 14,496, so mean degree about 5.5 by arithmetic on the page's counts) and `ca-CondMat` (23,133 / 93,497, mean about 8.1 by the same arithmetic) are intermediate-degree collaboration graphs; I did not compute their CVs.

## Things that did not pan out

- **Network Repository `rec-movielens-user-movies-10m`** (page: 7.6K nodes, 95.6K edges, avg degree 25, 1.1M triangles, clustering 0.20). The download is a user-movie list with overlapping ID spaces (`% bip positive`); read as bipartite it has 4,009 users + 7,601 items, 55,484 unique pairs, mean degree 9.6, and no triangles. The page's statistics come from reading it as an ordinary graph and are wrong for the real structure. Treat every Network Repository `rec-*` statistic as suspect until recomputed from the file.
- **Gene-disease**: DisGeNET's licence is unclear from what I could find (one source says ODbL 1.0 for a processing project, another lists CC BY-NC-SA 4.0 for a derived Therapeutics Data Commons benchmark; the official site was not read). The TDC curated subset is 52,476 pairs over 7,399 genes + 7,095 diseases, so mean degree about 7 (my arithmetic from the search snippet, unverified). Weak match on degree. Not downloaded.
- **Author-paper (PyG `DBLP`, `AMiner`), OGB**: sizes and stats not obtained; author-paper bipartite graphs are typically very sparse (papers have a few authors), so likely a weak match on mean degree. Not verified.
- **KONECT**: unreachable; nothing recorded.

## Ranked shortlist

1. **ca-AstroPh** (SNAP). Best practical test of the degree/CV/clustering profile in one small, well-documented file: n = 18.8k, mean degree 21.1, CV 1.45, local clustering 0.63. Weaknesses: unipartite (so does not exercise the one confirmed family, bipartite); its CV and clustering are far above the synthetic winners (0.68 / 0.084), so it is an extrapolation; no node features (identity features, as com-DBLP).
2. **Gowalla user-location, subsampled** (LightGCN split). Closest to the confirmed winning family (bipartite, mean degree 29, CV 1.58) with a heavy-tailed degree on both sides. Weaknesses: 70.8k nodes (subsample required, which changes the degree stats; recompute after subsampling), zero triangle clustering by construction, unclear licence, needs a new bipartite loader, and the check-in graph is not an observed-vs-latent probability model any more than the others.
3. **MovieLens-100K.** Smallest, fastest, PyG loader exists, licence explicit (research use, acknowledge, no redistribution). Bipartite, CV 1.2, but mean degree 76 is above the target 15-30 and the graph is 6% dense, which in the synthetic search is outside the explored range; a k-core or degree-capped version would be a post hoc choice.
4. **ca-HepPh.** Mean degree 19.7, CV 2.36, transitivity 0.66, clique-dominated; a stress case for the clustering-error metrics. Same loader and caveats as ca-AstroPh; highly similar to it in profile, so a second data point only.
5. **Yelp2018 (LightGCN)** or **NR Yelp user-business** as a second bipartite check: mean 44.8 (LightGCN) or 8.0 (Network Repository, sparse and ultra heavy-tailed, CV 3.0). The NR version is a deliberate within-bipartite sparse contrast: same family, mean degree about 8, so it speaks to H1 (degree within a family) rather than to family identity.
6. **Contrast set: US power grid (and a roadNet-PA patch).** Mean degree under 3 with homogeneous (road, CV 0.36) or moderately heterogeneous (power grid, CV 0.67) degrees. Expect, per the descriptive pattern, no TNBBeta advantage; a TNBBeta win here would count against the "degree" reading of the pattern.

## Honest summary

Good matches on the stated numbers: ca-AstroPh, ca-HepPh (unipartite), Gowalla (bipartite, needs subsampling). MovieLens-100K matches on bipartite and heterogeneity but is denser than the target. Everything real is much more heavy-tailed than the synthetic winners, and real data has no truth probabilities, so only a subset of the metrics that defined the pattern can be evaluated. None of these licences were reviewed in full except MovieLens; check each before publication use.
