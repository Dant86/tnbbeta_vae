# Sampling speed, vMF vs TNBBeta, GPU, 2026-09-26

Answers the standing efficiency question from earlier in this project ("we don't
need rejection sampling -- does that mean faster training?") directly, at the
level where the theoretical difference actually lives: `rsample()` wall-clock
time. Produced by:

```
sbatch --wrap='uv run --no-sync python -m apps.benchmark.sampling_speed \
    --dims 2 5 10 20 40 100 --batch-sizes 64 1024 8192 --repeats 30 --warmup 5' \
    --partition=general --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:20:00
```

## Headline result

**TNBBeta's sampling beat vMF's at every one of 72 tested cells** (6 dimensions x
3 batch sizes x 4 concentration levels): 2.1x to 27.5x faster, mean 9.4x overall.
At the actual batch size used for MNIST training in this project (64): mean 4.4x,
range 2.1x-6.6x.

```latex
\begin{table}[t]
\centering
\caption{\texttt{rsample()} wall-clock time, batch size 64, mean $\pm$ std over
30 timed calls (5 untimed warm-up calls first), GPU. "Weak" and "extreme" are
$\kappa{=}1$ and $\kappa{=}1000$ for vMF, and independently chosen $\epsilon$
values for TNBBeta (there is no exact translation between them, and TNBBeta's
cost should not depend on concentration at all by construction -- itself part of
what is being tested).}
\label{tab:sampling-speed}
\begin{tabular}{llccc}
\toprule
$d$ & concentration & vMF (ms) & TNBBeta (ms) & speedup \\
\midrule
2   & weak    & 3.83 $\pm$ 1.20 & 0.89 $\pm$ 0.01 & 4.3$\times$ \\
2   & extreme & 5.77 $\pm$ 0.78 & 0.90 $\pm$ 0.01 & 6.4$\times$ \\
5   & weak    & 2.83 $\pm$ 0.43 & 0.91 $\pm$ 0.01 & 3.1$\times$ \\
5   & extreme & 5.23 $\pm$ 0.91 & 0.90 $\pm$ 0.00 & 5.8$\times$ \\
10  & weak    & 2.03 $\pm$ 0.38 & 0.89 $\pm$ 0.01 & 2.3$\times$ \\
10  & extreme & 5.11 $\pm$ 0.63 & 0.89 $\pm$ 0.00 & 5.8$\times$ \\
20  & weak    & 2.05 $\pm$ 0.38 & 0.89 $\pm$ 0.00 & 2.3$\times$ \\
20  & extreme & 5.30 $\pm$ 0.98 & 0.89 $\pm$ 0.01 & 5.9$\times$ \\
40  & weak    & 2.00 $\pm$ 0.31 & 0.89 $\pm$ 0.00 & 2.2$\times$ \\
40  & extreme & 5.22 $\pm$ 1.02 & 0.89 $\pm$ 0.01 & 5.8$\times$ \\
100 & weak    & 1.91 $\pm$ 0.01 & 0.90 $\pm$ 0.01 & 2.1$\times$ \\
100 & extreme & 4.69 $\pm$ 1.00 & 0.90 $\pm$ 0.01 & 5.2$\times$ \\
\bottomrule
\end{tabular}
\end{table}
```

(full 72-cell sweep, including batch sizes 1024 and 8192, in
`sampling_speed_benchmark.json`)

A colorized heatmap of the same speedup ratio, all three swept variables at
once (one panel per batch size, dimension x concentration, Plasma colorscale,
shared colorbar) -- `sampling_speed_heatmap.png`, produced by:

```
uv run python -m apps.benchmark.sampling_speed_plot \
    --input sampling_speed_benchmark.json --output sampling_speed_heatmap.png
```

## Reading it

- **TNBBeta's cost is flat**: 0.900 ms mean, 0.009 ms std, across all 72 cells --
  every dimension, batch size (up to 8192) and concentration level tested. A
  single deterministic `Beta(epsilon, epsilon)` transform has no dependence on
  any of these, exactly as predicted; at these batch sizes it looks
  latency-bound (fixed Python/kernel-launch overhead) rather than compute-bound.
- **vMF's cost scales with three things independently, all consistent with a
  rejection sampler paying a per-round CPU-GPU synchronization cost**:
  - *Batch size*: 5.8ms to 19.6ms (dim=40, extreme) going from batch 64 to 8192 --
    a 128x larger batch only costs ~3.8x more. The mechanism isn't more total
    proposals so much as order statistics: the while loop
    (`src/tnbbeta_vae/distributions/von_mises_fisher.py::_while_loop`) only
    terminates once *every* row in the batch has been accepted, so a larger
    batch has a higher chance of containing one unlucky row that needs several
    extra rounds, stalling the whole call.
  - *Concentration*: 6.85ms to 19.6ms (dim=40, batch=8192) from weak to extreme
    -- higher kappa lowers Wood's proposal acceptance rate, needing more rounds.
    TNBBeta shows no analogous trend at all.
  - *Dimension*, non-monotonically: worst at dim=2 (15.7x), improving through
    dim=10-20 (~6.6-6.9x), rising again by dim=100 (10.5x) -- a U-shape from how
    Wood's proposal envelope quality varies with the interaction between kappa
    and ambient dimension in the `b, a, d` derivation, not something either
    family's design choices control directly.
- **Worst case**: dim=2, batch=8192, extreme concentration -- 27.5x. **Best case
  for vMF**: dim=100, batch=64, weak/strong -- 2.1x. vMF is slower everywhere;
  the only question the sweep answers is by how much.

## What this doesn't measure

This times `rsample()` alone, in isolation, with one shared concentration per
call. Two things would make real training's gap *larger*, not smaller:

1. A real minibatch has a different kappa/epsilon per example (from the
   encoder), not one shared value. Since vMF's while loop only terminates once
   every row is accepted, a single unusually-confident example is enough to
   stall the whole batch -- exactly the batch-size-driven straggler effect
   observed above, except now driven by heterogeneity within one batch rather
   than just batch size.
2. This isn't full end-to-end training-step time (encoder forward, decoder
   forward, backward pass all add more). Using the batch=64 numbers as a
   lower-bound estimate of the *sampling-only* saving: mean vMF (3.96ms) minus
   mean TNBBeta (0.90ms) is about 3.1ms/step. A run of the length seen in the
   fixed-mean-direction ablation (~84,000 steps before an early failure) would
   have spent roughly 4-5 minutes more on sampling alone with vMF -- real,
   but a fraction of total wall-clock time, not the whole story of "faster
   training."

## Bottom line

The efficiency question from earlier in the project has a clear, quantified
answer for the piece it was actually about: TNBBeta's rejection-free sampling
is faster than vMF's rejection sampler, at every setting tested, by a wide and
consistent margin (2x at best, order-of-magnitude at worst). Whether that
translates into a meaningfully faster *end-to-end* training run, or into a
lower-variance gradient estimator (the other half of the original question),
is still open and would need its own measurement.
