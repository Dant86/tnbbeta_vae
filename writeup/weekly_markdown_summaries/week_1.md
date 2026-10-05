# Week 1 summary (through 2026-09-21) -- DRAFT, several numbers still pending

Goal for the week: compare TNBBeta with the Gaussian and vMF (S-VAE) latents on the experiments of
the S-VAE paper (Davidson et al. 2018): unsupervised MNIST (Table 1), latent k-NN (Table 2),
semi-supervised MNIST (Table 3), link prediction on citation graphs (Table 4), and the synthetic
S^1 recovery (section 5.1). Status: all experiments are built and tested; Tables 1, 2 (corrected
convention) and 3 are still running on the cluster. Marked **TBD** below.

## Headline results so far

1. **Our pipeline reproduces the paper's Gaussian baseline on MNIST.** With a conv encoder/decoder and
   the paper's protocol (dynamic binarization, 100-epoch KL warm-up, early stopping with patience 50,
   500-sample importance-weighted LL) the Gaussian's KL is within about 1 nat of the paper's and its
   reconstruction term within 1-4 nats at d = 2, 5, 10 (LL -132.6 / -105.4 / -88.8 vs the paper's
   -135.7 / -110.2 / -93.8; a few nats better, as expected for a conv model).
2. **The first sphere results were not like-for-like: the paper's d is the manifold dimension.** The
   paper's d=2 S-VAE is S^2 in R^3 (Figure 2: "Hammer projection of S^2 latent space"; the authors'
   example code builds the vMF model with `z_dim + 1`). We gave the sphere ambient dimension d, so at
   equal d it had one degree of freedom fewer than the Gaussian, and at d=2 it was a circle. That
   explained a vMF d=2 LL of -150.0 (paper: -132.5). With ambient d+1 the first finished d=2 run
   reaches a validation -ELBO of 133.9 (the paper's S-VAE ELBO: -133.7; our Gaussian's test ELBO:
   -136.8), i.e. it now shows the paper's roughly 3-nat advantage. Final numbers: **TBD**.
3. **The vMF head collapses at high dimension unless its concentration starts large.** With the
   reference initialization (kappa = softplus(raw) + 1, kappa about 1.7) a sample's expected cosine with
   its mean direction is 0.04 at d=40, so z is noise, the decoder learns to ignore it, and the model
   sits at the mean image (KL 1.1, log-likelihood -187; kappa stuck at about 10) for the whole run. An
   informative code needs kappa of about 26 (cosine 0.5) to 185 (0.9). Starting kappa at the latent
   dimension fixes it (validation loss about 97 at d=40, level with TNBBeta). TNBBeta is not affected.
   A synthetic proxy reproduced the collapse and the fix before the cluster runs confirmed them.
4. **On the synthetic S^1 recovery, TNBBeta ties vMF on reconstruction and beats the Gaussian on prior
   coverage, but does not beat vMF.** Three seeds, latent dimension 2, decoded prior samples' distance
   to the data manifold relative to held-out data: Gaussian 0.92 +- 0.17, vMF 0.19 +- 0.01, TNBBeta
   0.43 +- 0.24 (lower is better). All three locate the true point to about 0.01-0.013 rad after a
   round trip; TNBBeta's latent is not an aligned circle (p stays near 0.5, angle error of its centre
   0.76 rad), i.e. it encodes the point differently. Test LL 150.9 / 152.0 / 151.5, KL 6.05 / 4.84 /
   4.86 (Gaussian / vMF / TNBBeta).
5. **Link prediction (Table 4, partial):** sphere latents beat the Gaussian on Cora and Citeseer, as in
   the paper, and TNBBeta is level with vMF on Cora and slightly behind on Citeseer.
   Test AUC (%), mean +- std over 5 seeds, configuration chosen by validation AUC:

   | | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE | paper N / S |
   |---|---|---|---|---|
   | Cora | 92.8 +- 0.6 | 93.7 +- 0.8 | 93.5 +- 0.4 | 92.7 / 94.1 |
   | Citeseer | 92.0 +- 0.5 | 94.1 +- 0.7 | 93.4 +- 0.9 | 90.3 / 94.7 |
   | Pubmed | 96.0 +- 0.4 | **TBD** | 95.4 +- 0.3 | 97.1 / 96.0 |

   The sphere models multiply the inner product of unit vectors by a learned temperature (initially 5),
   which the paper does not describe; the paper's dimension convention is not stated for the GCN
   latents either, so d_z is simply searched over {16, 32, 64}.

## Not yet in (pending cluster runs)

- **Table 1 in the paper's convention** (`vmfs`, `tnbbeta` with ambient d+1 as `tnbs`, and the
  fixed-initialization vMF `vmfks`), d = 2, 5, 10, 20, 40, 5 seeds: **TBD**.
- **Table 2 (latent k-NN)** for the same runs: **TBD**. The earlier table (ambient d) already showed the
  sphere models ahead of the Gaussian at d = 5-20 (N=100: 0.86 / 0.85 vs 0.81 at d=5) and TNBBeta best at
  d=40 for N = 600 and 1000, but those rows are not like-for-like.
- **Table 3 (semi-supervised, 100 labels):** a 36-task first look (N+N, S+S, S+N, z1 and z2 in
  {10, 50}, 3 seeds): **TBD**. A 5-epoch smoke run reached 78.4% test accuracy.
- **Pubmed vMF link prediction:** the full grid probably exceeds the 6-hour job limit; a reduced grid or
  the paper's selected configuration is needed.

## CIFAR-10, d=128 (from the start of the week)

The width-256, 200-epoch runs overfit heavily (train PSNR about 7 dB above test; test PSNR 21.7 vs 23.1
for the earlier width-32, 50-epoch models) and their prior samples are poor for both families (FID 121.9
TNBBeta vs 109.0 Gaussian, real-data floor 5.2), so that checkpoint is not a good one to memorialize. The
architecture is not too small (25M parameters); a single flat latent is the limit. No further CIFAR work
was done this week.

## Wrong turns and corrections (kept for the record)

- I first guessed that our conv head simply *started* at kappa about 10; it starts at about 1.65, and the
  training log shows kappa rising to about 10 in epoch 0 and then stalling because the decoder had already
  ignored z.
- A log-scale kappa option ("exp") was tried to fix slow kappa growth at d=2. It sent kappa to its cap in
  epoch 0; at 4.85e8 the float32 KL collapsed to exactly log(2 pi) (the sample's cosine with its mean is
  exactly 1 from kappa about 1e7), then at a 1e6 cap it was unstable and ended in NaN weights. It was
  removed. (The float32 KL is accurate to about 0.01 nats up to kappa = 1e6.)
- The Citeseer link-prediction crashes were NaN directions from featureless, isolated nodes (0/0 in the
  vMF head; the TNBBeta head returned a zero vector). Both heads now fall back to a fixed unit vector.
- The restored vMF distribution is a faithful port of the authors' code (same sampler, Bessel function
  and gradient, kappa = softplus + 1, analytic KL); the only intentional deviation is exact Householder
  normalization, which fixed samples drifting off the sphere at d=2.
- Infrastructure failures unrelated to the model: node k003 has GPUs that do not initialize and m001
  returns `Input/output error` on scratch. Both are now handled by an automatic retry that excludes the
  node.

## Repository changes this week (PR numbers)

MNIST loader with dynamic binarization, Bernoulli likelihood, validation/early stopping/KL warm-up and
best-epoch checkpoints in `Trainer` (#17, #18); restored vMF baseline; importance-weighted Table 1
metrics, k-NN, S^1 recovery (registered `mlp_vae`), link prediction (`graph_vae`, Planetoid loader), and
the M1+M2 semi-supervised model with any latent family (#17, #18, #20); GPU-resident MNIST batches (#18);
FID, linear probes and summaries for CIFAR (#14); interactive d=3 sphere plots (#16); node-failure
retries (#19, #26); fixes for degenerate directions (#21), the vMF concentration (#22-#24, #25) and the
paper's dimension convention in the sweeps (#24, #25).

## Open questions

1. Does TNBBeta's extra flexibility ever help against a properly initialized vMF at the paper's
   dimension? Table 1 and Table 2 in the like-for-like convention decide this; so far it ties on
   reconstruction and coverage.
2. Is the Gaussian's lead at d >= 20 on MNIST (Table 1, ambient convention) confirmed with the
   corrected sphere models, or was part of it the missing degree of freedom?
3. TNBBeta's q stays near 0 on CIFAR; does it on MNIST (the exported posteriors were not analysed yet)?
