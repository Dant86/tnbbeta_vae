# TNBBeta-Spherical Diffusion: Forward-Sampling Mechanics

Companion to
[`docs/superpowers/specs/2026-10-05-sphere-diffusion-prior-design.md`](superpowers/specs/2026-10-05-sphere-diffusion-prior-design.md),
which records the scope decisions (why direct regression instead of DSM,
why a frozen VAE, what's deferred). This document is the derivation that
spec's "Background" section condenses to eight bullet points: exactly how
a forward-noised sample is drawn, and exactly which steps are "conjugacy"
versus something else wearing similar clothes. Implementation:
`src/tnbbeta_vae/diffusion/{forward,schedule,noising}.py`. Numerical
verification (KS tests against `TNBBetaUnivariate`'s closed-form density):
`notebooks/tnbbeta_*.py`, promoted into `tests/diffusion/`.

## The object being noised

TNBbeta's density (Lederman & Schein 2026, arXiv:2606.11624, Definition
3.1) has a closed form directly in `y`. But the construction this document
cares about is their Theorem 4.1 augmentation, which represents the same
`y` via four latent variables:

```
C          ~ NB(eps, 1-q)
A | C = c  ~ NB(eps+c, 1-p)
B | C = c  ~ NB(eps+c, p)
Y | A,B,C  ~ Beta(eps+C+A, eps+C+B)
```

`C`'s role is the one this document is about: it's a NB count whose *size*
parameter is `eps` and whose *probability* parameter is `1-q`. Everything
below is about moving `C` from one `(eps, q)` to another, exactly, in
closed form, without simulating anything in between.

Every NB formula in this document uses the paper's convention above
(`q`/`p` as the *stopping* probability). The code
(`src/tnbbeta_vae/diffusion/forward.py`) writes the complements throughout
-- `torch.distributions.NegativeBinomial(r, probs=theta)`'s `theta` is the
probability of the *counted* event, i.e. `theta = 1 - q` (or `1 - p`) in
this notation -- so `probs=q` in the code is the correct implementation of
`NB(r, 1-q)` above, not a bug.

## Forward sampling, step by step

Given a starting point (either the data's own `(p_data, q_data, eps_data)`,
or — as actually used in `SphereDiffusionPrior` — a fixed near-point-mass
`(p_start≈1, q_start=0, eps_start=large)` centered at `mean_direction =
z_0`), and a target time `t`:

**1. Resize `eps`** (`forward.resize_eps`): compute `eps_t = eps_target +
(eps_data - eps_target) * exp(-t)`, then move `C` from `eps_data` to
`eps_t`, holding `q` fixed throughout. If `eps_t < eps_data`, split: draw
`C_t ~ BetaBinomial(C; eps_t, eps_data - eps_t)`. If `eps_t > eps_data`,
merge: draw an independent fresh `NB(eps_t - eps_data, 1-q)` and add it to
`C`. Exact either way (see "Where conjugacy comes in," fact 1 and 3).

**2. Move `q` toward `q_target`** (`forward.leisen_step`): a *second*,
independent reshaping of `C`, this time changing its probability
parameter rather than its size. One-shot, closed form:
```
Theta_t = (1 - q_target) / (exp(speed*t) - q_target)
Y_bin ~ Binomial(C_t, Theta_t)
Z     ~ NB(eps_t + Y_bin, 1 - q_target*(1 - Theta_t))
C_t  <- Y_bin + Z
```
`Theta_t` is engineered (Leisen, Mena, Palma Mancilla & Rossini 2019,
arXiv:1812.07271, their Proposition 1) so this is a genuine continuous-time
Markov *semigroup* in `t` — not just a one-off coupling. At `t=0`,
`Theta_0=1`, making this exactly the identity. As `t→∞`, `Theta_t→0`,
making `C_t`'s law converge to `NB(eps_t, q_target)` *regardless of the
C_t it started from*. See fact 2 below for why this is closed-form at
all.

**3. Redraw `A`, `B` fresh at the new `(eps_t, C_t)`**, with `p` pinned at
0.5 (not the caller's actual `p`): `A_t ~ NB(eps_t+C_t, 0.5)`, `B_t ~
NB(eps_t+C_t, 0.5)`, then `U_t ~ Beta(eps_t+C_t+A_t, eps_t+C_t+B_t)`. This
is just Theorem 4.1's own recipe, re-applied at the new state — no new
math, but it's the step that turns the resized/renoised `C_t` back into a
continuous `(0,1)` value.

**4. Reinstate `p`** (`_logit`/`torch.sigmoid` in `forward.draw_latitude`):
`U_t` above is the `p=0.5` member of the TNBbeta family. The actual `p` is
folded back in via an *exact algebraic identity*, not a probabilistic
construction: `logit(Y) = logit(p) + logit(U)` (derivable directly from
the Möbius-type transform in `TNBBetaUnivariate.rsample()`). So `psi_t =
logit(p_start) * exp(-t)`, `Y_t = sigmoid(psi_t + logit(U_t))`, `w_t = 2
Y_t - 1`.

**5. Lift to the sphere** (`noising.noise_to`): draw a fresh `v ~
Uniform(S^(dim-2))`, independent of everything above; form the pole-frame
point `(w_t, sqrt(1-w_t^2)*v)`; Householder-reflect it onto
`mean_direction`. This step needs no new theory — `TNBBetaSpherical`
already draws its non-latitude direction this way regardless of `(p,q,
eps)`, which is exactly why a uniform latitude plus an independent uniform
azimuthal direction, reflected through *any* fixed pole, is rotationally
symmetric: the whole diffusion problem lives in steps 1-4, on a scalar.

Every step above is **one-shot**: given `t`, there is a direct formula for
the state at `t` from the state at `0` — no simulated path through
intermediate `t`'s, unlike RSGM's Geodesic Random Walks. That's the
practical payoff of the conjugacy below: it's what makes step 2 a formula
instead of a numerical integration.

## Where conjugacy comes in

Three distinct facts get called "conjugacy" in this project's notes, and
they're related (all three ultimately trace back to the same
Gamma-Poisson representation of a negative binomial) but are not the same
*kind* of fact. Worth keeping separate:

**Fact 1 — NB is infinitely divisible in its size parameter.** For a fixed
probability `θ`, `NB(r1, θ) + NB(r2, θ) ~ NB(r1+r2, θ)` when independent.
This is just a convolution-closure property (the same kind of fact that
makes sums of independent Gaussians Gaussian, or sums of independent
Gammas Gamma) — not Bayesian conjugacy in the textbook sense, no prior or
posterior involved. It's what licenses step 1's *merge* direction, and
it's the reason the Gamma-ratio realization of this same operation
(`Beta(eps,eps) = Gamma(eps,1)/(Gamma(eps,1)+Gamma(eps,1))`, Gamma also
being shape-additive) is fully continuous and differentiable — verified in
`notebooks/tnbbeta_gamma_merge_check.py`, not used in the current
implementation (sampling doesn't need to be differentiable here; see the
design spec, point 6).

**Fact 2 — NB is conjugate to Binomial thinning, which is textbook
Bayesian conjugacy.** This is the one actually load-bearing for step 2.
Leisen et al.'s construction (following Pitt et al. 2002 / Mena & Walker
2009's general recipe for building a reversible Markov chain from a target
stationary law) is literally: treat the target `NB(r,q)` as a *prior* on a
count `X`; observe a `Binomial(X,Θ)`-thinned version `Y` of it (the
*likelihood*); the *posterior* `X|Y` is again negative binomial,
`Y + NB(Y+r, q(1-Θ))` — this is the actual conjugate-prior relationship
(prior family = posterior family under this specific observation model),
the same structural pattern as Beta-Binomial or Gamma-Poisson conjugacy,
just with NB playing both roles. The one-step reversible kernel is then
built by averaging over the augmenting variable: `P(x,A) =
E_{Y|x}[P_{X|Y}(A)]`. This conjugacy is *why* step 2 has a closed form at
all — the alternative, without it, is what RSGM has to do instead
(truncate an infinite Sturm-Liouville eigenfunction series, or fall back
to Varadhan short-time asymptotics) for a general manifold's heat kernel.

**Fact 3 — Beta-Binomial splitting, a close cousin of Fact 2.** Step 1's
*split* direction uses: if `C1 ~ NB(eps1,θ)`, `C2 ~ NB(eps2,θ)`
independent, then `C1 | C1+C2=c ~ BetaBinomial(c; eps1, eps2)`. This
follows from the same Gamma-Poisson representation underlying Fact 2 (the
ratio of two independent Gammas with shared rate is Beta, independent of
their sum — a standard, separately well-known fact), and is *also* a
conjugate-posterior statement (Beta prior on a split proportion, Binomial
observation, Beta-Binomial posterior predictive) — but it's a different
conjugate pair than Fact 2's NB-on-NB, even though both ultimately come
from the same underlying Gamma-Poisson mixture.

**What is explicitly *not* conjugacy:** step 4 (the `p`-decoupling) is a
closed-form algebraic identity about a deterministic reparameterization,
not a probabilistic/Bayesian fact at all — it holds because of how the
Möbius-type shift is constructed, not because of any prior/posterior
relationship. Step 5 (the spherical lift) is pure rotational-symmetry
geometry. Neither needed new derivation beyond noticing they were already
true of the existing, validated `TNBBetaSpherical`/`TNBBetaUnivariate`
code.

## What this does *not* give us

The conjugacy above makes **sampling** `w_t | w_0` closed-form at any `t`.
It does not, by itself, give a closed-form **density** `p_t(w_t | w_0)` —
that would require marginalizing the above over every possible integer
value of the latent counts, which (unlike the sampling recursion) doesn't
collapse to something simple. That gap is exactly why `SphereDiffusionPrior`
trains via direct `z_0`-regression rather than literal denoising score
matching (design spec, point 6) — the forward process being closed-form
for sampling was enough to build correct training *pairs*, not enough to
write down an exact score.
