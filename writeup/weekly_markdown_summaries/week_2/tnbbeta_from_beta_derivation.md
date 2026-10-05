# Deriving the TNBBeta density from a $\mathrm{Beta}(\varepsilon,\varepsilon)$ sample

This note works through the univariate half of Algorithm 1 (the part before the
Householder lift): how a $\mathrm{Beta}(\varepsilon,\varepsilon)$ draw is turned into a
$\mathrm{TNBBeta}(p,q,\varepsilon)$ sample, and how that same transformation, run
through the change-of-variables formula, reproduces the closed-form density from
[Lederman & Schein, 2026]. It's a from-scratch derivation, not a transcription of the
source paper's proof, and it's been checked numerically against
`TNBBetaUnivariate`'s closed form (composed forward map + finite-difference Jacobian
vs. the closed form, 20 random $(p,q,\varepsilon,y)$ points, worst relative error
$3\times 10^{-9}$).

## 1. The transform

Start with $Z \sim \mathrm{Beta}(\varepsilon, \varepsilon)$, density

$$
f_Z(z) = \frac{z^{\varepsilon - 1}(1-z)^{\varepsilon-1}}{B(\varepsilon,\varepsilon)}, \qquad z \in (0,1).
$$

Build $Y \sim \mathrm{TNBBeta}(p,q,\varepsilon)$ via two successive, invertible
reparameterizations:

$$
S = 2Z - 1 \in (-1,1),
$$

$$
U = H_q(S) \defeq \frac{1}{2}\left(1 + S\sqrt{\frac{1-q}{1 - qS^2}}\right) \in (0,1),
$$

$$
Y = T_p(U) \defeq \frac{pU}{(1-p)(1-U) + pU} \in (0,1).
$$

$H_q$ is the "concentration tilt": at $q=0$ it's the identity ($U=S$ up to the
$[-1,1]\to[0,1]$ shift), and as $q\to 1$ it pushes $S$'s mass toward $\pm 1$. $T_p$
re-centers the median from $1/2$ to $p$. Both are strictly increasing bijections on
their domains for $q\in[0,1)$, $p\in(0,1)$, so the whole map $Z \mapsto Y$ is a
monotonic bijection of $(0,1)$ and the ordinary 1-D change-of-variables formula
applies directly:

$$
f_Y(y) = f_Z(z(y))\left|\frac{dz}{dy}\right|, \qquad
\frac{dz}{dy} = \frac{dz}{ds}\cdot\frac{ds}{du}\cdot\frac{du}{dy}.
$$

## 2. Two structural lemmas

Everything downstream falls out of two facts about $T_p$ and $H_q$ that are worth
isolating first, because they're what turn a page of algebra into a few lines.

**Lemma 1 ($T_p$ adds log-odds).** $T_p$ satisfies

$$
\operatorname{logit}(Y) = \operatorname{logit}(U) + \operatorname{logit}(p),
$$

i.e. $Y = \operatorname{sigmoid}\big(\operatorname{logit}(U) + \operatorname{logit}(p)\big)$, equivalently
$U = \operatorname{sigmoid}(L)$ with $L \defeq \operatorname{logit}(y) - \operatorname{logit}(p)$.

*Proof.* $\operatorname{sigmoid}(a+b) = \dfrac{e^{a+b}}{1+e^{a+b}}$; writing $e^a =
u/(1-u)$ and $e^b = p/(1-p)$ and simplifying gives exactly $T_p(u)$. $\blacksquare$

This is also why $T_p^{-1} = T_{1-p}$: flipping the sign of $\operatorname{logit}(p)$ undoes the
shift.

**Lemma 2 ($\gamma$ is a variance term).** Recall
$\gamma(y,p) \defeq \tfrac14\operatorname{sech}^2\!\left(\tfrac{L}{2}\right)$ with $L$ as above. Then

$$
\gamma(y,p) = U(1-U).
$$

*Proof.* Standard logistic identity: for $L = \operatorname{logit}(u)$,
$\operatorname{sigmoid}'(L) = u(1-u)$, and also $\operatorname{sigmoid}'(L) = \tfrac14\operatorname{sech}^2(L/2)$
(from $\operatorname{sigmoid}(L) = \tfrac12(1+\tanh(L/2))$). Combine with Lemma 1's
$U=\operatorname{sigmoid}(L)$. $\blacksquare$

So $\gamma(y,p)$ isn't an arbitrary-looking $\operatorname{sech}^2$ term — it's just the
Bernoulli variance of $U$ under the $p$-recentering, written in $y$-coordinates.

## 3. The three Jacobian factors

**(a) $Z \to S$.** $z = (1+s)/2 \Rightarrow dz/ds = \tfrac12$.

**(b) $S \to U$.** Write $u = \tfrac12(1+h(s))$ with $h(s) = s\sqrt{\tfrac{1-q}{1-qs^2}}$. Differentiating,

$$
h'(s) = \frac{(1-q)^{1/2}}{(1-qs^2)^{3/2}}\Big[(1-qs^2) + qs^2\Big] = \frac{(1-q)^{1/2}}{(1-qs^2)^{3/2}},
$$

so $\dfrac{du}{ds} = \dfrac12 (1-q)^{1/2}(1-qs^2)^{-3/2}$.

It's cleaner to re-express $1-qs^2$ in terms of $T \defeq 2U-1 = h(S)$. Squaring $h$
and solving for $s^2$ gives $s^2 = \dfrac{T^2}{(1-q)+qT^2}$, hence

$$
1 - qs^2 = \frac{1-q}{(1-q) + qT^2}.
$$

Using $1 - T^2 = 1-(2U-1)^2 = 4U(1-U) = 4\gamma$ (Lemma 2), the denominator simplifies to a single, very recognizable quantity:

$$
M \defeq (1-q) + qT^2 = (1-q) + q(1-4\gamma) = 1 - 4q\gamma.
$$

That's the $[1-4q\gamma]$ term from the closed form, falling directly out of the
$H_q$ Jacobian — before $p$ or $\varepsilon$ have even entered the picture. With
$1-qs^2 = (1-q)/M$,

$$
\frac{du}{ds} = \frac12 (1-q)^{1/2}\left(\frac{1-q}{M}\right)^{-3/2} = \frac{M^{3/2}}{2(1-q)}
\quad\Longrightarrow\quad
\frac{ds}{du} = \frac{2(1-q)}{M^{3/2}}.
$$

**(c) $U \to Y$.** From $y = pu/D(u)$ with $D(u) \defeq (1-p)(1-u)+pu$, a short computation gives $D(u) - u(2p-1) = 1-p$, so

$$
\frac{dy}{du} = \frac{p(1-p)}{D(u)^2} \quad\Longrightarrow\quad \frac{du}{dy} = \frac{D(u)^2}{p(1-p)}.
$$

To eliminate $D(u)$, note $1-y = (1-p)(1-u)/D(u)$, so

$$
y(1-y) = \frac{p(1-p)\,u(1-u)}{D(u)^2} = \frac{p(1-p)\gamma}{D(u)^2}
\quad\Longrightarrow\quad
D(u)^2 = \frac{p(1-p)\gamma}{y(1-y)}.
$$

Substituting back,

$$
\frac{du}{dy} = \frac{1}{p(1-p)}\cdot\frac{p(1-p)\gamma}{y(1-y)} = \frac{\gamma}{y(1-y)}.
$$

$p$ and $(1-p)$ cancel completely — the $U\to Y$ Jacobian only remembers $p$ through $\gamma$.

## 4. Assembling the density

Chain the three factors:

$$
\frac{dz}{dy} = \frac{dz}{ds}\cdot\frac{ds}{du}\cdot\frac{du}{dy}
= \frac12 \cdot \frac{2(1-q)}{M^{3/2}} \cdot \frac{\gamma}{y(1-y)}
= \frac{(1-q)\gamma}{M^{3/2}\,y(1-y)}.
$$

For $f_Z(z)$, use $z(1-z) = (1-s^2)/4$ and, by the same substitution as above,
$1-s^2 = (1-q)(1-T^2)/M = 4(1-q)\gamma/M$, so

$$
z(1-z) = \frac{(1-q)\gamma}{M}.
$$

Then

$$
f_Z(z) = \frac{[z(1-z)]^{\varepsilon-1}}{B(\varepsilon,\varepsilon)}
= \frac{1}{B(\varepsilon,\varepsilon)}\left[\frac{(1-q)\gamma}{M}\right]^{\varepsilon-1}.
$$

Multiplying $f_Z(z)$ by $dz/dy$, one power of $(1-q)\gamma/M$ combines with the
leftover $(1-q)\gamma/M^{1/2}$ to give:

$$
f_Y(y) = f_Z(z)\left|\frac{dz}{dy}\right|
= \frac{1}{B(\varepsilon,\varepsilon)}\cdot\frac{[(1-q)\gamma]^{\varepsilon}}{M^{\varepsilon + 1/2}\, y(1-y)}.
$$

Substituting $M = 1-4q\gamma$ and splitting off $y^{\varepsilon-1}(1-y)^{\varepsilon-1}/B(\varepsilon,\varepsilon) = \mathrm{Beta}(y;\varepsilon,\varepsilon)$:

$$
\boxed{\
f_Y(y) = \mathrm{TNBBeta}(y; p, q, \varepsilon)
= \mathrm{Beta}(y;\varepsilon,\varepsilon)\left[\frac{1-q}{y(1-y)}\gamma(y,p)\right]^{\varepsilon}
\big[1-4q\gamma(y,p)\big]^{-\left(\varepsilon + \frac12\right)}\ }
$$

which is exactly the closed form in [Lederman & Schein, 2026] and in
`TNBBetaUnivariate`.

## 5. Where each piece of the formula actually comes from

Worth stating plainly, since the closed form looks unmotivated until you've done
the derivation:

- The $\mathrm{Beta}(y;\varepsilon,\varepsilon)$ factor is $f_Z$ itself, just relabeled
  in $y$-coordinates — $\varepsilon$ never interacts with $p$ or $q$ in the
  derivation above, which is why it appears as a clean multiplicative baseline
  rather than being tangled up with the other two parameters.
- The $\gamma(y,p)^\varepsilon$ and $y(1-y)$ terms come from $z(1-z)$ being
  re-expressed via $\gamma$ (Lemma 2) and raised to the $(\varepsilon-1)$ power
  from $f_Z$, plus one more power of $\gamma/y(1-y)$ contributed by the $du/dy$
  Jacobian in step 3(c) — that's the whole reason the exponent on $\gamma$ ends up
  as $\varepsilon$ (from $f_Z$) rather than $\varepsilon - 1$.
- The $[1-4q\gamma]^{-(\varepsilon+1/2)}$ term is entirely a $q$-effect: $\varepsilon-1$
  powers of $1/M$ from $f_Z$, plus $3/2$ powers of $1/M$ from the $H_q$ Jacobian
  alone (step 3(b)) — $p$ plays no role in this factor at all, which matches $q$'s
  role as a pure concentration/tilt parameter independent of where the median sits.

## 6. Where the transform itself comes from

Sections 1-5 verify that $H_q$ and $T_p$ reproduce the paper's closed-form density
exactly, but that alone doesn't explain *why* they take the specific form they do —
neither map is written down anywhere in Lederman & Schein (2026) as a sampling
recipe. It's worth being precise about what *is* and *isn't* in the paper, since the
two maps have very different provenance.

**The paper's own generative story is a discrete hierarchical mixture, built for
Gibbs sampling, not reparameterized sampling.** Theorem 4.1 defines TNBbeta as the
marginal of

$$
C \sim \mathrm{NB}(\varepsilon, 1-q), \quad
A \mid C{=}c \sim \mathrm{NB}(\varepsilon+c, 1-p), \quad
B \mid C{=}c \sim \mathrm{NB}(\varepsilon+c, p),
$$
$$
Y \mid A{=}a, B{=}b, C{=}c \sim \mathrm{Beta}(\varepsilon+c+a,\ \varepsilon+c+b).
$$

Three coupled negative-binomial draws perturb a Beta's two shape parameters — $A,B$
pulling them apart toward $p$, $C$ inflating both together toward concentration.
This exists because the paper's goal is closed-form Bayesian regression: each
auxiliary variable has a negative-binomial complete conditional, which is exactly
what makes a Pólya-Gamma-augmented Gibbs sampler work. None of that requires
differentiability, so reading the paper top to bottom gives no hint of a smooth
sampling transform — the mismatch in goals (conjugate MCMC vs. a VAE's
reparameterization trick) is the whole reason it looks unmotivated.

**$T_p$ is not invented — it's the paper's own $q=0$ special case.** The paper
analytically marginalizes $A,B$ out (Corollary 3.1):

$$
\mathrm{TNBbeta}(y,p,q,\varepsilon) = \mathrm{LNbeta}\!\left(y;\varepsilon,\varepsilon,\tfrac{1-p}{p}\right)
(1-q)^{\varepsilon}\big[1-4q\gamma(y,p)\big]^{-(\varepsilon+1/2)},
$$

where $\mathrm{LNbeta}(y;\alpha_1,\alpha_2,c) = \mathrm{beta}(y;\alpha_1,\alpha_2)\,
c^{\alpha_1}/(1-(1-c)y)^{\alpha_1+\alpha_2}$ is the Libby–Novick beta (an
odds-ratio-tilted Beta). At $q=0$, $(1-q)^\varepsilon=1$ and the bracket is $1$,
leaving exactly $\mathrm{LNbeta}(y;\varepsilon,\varepsilon,\frac{1-p}{p})$ with no
residual randomness. That tilted-Beta density is, by construction, exactly what
$T_p$ applied to a $\mathrm{Beta}(\varepsilon,\varepsilon)$ draw produces (Lemma 1
above is precisely "add $\operatorname{logit}(p)$," which is the odds-ratio tilt).
So $T_p$ is the paper's Corollary 3.1 reduction, just re-expressed as a *sampling*
map instead of a *density* formula.

**$H_q$ is a differentiable stand-in for the discrete mixing variable $C$.** The
paper states the general (non-degenerate) mixture directly:

> "If $Y \mid C{=}c \sim \mathrm{LNbeta}(\varepsilon+c,\varepsilon+c,\tfrac{1-p}{p})$
> and $C \sim \mathrm{NB}(\varepsilon, 1-q)$, then the marginal is
> $Y \sim \mathrm{TNBbeta}(p,q,\varepsilon)$."

$\mathrm{NB}(\varepsilon, 1-q)$ has success probability $1-q$; at $q=0$ that's
probability $1$, so $C \equiv 0$ deterministically, exactly matching Corollary 3.1.
For $q>0$, $C$ is a genuine random shape boost, analytically summed out to produce
the $[1-4q\gamma]^{-(\varepsilon+1/2)}$ term — a scale-mixture construction, the same
family of trick as building a Student-$t$ by mixing a Gaussian's variance over an
Inverse-Gamma. It's elegant for conjugacy, but $C$ is discrete: there is no way to
reparameterize through it.

So $H_q$ was built (for this project, not sourced from the paper) by solving for a
smooth, deterministic, invertible map that reproduces the *exact same marginal* you
would get by averaging over $C$'s distribution, without ever drawing $C$. That's
precisely why $H_q$'s Jacobian alone produces $M=(1-q)+qT^2$, which collapses to
$1-4q\gamma$ (Section 3(b)) — the identical quantity the paper derives by explicitly
summing the negative-binomial mixing distribution over $c$. Same target term, two
mechanisms: the paper gets it by averaging over a discrete auxiliary variable; the
sampler gets it by change-of-variables on a continuous bijection, engineered so the
two are forced to agree.

**Summary.** Nothing in the transform is unrelated to the paper — $T_p$ *is* the
paper's own Corollary 3.1 special case, and $H_q$ is a hand-built, differentiable
replacement for the paper's $C \sim \mathrm{NB}(\varepsilon,1-q)$ mixing variable,
constructed to land on the identical marginal by design rather than by following the
paper's own (non-differentiable, Gibbs-sampling-oriented) generative recipe. The
paper never needed a reparameterizable sampler — its own inference machinery is
built entirely around the discreteness of $C$ — so this transform is original to
this project's use case, not a transcription of anything in the source paper.

*Caveat:* the quotes above come from an arXiv HTML fetch-and-summarize pass, not a
full manual read of the PDF, and the exact corollary/theorem numbering was slightly
inconsistent across two independent fetches (4.1 vs. 4.2 for the mixture statement)
even though the content agreed both times. Worth pinning down precisely before
citing a specific corollary number in the writeup.
