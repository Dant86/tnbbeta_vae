# TNBBeta vs. Power Spherical: non-containment and an expressivity gap

This note settles the two theory items under "Avenue 2" of
[`week_2_plan.md`](week_2_plan.md): whether the TNBBeta and Power Spherical
families are nested (**no** — they meet only at $\mathrm{Uniform}(S^{d-1})$, and this is a
*proof*, not numerical evidence), and whether TNBBeta can express shapes Power
Spherical provably cannot (**yes** — but via a different parameter than the plan
expected; see §7, which corrects the plan).

Notation and the $T_p$/$H_q$ transform are as in
[`tnbbeta_from_beta_derivation.md`](tnbbeta_from_beta_derivation.md); that note is
assumed, not re-derived. Every claim below is labelled **Proof** (closed, checkable)
or **Numerical** (verified by computation over a finite parameter sweep, with the
sweep's extent stated). §8 has the code and the raw outputs.

## 0. The two families, written in comparable form

Both families are axially symmetric about a direction $\boldsymbol\mu\in S^{d-1}$,
and both are constructed as "draw a latitude, draw a uniform tangent direction,
Householder-reflect." So each is determined by its *latitude marginal*
$W=\boldsymbol\mu^\top X\in[-1,1]$, which both constructions express as $W=2Y-1$ for a
variable $Y$ on $(0,1)$:

| family | latitude variable $Y$ |
|---|---|
| $\mathrm{PowerSpherical}(\boldsymbol\mu,\kappa)$ | $\mathrm{Beta}\!\left(\tfrac{d-1}2+\kappa,\ \tfrac{d-1}2\right)$ |
| $\mathrm{TNBBetaSpherical}(\boldsymbol\mu,p,q,\varepsilon)$ | $\mathrm{TNBBeta}(p,q,\varepsilon)$ |

Write $\beta \defeq \tfrac{d-1}{2}$. Power Spherical's $\beta$ is *pinned by the
dimension*; only $\alpha=\beta+\kappa$ moves. TNBBeta's $\varepsilon$ is free, but its
base Beta stays symmetric and the asymmetry is produced by the nonlinear maps
$T_p,H_q$ instead.

Two facts fix the comparison:

* **$\mathrm{Uniform}(S^{d-1})$ is a member of both.** Power Spherical at $\kappa=0$
  gives $\mathrm{Beta}(\beta,\beta)$; TNBBeta at $(p,q,\varepsilon)=(\tfrac12,0,\beta)$
  gives the same $\mathrm{Beta}(\beta,\beta)$ (`uniform_prior_params`). Same latitude
  marginal, same construction $\Rightarrow$ the same distribution on the sphere.
* **The latitude marginal is a complete invariant.** Two axially symmetric
  distributions on $S^{d-1}$ with the *same* axis are equal iff their
  $W$-marginals are equal; and total variation distance between them equals TV
  between the $W$-marginals. (Both are pushforwards along $x\mapsto
  \boldsymbol\mu^\top x$ of the same construction, and the construction is a
  measure-isomorphism onto the axially symmetric distributions.) Lemma 2.2 below
  removes the "same axis" caveat.

## 1. A normal form for the TNBBeta density

Everything in this note runs off one rewriting. Put

$$
c \defeq \frac{1-p}{p} > 0, \qquad
\lambda \defeq 1 - c = \frac{2p-1}{p}, \qquad
R(y) \defeq (1-\lambda y)^2 - 4qc\,y(1-y).
$$

**Theorem 1.1 (normal form).** For $p\in(0,1)$, $q\in[0,1)$, $\varepsilon>0$,

$$
\boxed{\
f(y;p,q,\varepsilon)
= \frac{\big[(1-q)\,c\big]^{\varepsilon}}{B(\varepsilon,\varepsilon)}\;
y^{\varepsilon-1}(1-y)^{\varepsilon-1}\,\big(1-\lambda y\big)\,R(y)^{-\left(\varepsilon+\frac12\right)}
\ }
$$

*Proof.* Start from the closed form of
[`tnbbeta_from_beta_derivation.md`](tnbbeta_from_beta_derivation.md) §4,

$$
f = \frac{1}{B(\varepsilon,\varepsilon)}\cdot\frac{[(1-q)\gamma]^{\varepsilon}}{M^{\varepsilon+1/2}\,y(1-y)},
\qquad M = 1-4q\gamma .
$$

The denominator of $\gamma(y,p)=\dfrac{p(1-p)y(1-y)}{\big(p(1-y)+(1-p)y\big)^2}$
satisfies $p(1-y)+(1-p)y = p - (2p-1)y = p\,(1-\lambda y)$, so

$$
\gamma(y,p) = \frac{c\,y(1-y)}{(1-\lambda y)^2},
\qquad
M = 1 - \frac{4qc\,y(1-y)}{(1-\lambda y)^2} = \frac{R(y)}{(1-\lambda y)^2}.
$$

Substituting, $[(1-q)\gamma]^\varepsilon M^{-(\varepsilon+1/2)} = [(1-q)c]^\varepsilon
\big(y(1-y)\big)^{\varepsilon}(1-\lambda y)\,R^{-(\varepsilon+1/2)}$, and dividing by
$y(1-y)$ gives the boxed form. $\blacksquare$

**Lemma 1.2 (positivity and the shape of $R$).** On $[0,1]$: $1-\lambda y =
\big(p(1-y)+(1-p)y\big)/p > 0$, and $R(y) = M\,(1-\lambda y)^2 > 0$ because
$M = 1-4q\gamma \ge 1-q > 0$ (as $\gamma = u(1-u)\le\tfrac14$). Moreover
$R(0)=1$, $R(1)=(1-\lambda)^2=c^2$, and writing $R(y)=Ay^2+By+1$ with
$A=\lambda^2+4qc$, $B=-(2\lambda+4qc)$,

$$
\operatorname{disc} R = B^2-4A = 16qc(\lambda+qc-1) = -16qc^{2}(1-q) \le 0 ,
$$

using $\lambda - 1 = -c$. So for $q\in(0,1)$ the quadratic $R$ has **no real root at
all**, and for $q=0$ it is the perfect square $(1-\lambda y)^2$ whose double root
$1/\lambda$ lies outside $[0,1]$. Hence $y\mapsto R(y)^{-(\varepsilon+1/2)}$ and
$y\mapsto(1-\lambda y)$ are real-analytic and strictly positive on a neighbourhood
of $[0,1]$. $\blacksquare$

**Lemma 1.3 (reflection).** $1-Y \sim \mathrm{TNBBeta}(1-p,q,\varepsilon)$ whenever
$Y\sim\mathrm{TNBBeta}(p,q,\varepsilon)$; equivalently $f(y;p,q,\varepsilon) =
f(1-y;1-p,q,\varepsilon)$.

*Proof.* In the sampling construction, $Z\sim\mathrm{Beta}(\varepsilon,\varepsilon)$ is
symmetric about $\tfrac12$, so $S=2Z-1$ is symmetric about $0$; $H_q$ is odd, so
$T=H_q(S)$ is symmetric about $0$ and $U=\tfrac12(1+T)$ about $\tfrac12$. By Lemma 1
of the derivation note, $\operatorname{logit}Y = \operatorname{logit}U +
\operatorname{logit}p$, so $\operatorname{logit}Y$ is symmetric about
$\operatorname{logit}p$, hence $\operatorname{logit}(1-Y) = -\operatorname{logit}Y$
is symmetric about $\operatorname{logit}(1-p)$ with the same spread —
i.e. $1-Y\sim\mathrm{TNBBeta}(1-p,q,\varepsilon)$. $\blacksquare$

*Numerical check (§8.1).* Theorem 1.1 was checked against
`TNBBetaUnivariate.log_prob` at $2{,}400$ random $(p,q,\varepsilon,y)$ points spanning
$p\in(10^{-3},1-10^{-3})$, $q\in[0,1)$, $\varepsilon\in[10^{-2},2\times10^{2}]$:
**worst relative log-density error $2.7\times10^{-13}$**. The normal form also
integrates to $1.000000000000$ by quadrature at four test parameter settings, so the
constant is right, not just the shape. Lemma 1.3: worst
$|\log f(y;p)-\log f(1-y;1-p)|$ over 300 random points was $1.3\times10^{-13}$.

## 2. Task 1 — non-containment, proved

**Theorem 2.1 (TNBBeta $\cap$ Beta).** For $p\in(0,1)$, $q\in[0,1)$, $\varepsilon>0$
and $a,b>0$,

$$
\mathrm{TNBBeta}(p,q,\varepsilon) = \mathrm{Beta}(a,b)
\quad\Longleftrightarrow\quad
p=\tfrac12,\ \ q=0,\ \ a=b=\varepsilon .
$$

**In particular a TNBBeta is a Beta only when it is a *symmetric* Beta.**

*Proof.* ($\Leftarrow$) $p=\tfrac12\Rightarrow c=1,\lambda=0$; $q=0\Rightarrow
R\equiv1$; Theorem 1.1 collapses to
$y^{\varepsilon-1}(1-y)^{\varepsilon-1}/B(\varepsilon,\varepsilon)$.

($\Rightarrow$) Both densities are continuous and strictly positive on $(0,1)$, so
equality as distributions gives equality pointwise there. Dividing the $\mathrm{Beta}$
density by Theorem 1.1 and rearranging,

$$
y^{\varepsilon-a}(1-y)^{\varepsilon-b}
= K\,\frac{R(y)^{\varepsilon+1/2}}{1-\lambda y},
\qquad
K \defeq \frac{B(a,b)}{B(\varepsilon,\varepsilon)}\big[(1-q)c\big]^{\varepsilon} > 0 .
\tag{$\ast$}
$$

By Lemma 1.2 the right-hand side extends continuously and strictly positively to
all of $[0,1]$, with value $K$ at $y=0$ and $K\,c^{2\varepsilon}$ at $y=1$.

*Step 1.* Let $y\to0^+$. Then $(1-y)^{\varepsilon-b}\to1$, so the left side behaves as
$y^{\varepsilon-a}$, which converges to a finite nonzero limit iff $\varepsilon-a=0$.
Hence $a=\varepsilon$. Symmetrically, letting $y\to1^-$ gives $b=\varepsilon$.

*Step 2.* With $a=b=\varepsilon$ the left side of $(\ast)$ is identically $1$, so
$R(y)^{\varepsilon+1/2} = K^{-1}(1-\lambda y)$ on $(0,1)$ and, by continuity, on
$[0,1]$. Evaluating at $y=0$ gives $K=1$, hence

$$
R(y) = (1-\lambda y)^{\theta}, \qquad \theta \defeq \frac{1}{\varepsilon+\frac12} > 0 ,
\qquad y\in[0,1].
\tag{$\ast\ast$}
$$

*Step 3, case $\lambda=0$ (i.e. $p=\tfrac12$).* Then $(\ast\ast)$ reads $R\equiv1$,
i.e. $4qc\,y(1-y)\equiv0$, which forces $q=0$ since $c>0$. This is the claimed
conclusion.

*Step 4, case $\lambda\neq0$ — impossible.* Both sides of $(\ast\ast)$ are
real-analytic on the open interval $I=\{y: 1-\lambda y>0\}$ (Lemma 1.2 puts
$[0,1]\subset I$), so the identity extends from $[0,1]$ to all of $I$ by analytic
continuation. Substituting $v=1-\lambda y$, an affine bijection carrying $I$ onto
$(0,\infty)$, turns $R$ into a polynomial $P$ of degree $\le2$ and gives

$$
P(v) = v^{\theta} \quad\text{for all } v\in(0,\infty).
$$

$P\not\equiv0$ since $v^\theta>0$. If $P$ has degree $k\in\{0,1,2\}$ and leading
coefficient $\alpha_k\ne0$, then $P(v)/v^{\theta}\sim\alpha_k v^{k-\theta}\to1$
forces $k=\theta$ and $\alpha_k=1$; so $\theta\in\{0,1,2\}$. Now $\theta>0$ rules out
$0$, and $\theta=2$ would give $\varepsilon+\tfrac12=\tfrac12$, i.e. $\varepsilon=0$,
excluded. So $\theta=1$, $\varepsilon=\tfrac12$, and $R(y)=1-\lambda y$ identically.
Matching coefficients against $R=Ay^2+By+1$ gives $A=\lambda^2+4qc=0$; since
$q\ge0$ and $c>0$ this forces $\lambda=0$ *and* $q=0$, contradicting
$\lambda\neq0$. $\blacksquare$

**Lemma 2.2 (axis rigidity).** Fix $d\ge2$. If $\mathrm{PowerSpherical}(\boldsymbol\mu',\kappa)$
with $\kappa>0$ equals an axially symmetric distribution with axis
$\boldsymbol\mu$, then $\boldsymbol\mu'=\pm\boldsymbol\mu$.

*Proof.* By Proposition 4.1 below, the Power Spherical density has a *unique* local
maximum, at $\boldsymbol\mu'$. A density that is a function of
$w=\boldsymbol\mu^\top x$ alone has, for each $w^\star$, a constant value on the whole
orbit $\{x: \boldsymbol\mu^\top x = w^\star\}$, which is a sphere $S^{d-2}$ — a set of
cardinality $\ge2$ for every $d\ge2$ unless $w^\star=\pm1$. A singleton set of local
maxima therefore forces $w^\star=\pm1$, i.e. $\boldsymbol\mu'\in\{\boldsymbol\mu,-\boldsymbol\mu\}$. $\blacksquare$

**Corollary 2.3 (the two families are not nested, in either direction).** Fix
$d\ge2$. Then

$$
\{\mathrm{TNBBetaSpherical}(\boldsymbol\mu,p,q,\varepsilon)\}
\ \cap\
\{\mathrm{PowerSpherical}(\boldsymbol\mu',\kappa)\}
= \{\mathrm{Uniform}(S^{d-1})\}.
$$

*Proof.* $\supseteq$ is the shared uniform point of §0. For $\subseteq$: suppose the
two coincide. If $\kappa=0$ the Power Spherical member *is*
$\mathrm{Uniform}(S^{d-1})$ and we are done. If $\kappa>0$, Lemma 2.2 gives
$\boldsymbol\mu'=\pm\boldsymbol\mu$; replacing $(p,q,\varepsilon)$ by
$(1-p,q,\varepsilon)$ if necessary (Lemma 1.3) we may take
$\boldsymbol\mu'=\boldsymbol\mu$, and then the latitude marginals must agree:

$$
\mathrm{TNBBeta}(p,q,\varepsilon) = \mathrm{Beta}\!\left(\beta+\kappa,\ \beta\right),
\qquad \beta=\tfrac{d-1}2 .
$$

Theorem 2.1 forces $\beta+\kappa = \beta = \varepsilon$, i.e. $\kappa=0$ —
contradicting $\kappa>0$. $\blacksquare$

Both directions of non-containment follow immediately: $\mathrm{TNBBeta}(0.8,0,1)$ is
not in the intersection, so TNBBeta $\not\subseteq$ Power Spherical; and
$\mathrm{PowerSpherical}(\boldsymbol\mu,2)$ likewise gives Power Spherical
$\not\subseteq$ TNBBeta. The intersection is a single point, not a curve.

**Where the obstruction lives, in one sentence.** Theorem 2.1 Step 1 says the
boundary exponents must match ($a=b=\varepsilon$ — these come from the base
$\mathrm{Beta}(\varepsilon,\varepsilon)$ and nothing else can move them), and then Step
2 leaves the *whole* $p,q$ dependence — the factor
$(1-\lambda y)\,R(y)^{-(\varepsilon+1/2)}$, which is exactly the $\gamma(y,p)$ and
$[1-4q\gamma]$ terms of the closed form — with nowhere to go: a Beta density has no
room for any factor other than the two boundary powers, so that factor must be
constant, and it is constant only at $p=\tfrac12, q=0$. This is the precise form of
the plan's intuition that "pushing a symmetric Beta through a Möbius
reparameterization doesn't land back in the Beta family."

### 2.4 Numerical corroboration (the test the plan asked for)

The proof above makes the numerics redundant as evidence, but they were run first
and are worth recording as an independent check, and because they quantify *how
far* apart the families are.

**(a) Moment mismatch.** For each $(p,\varepsilon)$ at $q=0$ I computed
$\mathbb E[Y^k]$, $k=1,2,3$, by adaptive quadrature on the normal form, then solved
for the unique $\mathrm{Beta}(a^\star,b^\star)$ with the *same mean and variance* and
compared skewness. If TNBBeta's $q=0$ marginal were ever a Beta, the matched Beta
would have to be it and the skewnesses would agree.

| $p$ | $\varepsilon$ | mean | var | skew | $a^\star$ | $b^\star$ | skew$(a^\star,b^\star)$ | rel. gap |
|---|---|---|---|---|---|---|---|---|
| 0.55 | 1.0 | 0.533400 | 0.0828873 | $-0.139171$ | 1.0682 | 0.9345 | $-0.115935$ | 16.7% |
| 0.60 | 1.0 | 0.567209 | 0.0815297 | $-0.282090$ | 1.1406 | 0.8703 | $-0.234738$ | 16.8% |
| 0.70 | 1.0 | 0.637922 | 0.0757815 | $-0.597849$ | 1.3064 | 0.7415 | $-0.495081$ | 17.2% |
| 0.80 | 1.0 | 0.717203 | 0.0648272 | $-1.008756$ | 1.5267 | 0.6020 | $-0.826487$ | 18.1% |
| 0.90 | 1.0 | 0.816015 | 0.0451535 | $-1.723452$ | 1.8972 | 0.4278 | $-1.375430$ | 20.2% |
| 0.70 | 0.5 | 0.604356 | 0.1195549 | $-0.452715$ | 0.6044 | 0.3956 | $-0.402414$ | 11.1% |
| 0.70 | 2.0 | 0.663868 | 0.0424860 | $-0.664614$ | 2.8229 | 1.4293 | $-0.508620$ | 23.5% |
| 0.70 | 4.5 | 0.682487 | 0.0196737 | $-0.608468$ | 6.8348 | 3.1798 | $-0.433149$ | 28.8% |
| 0.70 | 9.5 | 0.691410 | 0.0093577 | $-0.484348$ | 15.0733 | 6.7275 | $-0.332545$ | 31.3% |
| 0.70 | 20.0 | 0.695856 | 0.0044338 | $-0.357451$ | 32.5200 | 14.2138 | $-0.241424$ | 32.5% |
| **0.50** | **2.0** | 0.500000 | 0.0500000 | $0.000000$ | 2.0000 | 2.0000 | $0.000000$ | $-8\times10^{-17}$ |

The skewness gap is $11$–$33\%$ of the skewness itself and, notably, does **not**
shrink as $\varepsilon$ grows — it grows. The last row is the control: at $p=\tfrac12$
the gap is at machine precision, as Theorem 2.1 requires.

**(b) Distance to the whole Power Spherical family.** Minimising total variation
over $\kappa\ge0$ *and* over both axis orientations $\boldsymbol\mu'\in\{\pm\boldsymbol\mu\}$
(which by Lemma 2.2 is the only orientation that could ever achieve equality),
for TNBBeta at the dimension-matched $\varepsilon=\tfrac{d-1}2$ and $q=0$:

| $d$ | $p$ | $\kappa^\star$ | $\min$ TV |
|---|---|---|---|
| 3 | **0.50** | 0.0000 | $1\times10^{-6}$ *(control)* |
| 3 | 0.55 | 0.1436 | 0.021097 |
| 3 | 0.60 | 0.3257 | 0.041384 |
| 3 | 0.70 | 0.8921 | 0.079932 |
| 3 | 0.80 | 2.0863 | 0.115996 |
| 3 | 0.90 | 5.8225 | 0.149393 |
| 3 | 0.95 | 13.4090 | 0.164977 |
| 5 | 0.80 | 5.0718 | 0.115701 |
| 10 | 0.80 | 12.5710 | 0.114551 |
| 20 | 0.80 | 27.5718 | 0.113885 |
| 10 | **0.50** | 0.0000 | $1\times10^{-6}$ *(control)* |

The residual TV is $\sim0.11$–$0.17$ at moderate $p$ and is **essentially
dimension-independent** ($0.1160,0.1157,0.1146,0.1139$ at $p=0.8$ for
$d=3,5,10,20$) — the gap is not a low-dimensional artifact that washes out as $d$
grows. The $p=\tfrac12$ controls land at quadrature noise.

## 3. Task 2 — the log-concavity lemma, and the $d=2$ edge case

**Lemma 3.1 (log-concavity of the Beta).** For $a,b>0$ and $y\in(0,1)$,

$$
\big(\log f_{\mathrm{Beta}(a,b)}\big)''(y) = -\frac{a-1}{y^2} - \frac{b-1}{(1-y)^2}.
$$

Hence if $a\ge1$ and $b\ge1$ the log-density is concave, and if additionally $a>1$
or $b>1$ it is *strictly* concave on $(0,1)$.

*Proof.* $\log f = (a-1)\log y + (b-1)\log(1-y) - \log B(a,b)$; differentiate twice.
Both terms of the result are $\le0$ when $a,b\ge1$, and at least one is $<0$ under
the strictness hypothesis. $\blacksquare$

**Corollary 3.2 (no bimodality for $a,b\ge1$).** A log-concave density on $(0,1)$ has
convex superlevel sets $\{f\ge t\}$, i.e. intervals; so it is non-decreasing then
non-increasing, has no interior local minimum, and its set of maximisers is a single
interval. In particular it cannot have two distinct local maxima.

**Corollary 3.3 (Power Spherical's latitude marginal is unimodal, $d\ge3$).** For
$d\ge3$ the Power Spherical latitude marginal in $y$-coordinates is
$\mathrm{Beta}(\beta+\kappa,\beta)$ with $\beta=\tfrac{d-1}2\ge1$ and
$\beta+\kappa\ge1$. By Corollary 3.2 it is unimodal for **every** $\kappa\ge0$ —
never bimodal, never U-shaped.

*Numerical check:* over $7$ dimensions $d\in\{3,4,5,10,20,40,128\}$ $\times$ $9$
concentrations $\kappa\in\{0,0.1,0.5,1,3,10,50,200,1000\}$, a $200{,}001$-point grid
scan found **0 parameter settings with more than one mode** (§8.4).

**Remark 3.4 (the $d=2$ edge case genuinely breaks this argument).** At $d=2$,
$\beta=\tfrac12$ and the marginal is $\mathrm{Beta}(\tfrac12+\kappa,\tfrac12)$. Then
$b=\tfrac12<1$ *always*, and $a=\tfrac12+\kappa<1$ whenever $\kappa<\tfrac12$. Lemma
3.1's hypothesis fails, and so does its conclusion: numerically (§8.4),
$\mathrm{Beta}(\tfrac12+\kappa,\tfrac12)$ for $\kappa\in\{0,0.1,0.4,0.49\}$ has
**zero interior maxima and a divergence at both endpoints** — a genuinely U-shaped,
two-mode latitude marginal. At $\kappa=0$ this is the arcsine law, i.e.
$\mathrm{Uniform}(S^1)$ itself, which makes the point sharp: the apparent bimodality
is produced entirely by the Jacobian of the coordinate $w=\cos\theta$, whose density
factor $(1-w^2)^{-1/2}$ blows up at both ends.

So at $d=2$, "TNBBeta's latitude marginal can be bimodal and Power Spherical's
cannot" is **false**. The plan's provable-gap claim therefore cannot be stated at the
level of the latitude marginal if it is to hold in all dimensions. §4–§6 state it on
the sphere instead, where it is both dimension-free and geometrically meaningful.
(This is not a technicality to be waved at: $d=2$ with `latent_dim = d+1 = 3` is a row
of the MNIST sweep.)

## 4. Task 2 — Power Spherical is a monotone cap, in every dimension

**Proposition 4.1.** The Power Spherical density with respect to the uniform measure
on $S^{d-1}$ is

$$
h_{\mathrm{PS}}(x) = C(\kappa,d)\,\big(1+\boldsymbol\mu^\top x\big)^{\kappa},
\qquad C(\kappa,d)>0 ,
$$

a strictly increasing function of $w=\boldsymbol\mu^\top x$ for $\kappa>0$, and constant
for $\kappa=0$. Consequently, for every $d\ge2$ and every $\kappa>0$:

1. the set of local maxima of $h_{\mathrm{PS}}$ is exactly $\{\boldsymbol\mu\}$;
2. $h_{\mathrm{PS}}$ is strictly decreasing in geodesic distance from $\boldsymbol\mu$;
3. $h_{\mathrm{PS}}$ has no ring mode, no antipodal mode, and no interior local
   minimum (its unique minimum is at $-\boldsymbol\mu$, where it is $0$).

*Proof.* The density is Definition 2 / Theorem 13 of De Cao & Aziz (2020), as
implemented in `power_spherical.py`. For (1): if $x_0\neq\boldsymbol\mu$ then
$w_0=\boldsymbol\mu^\top x_0<1$, and moving $x_0$ along the minimising geodesic toward
$\boldsymbol\mu$ strictly increases $w$ and hence strictly increases
$h_{\mathrm{PS}}$, so $x_0$ is not a local maximum; $\boldsymbol\mu$ is the unique
maximiser of $w$ and hence of $h_{\mathrm{PS}}$. (2) and (3) are restatements, using
$\cos(\text{geodesic distance}) = w$. $\blacksquare$

This is the airtight version of the plan's claim: it needs no log-concavity, no
case analysis in $d$, and no appeal to the coordinate $w$ — it is just "the density is
a monotone function of the cosine." **Power Spherical is strictly unimodal on the
sphere for every $\kappa$ and every $d\ge2$.** Numerically (§8.5), the unnormalized
log-density is strictly increasing in $w$ at every $\kappa$ tested, including
$\kappa=0.2$ and $0.49$ at $d=2$ where the *marginal* is U-shaped.

## 5. Task 2 — TNBBetaSpherical's modality, exactly

Let $h$ be the TNBBetaSpherical density with respect to the uniform measure on
$S^{d-1}$. From `tnbbeta_spherical.py`'s `log_prob` and Theorem 1.1, in the
coordinate $y=(1+w)/2$,

$$
h(y) \ \propto\ y^{m}(1-y)^{m}\,\big(1-\lambda y\big)\,R(y)^{-\left(\varepsilon+\frac12\right)},
\qquad
\boxed{\,m \defeq \varepsilon - \tfrac{d-1}{2}\,}
\tag{5.1}
$$

because the $(1-w^2)^{-(d-3)/2}$ Jacobian contributes $\big(y(1-y)\big)^{-(d-3)/2}$,
shifting the exponent $\varepsilon-1$ to $\varepsilon-1-\tfrac{d-3}2 = m$. (The
*latitude marginal* itself is the case $m=\varepsilon-1$, equivalently (5.1) at $d=3$.)
**$m$ is the parameter that controls modality** — it measures $\varepsilon$ against the
uniform-prior value $\tfrac{d-1}{2}$, and $m=0$ is exactly
`uniform_prior_params`'s $\varepsilon$.

Differentiating (5.1),

$$
(\log h)'(y) = \frac{N(y)}{y(1-y)(1-\lambda y)R(y)},
\qquad
\begin{aligned}
N(y) \defeq\ & m(1-2y)(1-\lambda y)R(y) \\
 &- \lambda\,y(1-y)R(y) \\
 &- \big(\varepsilon+\tfrac12\big)R'(y)\,y(1-y)(1-\lambda y),
\end{aligned}
\tag{5.2}
$$

and by Lemma 1.2 the denominator is **strictly positive on $(0,1)$**, so
$\operatorname{sign}(\log h)' = \operatorname{sign}N$ there. $N$ is a polynomial of
degree $\le4$.

**Theorem 5.1 (pole behaviour: a clean trichotomy in $m$).** $N(0)=m$ and
$N(1)=-m\,c^{3}$. Hence $(\log h)'$ has sign $\operatorname{sign}(m)$ near $y=0^+$
and $\operatorname{sign}(-m)$ near $y=1^-$, and:

* **$m>0$** ($\varepsilon>\tfrac{d-1}2$): $h\to0$ at both poles, and $h$ increases out
  of $-\boldsymbol\mu$ and decreases into $\boldsymbol\mu$. Neither pole is a mode.
* **$m=0$** ($\varepsilon=\tfrac{d-1}2$): $h$ is finite and strictly positive at both
  poles. Then $N(y) = -y(1-y)\,Q(y)$ with
  $Q(y) \defeq \lambda R(y) + (\varepsilon+\tfrac12)R'(y)(1-\lambda y)$ a quadratic,
  so $(\log h)' = -Q(y)/[(1-\lambda y)R(y)]$, and $-\boldsymbol\mu$ is a (one-sided)
  mode iff $Q(0)>0$ while $\boldsymbol\mu$ is one iff $Q(1)<0$. **At most one of these
  can hold.**
* **$m<0$** ($\varepsilon<\tfrac{d-1}2$): $h\to+\infty$ at **both** poles, $h$ is
  strictly decreasing on some $(0,\delta)$ and strictly increasing on some
  $(1-\delta,1)$, and (being continuous and integrable, hence finite somewhere) it
  attains an interior minimum. **$h$ is at least bimodal, with modes at the
  antipodal pair $\{\boldsymbol\mu,-\boldsymbol\mu\}$.**

*Proof.* $N(0)$: at $y=0$ the last two terms of (5.2) carry a factor $y$ and vanish,
and the first is $m\cdot1\cdot1\cdot R(0)=m$. $N(1)$: the last two terms carry
$(1-y)$ and vanish, and the first is $m(1-2)(1-\lambda)R(1) = -m\,c\cdot c^2$. The
sign statements follow from positivity of the denominator in (5.2). The $m=0$
factorisation holds because, with $m=0$, both surviving terms of (5.2) contain
$y(1-y)$. For the incompatibility, substitute $R(0)=1$, $R'(0)=B=-(2\lambda+4qc)$,
$R(1)=c^2$, $R'(1)=2A+B=2c(2q-\lambda)$ and $1-\lambda=c$:

$$
Q(0) = \lambda - (2\varepsilon+1)\big(\lambda+2qc\big),
\qquad
Q(1) = c^{2}\Big[\lambda + (2\varepsilon+1)\big(2q-\lambda\big)\Big].
$$

Then, using $1-(2\varepsilon+1) = -2\varepsilon$ and $\varepsilon>0$, $c>0$, $q\ge0$:

$$
Q(0)>0 \iff -2\varepsilon\lambda > 2qc(2\varepsilon+1) \iff \lambda < -\frac{qc(2\varepsilon+1)}{\varepsilon} \le 0,
$$
$$
Q(1)<0 \iff -2\varepsilon\lambda + 2q(2\varepsilon+1) < 0 \iff \lambda > \frac{q(2\varepsilon+1)}{\varepsilon} \ge 0 .
$$

The first requires $\lambda<0$, the second $\lambda>0$ (at $q=0$ both bounds are $0$
and the inequalities are still strict). They cannot hold together. $\blacksquare$

**Proposition 5.2 (interior structure).** $N$ has degree $\le4$, so $h$ has at most
$4$ interior critical points and hence at most $2$ interior local maxima; at $m=0$,
the deflated quadratic $Q$ allows at most $2$ interior critical points and hence at
most $1$ interior local maximum.

**Numerical (§8.3): at most *one* interior local maximum, ever.** Computing the
interior roots of $N$ exactly (companion-matrix root-finding on the degree-$\le4$
coefficient vector, with the $m=0$ boundary roots deflated as in Theorem 5.1) over
$401{+}4$ values of $p$ spanning $[10^{-5},1-10^{-5}]$, $12$ values of $q$ spanning
$[0,1-10^{-8}]$ and $16$ values of $\varepsilon$ spanning $[0.01,200]$ —
$77{,}760$ triples per dimension, repeated for the latitude marginal and for
$d\in\{2,3,5,10,20,40\}$, i.e. $544{,}320$ parameter settings — the number of
interior local maxima was **never more than 1**, and the number of total modes
(interior $+$ pole) was never more than $3$.

> **Honestly labelled:** Proposition 5.2's bound of $2$ is a proof; "never more than
> $1$" is a numerical finding over the sweep above, not a theorem. Nothing in §6
> depends on it — it is reported because it is what rules out the "two interior
> maxima" route and forces the correct mechanism (pole divergence) into view.

**Corollary 5.3 (where multimodality lives).** Combining Theorem 5.1 and the sweep:
TNBBetaSpherical is multimodal **iff $m<0$**, i.e. iff
$\varepsilon<\tfrac{d-1}{2}$ — bimodal at $\{\boldsymbol\mu,-\boldsymbol\mu\}$ when $N$ has no
interior maximum, and **trimodal** ($-\boldsymbol\mu$, a ring at some interior
$w^\star$, and $\boldsymbol\mu$) when it has one. For $m\ge0$ it is unimodal: at
$m>0$ by the sweep, and at $m=0$ by Theorem 5.1 plus the sweep (no $m\ge0$ setting in
$544{,}320$ had $\ge2$ modes).

**Why "the density diverges at the poles" is not a loophole.** $S^{d-1}$ has no
boundary; $\boldsymbol\mu$ and $-\boldsymbol\mu$ are ordinary interior points of a compact
manifold. $y\in\{0,1\}$ are endpoints only of the *coordinate* $w$. So $m<0$ is a
genuine pair of density spikes at two antipodal points, not a boundary effect, and
$h$ remains a bona fide probability density (integrability is automatic:
$\int_{S^{d-1}}h\,d\sigma = \int_0^1 f_Y = 1$, verified numerically in §8.6 to
$10^{-8}$). The mass accounting in §6 confirms the lobes are substantive, not
measure-zero spikes.

## 6. The expressivity gap, with certified examples

**Theorem 6.1 (strict dominance within the axially symmetric families).** For every
$d\ge2$:

1. Every $\mathrm{PowerSpherical}(\boldsymbol\mu,\kappa)$ is unimodal on $S^{d-1}$,
   with density monotone decreasing in geodesic distance from $\boldsymbol\mu$
   (Proposition 4.1).
2. For every $\varepsilon\in\big(0,\tfrac{d-1}2\big)$ and every $p\in(0,1)$,
   $q\in[0,1)$, the distribution $\mathrm{TNBBetaSpherical}(\boldsymbol\mu,p,q,\varepsilon)$
   is bimodal with modes at $\{\boldsymbol\mu,-\boldsymbol\mu\}$ (Theorem 5.1, $m<0$), and for
   suitable $(p,q)$ trimodal.
3. Hence no Power Spherical distribution equals any such TNBBetaSpherical, and the
   two families differ not merely in parameterisation but in the *shape classes* they
   can realise. Combined with Corollary 2.3, the containment fails in both
   directions and the intersection is the single point
   $\mathrm{Uniform}(S^{d-1})$.

Note that this statement is uniform in $d$ and in particular survives $d=2$, where
the latitude-marginal version of the claim fails (Remark 3.4): at $d=2$,
$\varepsilon<\tfrac12$ gives $m<0$ and a TNBBeta density diverging at both poles of
$S^1$, while $(1+\cos\theta)^{\kappa}$ is unimodal in $\theta$ for every $\kappa>0$
(§8.5). This is also why $\varepsilon$ is best read as "concentration relative to
$\tfrac{d-1}{2}$" rather than as an absolute number.

### 6.1 Certified example A — antipodal bimodal (the arcsine case)

$(p,q,\varepsilon) = (0.5,\,0,\,0.5)$, $d=3$. Here $m=-0.5$, and $\mathrm{TNBBeta}(\tfrac12,0,\tfrac12)
= \mathrm{Beta}(\tfrac12,\tfrac12)$ exactly (Theorem 2.1's $\Leftarrow$ direction).

* Exact interior critical points of $\log h$: a **single minimum** at $y=0.5$
  ($w=0$). No interior maximum.
* $4{,}000{,}001$-point grid: $\log f$ increasing into $y=0$ (**true**) and into
  $y=1$ (**true**), interior maxima **0**. So the modes are the two poles.
* Mass: $0.14357$ in $w\in[-1,-0.9]$, $0.14357$ in $w\in[0.9,1]$, $0.06377$ in
  $w\in[-0.1,0.1]$ — a $2.25\times$ pole-to-equator ratio over equal-width bands.
* `TNBBetaSpherical.log_prob` at $d=3$: $0.1249$ at $w=-0.999$, $-2.9826$ at $w=0$,
  $0.1249$ at $w=+0.999$. Agrees with (5.1) to $8.9\times10^{-16}$ (constant offset).
* Minimum TV to the entire Power Spherical family (over $\kappa\ge0$ and both
  orientations): $\mathbf{0.2105}$.

An asymmetric variant, $(p,q,\varepsilon)=(0.8,0,0.6)$, $d=3$ ($m=-0.4$): one
interior minimum at $w=-0.5099$, both poles divergent, log-density $-1.2036$ at
$w=-0.999$ vs. $+0.4577$ at $w=+0.999$ — a *large* cap at $\boldsymbol\mu$ plus a
*small* antipodal spike. Minimum TV to the Power Spherical family: $\mathbf{0.2550}$.

### 6.2 Certified example B — trimodal (pole $+$ ring $+$ antipode)

$(p,q,\varepsilon) = (0.85,\,0.95,\,0.3)$, $d=3$ (so $m=-0.7$).

Exact critical points of $\log h$ (roots of $N$):

| $y$ | $w=2y-1$ | type |
|---|---|---|
| $0.2701515560$ | $-0.4596968881$ | minimum |
| $0.8620598325$ | $+0.7241196650$ | **maximum** (a ring at $\approx 43.6^\circ$ from $\boldsymbol\mu$) |
| $0.9582474094$ | $+0.9164948189$ | minimum |

plus the two divergent pole modes at $w=\pm1$: **three modes.** Checks:

* $4{,}000{,}001$-point grid on $\log f$: interior maxima at $y=0.86206$ (matching the
  exact root), left spike **true**, right spike **true**.
* `TNBBetaUnivariate.log_prob` vs. the normal form at those critical points:
  $|{\cdot}|\le1.3\times10^{-15}$.
* Lobe masses by quadrature, and by $4\times10^{6}$ Monte Carlo draws from
  `TNBBetaUnivariate.rsample`:

  | lobe | $w$ range | quadrature | Monte Carlo |
  |---|---|---|---|
  | antipodal | $[-1,\,-0.4597)$ | $0.10343$ | $0.10335$ |
  | ring | $[-0.4597,\,0.9165)$ | $0.72334$ | $0.72318$ |
  | polar | $[0.9165,\,1]$ | $0.17323$ | $0.17347$ |

  All three lobes carry $\ge10\%$ of the mass. Quadrature and MC agree to
  $\le2.4\times10^{-4}$, so the shape is real and the sampler reproduces it.
* `TNBBetaSpherical.log_prob` at $d=3$: $-0.4220$ at $w=-0.999$, $-3.9246$ at $w=0$,
  $+0.6255$ at $w=+0.999$; agrees with (5.1) to $2.2\times10^{-15}$.
* $\int_{S^{2}} h\,d\sigma = 0.99999999$ (§8.6) — it is a normalised density.
* Minimum TV to the entire Power Spherical family: $\mathbf{0.2474}$ ($d=3$);
  $\mathbf{0.2188}$ for the $d=10$ analogue $(0.85,0.95,1.35)$.

### 6.3 The gap is quantitatively large and dimension-stable

| case | $d$ | $(p,q,\varepsilon)$ | $\kappa^\star$ | $\min$ TV to Power Spherical |
|---|---|---|---|---|
| uniform point *(control)* | 3 | $(0.5,0,1.0)$ | $0.0000$ | $0.000002$ |
| uniform point *(control)* | 10 | $(0.5,0,4.5)$ | $0.0000$ | $0.000001$ |
| $q=0$ cap, $p\neq\tfrac12$ | 3 | $(0.8,0,1.0)$ | $2.0863$ | $0.115996$ |
| $q=0$ cap, $p\neq\tfrac12$ | 10 | $(0.8,0,4.5)$ | $12.5709$ | $0.114551$ |
| antipodal bimodal | 3 | $(0.5,0,0.5)$ | $0.0000$ | $0.210514$ |
| antipodal bimodal | 3 | $(0.8,0,0.6)$ | $1.2777$ | $0.255002$ |
| antipodal bimodal | 10 | $(0.8,0,2.0)$ | $11.2389$ | $0.313329$ |
| trimodal | 3 | $(0.85,0.95,0.3)$ | $3.9011$ | $0.247356$ |
| trimodal | 10 | $(0.85,0.95,1.35)$ | $20.6970$ | $0.218803$ |

## 7. Corrections to `week_2_plan.md`

Two of the plan's working assumptions do not survive contact with the algebra. Both
should be fixed before any of this reaches the TeX writeup.

**(i) $q$ is not the bimodality knob — $\varepsilon$ is.** The plan says "TNBBeta's $q$
parameter can produce a U-shaped/bimodal marginal directly (visible in the high-$q$
cells of `pq_ablation.png`)." This is false as stated. `pq_ablation.py` fixes
$\varepsilon=1.0$ and renders $S^2$ ($d=3$), so $m=\varepsilon-\tfrac{d-1}2=0$ for
every cell — and by Theorem 5.1's $m=0$ case, bimodality is *provably impossible*
there. Counting modes exactly over that grid:

```
pq_ablation.py grid (epsilon = 1.0, S^2): total modes per cell
        q=       0     0.1     0.3     0.5     0.7     0.9
  p=0.05        1       1       1       1       1       1
  p=0.20        1       1       1       1       1       1
  p=0.50        0       1       1       1       1       1      <- (0.5, 0, 1) is Uniform(S^2)
  p=0.80        1       1       1       1       1       1
  p=0.95        1       1       1       1       1       1
```

All 30 cells are unimodal (the $0$ is the uniform cell, whose density is constant and
so has no strict local maximum at all). $q$'s actual role is to sharpen or relocate a
*single* interior mode.

The grid that does show it is the sibling sweep, `p_epsilon_ablation.py`
($q=0$, $\varepsilon\in\{0.5,0.8,1.0,1.5,3.0,10.0\}$, $S^2$):

```
p_epsilon_ablation.py grid (q = 0.0, S^2): total modes per cell
        eps=     0.5     0.8       1     1.5       3      10
  p=0.05        2       2       1       1       1       1
  p=0.20        2       2       1       1       1       1
  p=0.50        2       2       0       1       1       1
  p=0.80        2       2       1       1       1       1
  p=0.95        2       2       1       1       1       1
```

The entire $\varepsilon\in\{0.5,0.8\}$ block ($m=-0.5$ and $m=-0.2$) is the bimodal
region: each cell has both poles divergent plus exactly one interior minimum, e.g.
$(p,\varepsilon)=(0.8,0.5)\Rightarrow$ minimum at $y=0.297964$,
$(0.8,0.8)\Rightarrow y=0.128422$. **Cite `p_epsilon_ablation.png`, not
`pq_ablation.png`, for the expressivity figure.**

A corollary worth carrying forward: the *prior* used throughout the project,
$\varepsilon=\tfrac{d-1}2$ (`uniform_prior_params`), sits exactly on the $m=0$
boundary. Any posterior head that keeps $\varepsilon$ at or above the prior's value
can never be multimodal, regardless of what $p$ and $q$ do. If the com-DBLP
experiment is meant to exercise the expressivity gap, the encoder must be free to
push $\varepsilon$ *below* $\tfrac{d-1}2$, and that is the quantity the diagnostic
should log.

**(ii) The $d=2$ edge case is load-bearing, not a footnote.** Remark 3.4: at $d=2$
the Power Spherical latitude marginal $\mathrm{Beta}(\tfrac12+\kappa,\tfrac12)$ is
itself U-shaped for $\kappa<\tfrac12$, so the latitude-marginal formulation of the gap
is simply false at $d=2$ — and $d=2$ is in the sweep. State the gap on the sphere
(Proposition 4.1 / Theorem 6.1), where it holds for all $d\ge2$ with no case
analysis.

**(iii) One caveat the plan already has right, restated.** This is an *expressivity*
result, not a trainability one. $m<0$ makes the density unbounded at two points,
which is legitimate measure-theoretically but hands the ELBO unbounded log-density
terms; and week 1's ablations already showed gradient descent on MNIST never goes
anywhere near $\varepsilon<\tfrac{d-1}2$. Theorem 6.1 says the capability exists; it
says nothing about whether any optimiser will find it.

## 8. Methodology and raw outputs

All computations in double precision. Densities evaluated via Theorem 1.1's normal
form, cross-checked against `TNBBetaUnivariate.log_prob` and
`TNBBetaSpherical.log_prob` at every certified example. Integrals by
`scipy.integrate.quad` (adaptive, `limit` 600–800, `epsabs=1e-13`); TV minimisation
by `scipy.optimize.minimize_scalar(method="bounded")` over $\kappa\in[0,500]$.
Scripts lived in `/tmp/tnb_ps/` (scratch, not committed): `verify.py` (normal form,
moments, TV), `modality2.py` (exact modality census), `final.py` (certified
examples), `gap.py` (quantitative gap), `grids.py` (ablation-grid mode counts). The
two load-bearing routines are reproduced below.

**8.1 Normal form (the object everything else is computed from).**

```python
def tnb_logpdf(y, p, q, eps):
    """Theorem 1.1, including the exact normalizing constant."""
    y = np.asarray(y, dtype=float)
    c = (1.0 - p) / p
    lam = 1.0 - c
    one_lam = 1.0 - lam * y
    R = one_lam**2 - 4.0 * q * c * y * (1.0 - y)
    return (
        eps * np.log((1.0 - q) * c)
        - betaln(eps, eps)
        + (eps - 1.0) * np.log(y)
        + (eps - 1.0) * np.log1p(-y)
        + np.log(one_lam)
        - (eps + 0.5) * np.log(R)
    )
```

Against `TNBBetaUnivariate.log_prob`: worst relative error $3.4\times10^{-14}$ over
400 random points, $2.7\times10^{-13}$ over 2000 random points with
$\varepsilon$ up to $200$. $\int_0^1 f = 1.000000000000$ at
$(p,q,\varepsilon)\in\{(0.5,0,1.5),(0.2,0,2),(0.8,0.6,3),(0.3,0.9,0.7)\}$.

**8.2 Exact critical points (equation 5.2), with the $m=0$ deflation.**

```python
def numerator(m, eps, p, q):
    """N(y) from eq. (5.2); (log h)'(y) = N(y) / [y(1-y)(1-lam y)R(y)], denom > 0."""
    c = (1.0 - p) / p
    lam = 1.0 - c
    A = lam**2 + 4.0 * q * c
    B = -(2.0 * lam + 4.0 * q * c)
    R = np.array([1.0, B, A])
    Rp = np.array([B, 2.0 * A])
    one_lam = np.array([1.0, -lam])
    y1y = polymul([0.0, 1.0], [1.0, -1.0])
    t1 = m * polymul(polymul([1.0, -2.0], one_lam), R)
    t2 = -lam * polymul(y1y, R)
    t3 = -(eps + 0.5) * polymul(polymul(Rp, y1y), one_lam)
    return pad_and_sum(t1, t2, t3), lam, A, B, c


# at m == 0, N = -y(1-y)Q(y) exactly, so y=0 and y=1 are ALGEBRAIC roots of N that
# are NOT critical points of log h.  Without deflating them a naive polyroots call
# returns them at 1 - 1e-12 and reports spurious interior minima.
#   Q(y) = lam*R(y) + (eps+1/2)*R'(y)*(1 - lam y)   ->   (log h)' = -Q/[(1-lam y)R]
```

That deflation bug is worth flagging: an un-deflated first pass reported 71 spurious
"U-shaped" $(p,q)$ pairs at $\varepsilon=1$, every one with its claimed interior
minimum at $y=1-10^{-12}$. Theorem 5.1's $m=0$ case then proved they could not exist,
which is what exposed the bug — the algebra caught the numerics, not the other way
round.

**8.3 Modality census (Proposition 5.2's numerical half).** Grid:
$p\in\{0.0005,\ldots,0.9995\}$ (401 points) $\cup\{10^{-5},10^{-4},1-10^{-4},1-10^{-5}\}$;
$q\in\{0,0.05,0.2,0.4,0.6,0.8,0.9,0.95,0.99,0.999,1-10^{-5},1-10^{-8}\}$;
$\varepsilon\in\{0.01,0.05,0.2,0.5,0.8,0.95,1,1.05,1.3,2,3,4.5,9.5,19.5,50,200\}$.

```
census [w-marginal: m = eps - 1]            77760 triples; max interior maxima 1; max total modes 3
census [S^(d-1), d = 2: m = eps - 0.5]      77760 triples; max interior maxima 1; max total modes 3
census [S^(d-1), d = 3: m = eps - 1.0]      77760 triples; max interior maxima 1; max total modes 3
census [S^(d-1), d = 5: m = eps - 2.0]      77760 triples; max interior maxima 1; max total modes 3
census [S^(d-1), d = 10: m = eps - 4.5]     77760 triples; max interior maxima 1; max total modes 3
census [S^(d-1), d = 20: m = eps - 9.5]     77760 triples; max interior maxima 1; max total modes 3
census [S^(d-1), d = 40: m = eps - 19.5]    77760 triples; max interior maxima 1; max total modes 3
  -> in every census: triples with m >= 0 (bounded density) and >= 2 modes:  0
  -> in every census: every triple with 2 pole modes has m < 0
```

Independent check of Theorem 5.1's $m=0$ incompatibility over $840{,}420$ triples
($p$: 2001 points in $(0,1)$, $q$: 60 points in $[0,1)$, $\varepsilon$: 7 values from
$0.01$ to $100$): **0 triples with both poles as modes.**

**8.4 Power Spherical unimodality.** $7$ dims $\times$ $9$ $\kappa$ values,
$200{,}001$-point grids: $0$ settings with more than one mode. $d=2$ detail:

```
  kappa=0.0 : a=0.50 b=0.50  interior maxima=0  left-spike=True  right-spike=True  log-concave=NO
  kappa=0.1 : a=0.60 b=0.50  interior maxima=0  left-spike=True  right-spike=True  log-concave=NO
  kappa=0.4 : a=0.90 b=0.50  interior maxima=0  left-spike=True  right-spike=True  log-concave=NO
  kappa=0.5 : a=1.00 b=0.50  interior maxima=0  left-spike=False right-spike=True  log-concave=NO
  kappa=1.0 : a=1.50 b=0.50  interior maxima=0  left-spike=False right-spike=True  log-concave=NO
  kappa=3.0 : a=3.50 b=0.50  interior maxima=0  left-spike=False right-spike=True  log-concave=NO
  d=2 angular density (1+cos(theta))^kappa:  interior maxima in theta = 1 for every
  kappa in {0.1, 0.5, 1.0, 3.0, 10.0}
```

**8.5 Power Spherical monotonicity in $w$** (Proposition 4.1), unnormalized
$\log(1+w)^{\kappa}$ at $w\in\{-0.999,-0.9,-0.5,0,0.5,0.9,0.999\}$:

```
  kappa= 0.5: [-3.4539, -1.1513, -0.3466, 0.0, 0.2027, 0.3209, 0.3463]   strictly increasing: True
  kappa= 2.0: [-13.8155, -4.6052, -1.3863, 0.0, 0.8109, 1.2837, 1.3853]  strictly increasing: True
  kappa=10.0: [-69.0776, -23.0259, -6.9315, 0.0, 4.0547, 6.4185, 6.9265] strictly increasing: True
  at d = 2, kappa in {0.2, 0.49, 1.0}: strictly increasing in w: True, True, True
```

**8.6 Normalisation of the certified examples** (quadrature over the latitude
measure, using `TNBBetaSpherical.log_prob` itself as the integrand):

```
  p=0.5,  q=0.0,  eps=0.5, d=3 : int f_Y dy = 1.0000000000 ; int_{S^2}  h dsigma = 1.00000000
  p=0.85, q=0.95, eps=0.3, d=3 : int f_Y dy = 1.0000000000 ; int_{S^2}  h dsigma = 0.99999999
  p=0.8,  q=0.0,  eps=2.0, d=10: int f_Y dy = 1.0000000000 ; int_{S^9}  h dsigma = 1.00000000
```

## 9. Summary of what is and is not proved

| claim | status |
|---|---|
| Theorem 1.1, normal form $f\propto y^{\varepsilon-1}(1-y)^{\varepsilon-1}(1-\lambda y)R^{-(\varepsilon+1/2)}$ | **proof** (+ checked to $2.7\times10^{-13}$) |
| Theorem 2.1: a TNBBeta is a Beta iff $p=\tfrac12,q=0$, giving $\mathrm{Beta}(\varepsilon,\varepsilon)$ | **proof** |
| Corollary 2.3: the two spherical families intersect exactly in $\mathrm{Uniform}(S^{d-1})$; non-nested both ways | **proof** |
| Lemma 3.1/3.2/3.3: Beta log-concave for $a,b\ge1$; Power Spherical marginal unimodal for $d\ge3$ | **proof** |
| Remark 3.4: the $d\ge3$ restriction is necessary — at $d=2$ the Power Spherical *marginal* is U-shaped for $\kappa<\tfrac12$ | **proof** (+ numerics) |
| Proposition 4.1: Power Spherical is strictly unimodal **on the sphere**, all $d\ge2$, all $\kappa$ | **proof** |
| Theorem 5.1: pole-mode trichotomy in $m=\varepsilon-\tfrac{d-1}2$; at $m=0$ at most one pole mode | **proof** |
| Proposition 5.2: at most 2 interior local maxima (1 when $m=0$) | **proof** |
| "at most **1** interior local maximum, ever" | **numerical**, $544{,}320$ settings |
| Theorem 6.1: $\varepsilon<\tfrac{d-1}2\Rightarrow$ antipodal bimodal $\Rightarrow$ unreachable by Power Spherical, every $d\ge2$ | **proof** |
| trimodality at $(0.85,0.95,0.3)$, $d=3$ | **numerical**, but certified: exact roots of $N$, grid, package `log_prob`, and $4\times10^6$ MC draws all agree |
| $\min$ TV to the Power Spherical family $\approx0.11$–$0.31$, dimension-stable | **numerical** (quadrature + 1-D optimisation) |
| `pq_ablation.py`'s grid is entirely unimodal; `p_epsilon_ablation.py`'s $\varepsilon\in\{0.5,0.8\}$ block is the bimodal one | **proof** ($m=0$ case of Theorem 5.1) + exact mode count |
