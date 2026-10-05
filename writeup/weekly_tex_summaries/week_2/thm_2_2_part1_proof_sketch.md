# Theorem 2.2, Part 1: weak convergence to $\delta_{\boldsymbol\mu}$, via tightness

Scratch writeup of the tightness route worked out in chat — not part of the compiled
week 2 document, just a rendered copy to check before transcribing into Appendix A.

## Setup

Fix $\boldsymbol\mu \in \mathbb{S}^{d-1}$, $q \in [0,1)$, and $\varepsilon \ge 1$ (see the gap
noted at the end for $\tfrac{d-1}{2} < \varepsilon < 1$) — everything below is for a single
fixed $(\boldsymbol\mu, q, \varepsilon)$, with only $p$ varying.

**Definition ($T_p$).** For $p \in (0,1)$, let $Z \sim \mathrm{SphericalTNBBeta}(\boldsymbol\mu, p, q, \varepsilon)$,
let $W = \boldsymbol\mu^\top Z \in [-1,1]$ be its latitude, and let $Y = (1+W)/2 \in (0,1)$ —
by construction of the spherical lift, $Y$'s marginal law is exactly $\mathrm{TNBbeta}(p,q,\varepsilon)$
with density $f(\cdot\,;p,q,\varepsilon)$. Writing $\delta := 1-p \in (0,1)$, define
$$T_p \;:=\; \frac{1-Y}{1-p} \;=\; \frac{1-Y}{\delta}.$$
This is a family of real-valued random variables $\{T_p\}_{p\in(0,1)}$, equivalently indexed
by $\delta \in (0,1)$ as $\delta \to 0^+$. Writing $t$ for a realization of $T_p$, $Y = 1-\delta t$
and $g_\delta(t)$ denotes $T_p$'s density.

**Goal.** Show $\{T_p\}$ is **tight** as $p \to 1^-$: for every $\eta>0$ there exists
$M(\eta) < \infty$ such that
$$\sup_{0 < \delta \le \delta_0} P(T_p > M(\eta)) \;<\; \eta$$
for some $\delta_0>0$ — i.e. one threshold $M$ controls the tail *simultaneously* for every
$p$ close enough to $1$, not just for each $p$ taken on its own.

**Why tightness is enough.** If $T_p$ is tight, then for any fixed $\epsilon'>0$,
$$P(1-Y>\epsilon') = P\!\left(T_p > \frac{\epsilon'}{1-p}\right) \longrightarrow 0$$
as $p\to1$, since $\epsilon'/(1-p)\to\infty$ and the tail of a tight sequence vanishes
uniformly. So $1-Y \xrightarrow{P} 0$, i.e. $Y\to1$ in probability — exactly weak
convergence of $Z$ to $\delta_{\boldsymbol\mu}$. No separate argument is needed for Part 1
once tightness of $T_p$ is established.

## Step 1 — an exact (non-asymptotic) bound on $D(y,p)$

Recall $D(y,p) = p(1-y) + (1-p)y$. Substituting $p=1-\delta$, $y=1-\delta t$:
$$D = (1-\delta)(\delta t) + \delta(1-\delta t) = \delta t - \delta^2 t + \delta - \delta^2 t = \delta\big[(t+1) - 2\delta t\big].$$
For $\delta \le \tfrac12$: $(t+1)-2\delta t \ge (t+1) - t = 1$ (using $2\delta t \le t$), so
$$D(y,p) \ge \delta \qquad \text{for all } t\ge0,\ \delta\le\tfrac12.$$
This is an exact inequality, valid for every $t$ — not just as $\delta\to0$.

## Step 2 — the exact density of $T_p$

Let $g_\delta(t)$ denote the density of $T_p$. Since $f(y) = y^{\varepsilon-1}(1-y)^{\varepsilon-1}h(y)$
with $h(y) = \left[\frac{(1-q)\gamma(y,p)}{y(1-y)}\right]^\varepsilon[1-4q\gamma(y,p)]^{-(\varepsilon+1/2)}$
and $\gamma(y,p)/(y(1-y)) = p(1-p)/D(y,p)^2$, substituting $D = \delta[(t+1)-2\delta t]$
and $g_\delta(t) = \delta f(1-\delta t)$, every factor of $\delta^\varepsilon$ cancels exactly:
$$g_\delta(t) = (1-q)^\varepsilon(1-\delta)^\varepsilon \cdot \frac{t^{\varepsilon-1}(1-\delta t)^{\varepsilon-1}}{\big[(t+1)-2\delta t\big]^{2\varepsilon}} \cdot \big[1-4q\gamma(y,p)\big]^{-(\varepsilon+1/2)}.$$
No approximation anywhere in this step — it's an exact rewriting.

## Step 3 — a uniform bound on $g_\delta(t)$, for $\varepsilon \ge 1$ and $\delta \le \tfrac14$

Bound each factor:

- $(1-\delta)^\varepsilon \le 1$.
- $(1-\delta t)^{\varepsilon - 1} \le 1$, since $0 \le 1-\delta t \le 1$ and $\varepsilon - 1 \ge 0$.
  *(This step needs $\varepsilon \ge 1$ — see the gap below.)*
- $\gamma(y,p) \le \tfrac14$ always (the $\tfrac14\operatorname{sech}^2(\cdot)$ identity), so
  $1-4q\gamma(y,p) \ge 1-q$, hence
  $$[1-4q\gamma(y,p)]^{-(\varepsilon+1/2)} \le (1-q)^{-(\varepsilon+1/2)}$$
  (smaller positive base to a negative power is larger — direction checks out).
- $(t+1) - 2\delta t = t(1-2\delta) + 1 \ge t(1-2\delta) \ge t/2$ for $\delta \le \tfrac14$.

Multiplying through — the $(1-q)^\varepsilon$ and $(1-q)^{-(\varepsilon+1/2)}$ combine into a
single $(1-q)^{-1/2}$ — gives
$$g_\delta(t) \;\le\; (1-q)^{-1/2} \cdot \frac{t^{\varepsilon-1}}{(t/2)^{2\varepsilon}} \;=\; \underbrace{(1-q)^{-1/2}\,4^{\varepsilon}}_{=:C}\; t^{-\varepsilon - 1}.$$

## Step 4 — the tail integral, and the exact claim

Integrating the Step 3 bound, for every $M>0$ and every $\delta \in (0,\tfrac14]$:
$$P(T_p > M) = \int_M^{1/\delta} g_\delta(t)\, dt \;\le\; C\int_M^\infty t^{-\varepsilon-1}\,dt \;=\; \frac{C}{\varepsilon M^\varepsilon}, \qquad C = (1-q)^{-1/2}4^\varepsilon.$$

**Exact claim.** For $\varepsilon \ge 1$, the inequality
$$P(T_p > M) \;\le\; \frac{C}{\varepsilon M^{\varepsilon}}$$
holds for *every* $M>0$ and *every* $p \in [\tfrac34, 1)$ at once — $C$ does not depend on
$p$ (equivalently $\delta$) or $M$.

**Corollary (tightness).** Given $\eta>0$, take $M(\eta) := (C/(\varepsilon\eta))^{1/\varepsilon}$.
Then $C/(\varepsilon M(\eta)^\varepsilon) = \eta$, so
$$\sup_{p \in [3/4,\,1)} P\big(T_p > M(\eta)\big) \;\le\; \eta,$$
which is exactly the tightness goal stated in the Setup (with $\delta_0 = \tfrac14$). Combined
with the "why tightness is enough" argument above, this proves Part 1: $Z \to \delta_{\boldsymbol\mu}$
weakly as $p\to1$.

## Honest gap: $\tfrac{d-1}{2} < \varepsilon < 1$

Only reachable at $d=2,3$. Here $\varepsilon - 1 < 0$, so $(1-\delta t)^{\varepsilon-1} \ge 1$
instead, and can grow as $t \to 1/\delta$ (i.e. $y\to0$ — the *other* pole, where the
univariate density is allowed to diverge for $\varepsilon<1$, Proposition 3.1). This doesn't
actually threaten tightness — a divergent density can still carry arbitrarily small total
mass nearby, and $y^{\varepsilon - 1}$ is integrable near $0$ for any $\varepsilon>0$ — but
Step 3's clean uniform bound doesn't cover it as written, and needs a short separate argument
bounding the mass near $y=0$ directly before Theorem 2.2 can be stated without the
$\varepsilon \ge 1$ restriction.
