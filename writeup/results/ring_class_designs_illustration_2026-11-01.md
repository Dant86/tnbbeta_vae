# Four-class ring designs on S^2: illustration of the two candidate generative models

Date: 2026-11-01. Base commit: `ec2677e` (script and figures only; no experiment was run).
Illustration of designs under discussion, not a result; nothing here changes a
pre-registered analysis.

Command (login node, a few seconds; matplotlib is not a project dependency):

    uv run --with matplotlib python -m apps.synthetic.ring_class_designs_plot \
        --out writeup/results/figures/ring_class_designs_2026-11-01.png

Writes `figures/ring_class_designs_2026-11-01.png` and
`figures/ring_class_designs_2026-11-01_design2_per_class.png`.

Both designs use `x = h(a_c . z) + noise` with `h(w) = [w, sin 1.5 w, cos 2 w]` and
sigma = 0.2, so the exact posterior given `x` (uniform prior) is a latitude ring about the
class axis, with density proportional to `exp(-||x - h(w)||^2 / (2 sigma^2))` in `w`.

* **Design 1, one shared axis.** `z ~ Uniform(S^2)`; the class is the latitude band of
  `w = a . z` (four equal-mass bands, edges -1, -0.5, 0, 0.5, 1). The mixture of bands is
  exactly uniform (w is uniform on [-1, 1] at d = 3), so the VAE's uniform prior is the true
  marginal. This is the existing single-axis task with a smoother `h` and a class label.
* **Design 2, one axis per class.** Axes are the tetrahedron vertices; `z ~ Uniform(S^2)` for
  every class (so `p(z | class)` is uniform and the z samples of the four classes are
  indistinguishable, as the figure shows); the class changes the observation map,
  `x = h(a_c . z) + o_c + noise`, with a class offset `o_c` so the class is identifiable from
  `x`. The classes differ in their posteriors, which are rings about `a_c`.

Check run by the script: with this smoother `h` and sigma = 0.2, 0.000 of 2000 draws of `x`
have a second latitude mode with more than 5% of the posterior mass (the original
`h = [w, sin 3w, cos 5w]` had 18.9% on the stage-3 test set; see
`ring_posterior_strata_2026-11-01.md`).

Caveats: posteriors with this `h` are broad (the rings are bands of roughly 10 degrees), so
design 1's example posteriors spill over band edges. Colours are the documented reference
palette (slots 1, 2, 3 and 7); the palette validator could not be run (no `node` on this
machine), so the set is not validated for colour-blind separation, and each class also has
its own marker shape.
