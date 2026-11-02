# ITD-only HRTF feasibility gate on SADIE II (exploratory, no model trained)

Date: 2026-11-01. Base commit: `6e2e00d`; working tree uncommitted (new files
`apps/data/download_sadie2.py`, `apps/eval/hrtf_itd_gate.py`). Computes the "feasibility gate"
of `hrtf_itd_ring_preregistration_DRAFT_2026-11-01.md`. The draft is not frozen.

Commands (login node, light; `h5py` is not a project dependency):

    uv run python -m apps.data.download_sadie2
    uv run --with h5py python -m apps.eval.hrtf_itd_gate --estimator onset
    uv run --with h5py python -m apps.eval.hrtf_itd_gate --estimator xcorr

## Data

SADIE II (Zenodo record 10886409, Apache-2.0; cite DOI 10.3390/app8112029). The 18 human subjects
H3-H20, `H<k>_HRIR_SOFA.zip`, md5-verified against Zenodo, extracted to
`$TNBBETA_DATA_DIR/hrtf/sadie2/H<k>/`. Used: the 96 kHz, 512-tap files (SimpleFreeFieldHRIR),
2818 directions per subject at 1.2 m, az 0-355, el -90..90. File header: "Filters were low
frequency extended, diffuse field equalized and windowed ... (approx.) Linear phase HRIRs."

Direction `z` = head-frame unit vector (x front, y left, z up) from azimuth/elevation; ring axis
`a = +y`. ITD (right ear minus left ear, microseconds, positive = source on the left) from
HRIRs low-passed at 3 kHz (4th-order Butterworth, zero phase), by (a) first onset at -30 dB of
each ear's peak with linear interpolation, or (b) the lag maximising the interaural
cross-correlation with parabolic interpolation. Area weights = spherical-Voronoi cell areas
(the duplicated pole positions share their cell).

## Results (6 folds of 3 held-out subjects, split seed 0; 50 ITD quantile bins fitted on the
other subjects; area-weighted mean held-out NLL of the true direction, nats)

| | onset (-30 dB) | cross-correlation |
|---|---|---|
| ITD at w > 0.95 / w < -0.95 (us) | 500 / -504 | 699 / -694 |
| fraction of |ITD| > 750 us | 0.003 | 0.005 |
| corr(ITD, w) | 0.823 | 0.983 |
| ITD variance explained by binned w | 0.693 | 0.969 |
| median within-bin sd of w (ring thickness) | 0.333 | 0.085 |
| uniform (log 4 pi) | 2.531 | 2.531 |
| oracle ring (uniform azimuth about +y) | 1.834 | 0.520 |
| oracle cap (per-bin ML vMF) | 2.076 | 1.860 |
| free-form vMF-KDE, kappa 30 / 100 (context) | 1.737 / 1.622 | 0.853 / 0.373 |
| ring minus cap | **-0.242** | **-1.340** |

## Reading

* The draft's stop rule ("stop if the oracle ring is not better than the oracle cap by at least
  0.25 nats"; threshold still [OPEN]) is **narrowly failed with the onset estimator (0.242)** and
  **passed by a wide margin with the cross-correlation estimator (1.34)**.
* The two estimators disagree on the data, not on the method: with the onset estimator the ITD at
  the interaural poles is ~500 us against ~700 us (a typical human value) with cross-correlation,
  and only 69% of the ITD variance is explained by the lateral coordinate `w` against 97%. A
  plausible cause is the linear-phase, windowed, diffuse-field-equalised filters (pre-ringing can
  trigger a -30 dB onset threshold early), but this was not tested.
* This reverses the draft's choice of primary ITD estimator after seeing the gate. The change is
  justified by estimator quality (physical ITD at the poles, variance explained), not by a model
  result, but it is a researcher degree of freedom and is recorded as such: the primary analysis
  uses cross-correlation, the onset estimator is reported as a sensitivity analysis and, per the
  numbers above, would essentially fail the gate.
* With cross-correlation the posterior over direction given ITD is a thin ring (sd of `w` within a
  bin 0.085, about 5 degrees): the same regime as the synthetic rings, and the oracle ring beats
  the oracle cap by 1.34 nats, so a family difference is plausible.
* The axisymmetric ring is not the whole story: the free-form KDE (kappa 100) is 0.15 nats better
  than the oracle ring, so the real conditional has some non-axisymmetric structure (ITD varies
  with elevation / front-back at fixed `w`). It is a context number (a KDE fitted per bin is not
  an amortised model and kappa was not tuned).
* Some ITDs exceed 750 us (0.5% with cross-correlation; the range is about -959 to +894 us);
  these are outliers that were not removed or investigated.

## Caveats

One dataset (18 subjects; each fold fits on 15 and tests on 3), one fold split (seed 0), ITD binned into
50 quantile bins, ring density from a 40-bin histogram with light smoothing. The oracle cap is a
per-bin ML fit on training subjects, a stronger cap than an amortised network could be. Nothing
here estimates the difference between amortised families.

Figure explaining the two estimators on subject H3 (`apps/eval/hrtf_itd_figure.py`): `figures/hrtf_itd_estimators_2026-11-01.png`. For three horizontal-plane directions the -30 dB onsets fall 1.5-1.9 ms before the main pulse of the low-passed HRIR (e.g. az 0: peaks 2.333/2.323 ms but onsets 0.706/0.832 ms, giving an onset ITD of 126 us against -3 us by cross-correlation for a frontal source), i.e. the threshold fires on faint precursors. The cause of those precursors (filter processing) was not confirmed.
