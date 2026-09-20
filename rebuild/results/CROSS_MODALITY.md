# What a second modality showed

Kvasir-SEG (1,000 colonoscopy frames with polyp masks) and its sessile subset
(196 frames) were added as clinical references. They share nothing with the
wound datasets except being clinical images with expert lesion annotations.

Two findings. The first is strong and new. The second puts a hard boundary on
the framing curve, and the boundary is real rather than a matter of framing it
more carefully.

---

## 1. Rank agreement falls away in three clean steps

Every pair of clinical references, 22 methods, Spearman with permutation p.

| pair | rho | p | relationship |
|---|---|---|---|
| AZH test vs AZH train | +0.964 | <0.0001 | same dataset, disjoint slices |
| Kvasir vs Kvasir-sessile | +0.864 | <0.0001 | same source, different curation |
| Medetec vs FUSeg | +0.567 | 0.0071 | same modality, different source |
| FUSeg vs AZH | +0.513 | 0.0162 | same modality, different source |
| Medetec vs AZH | +0.451 | 0.0385 | same modality, different source |
| Medetec vs Kvasir-sessile | +0.322 | 0.1438 | across modality |
| Medetec vs Kvasir | +0.200 | 0.3738 | across modality |
| FUSeg vs Kvasir-sessile | +0.054 | 0.8091 | across modality |
| AZH vs Kvasir-sessile | -0.028 | 0.9027 | across modality |
| AZH vs Kvasir | -0.014 | 0.9503 | across modality |
| FUSeg vs Kvasir | -0.126 | 0.5748 | across modality |

Collapsed:

| relationship | mean rho | n pairs | any significant |
|---|---|---|---|
| same dataset | +0.964 | 1 | yes |
| same source | +0.864 | 1 | yes |
| same modality | **+0.511** | 3 | all three |
| across modality | **+0.068** | 6 | **none** |

The measurement is stable at 0.964 within a dataset, so none of the decline is
sampling noise. Which segmentation method is better is roughly half a rank
correlation within a modality and **nothing at all across one**. Not one of the
six cross-modality pairs reaches significance, and two are negative.

This bounds every benchmark claim in the field, including ours. A benchmark
validated against wound photography says nothing about endoscopy, and the size
of that nothing is now measured.

## 2. The framing curve cannot be tested across modality

Kvasir's median lesion fraction is 0.106, almost exactly Medetec's 0.102, so the
framing curve makes a prediction for it. Placed on the wound-built curve:

| dataset | lesion fraction | predicted | measured | residual |
|---|---|---|---|---|
| Kvasir | 0.1062 | -0.924 | -0.964 | -0.040 |
| Kvasir-sessile | 0.0427 | +0.643 | -0.964 | **-1.607** |

One near-perfect hit and one of the largest misses the scale allows. Before
reading anything into either, the underlying agreements:

| contrast | vs Medetec | vs Kvasir | vs Kvasir-sessile |
|---|---|---|---|
| 19.5 | +0.573 (p 0.01) | +0.081 (p 0.72) | +0.128 (p 0.56) |
| 28.4 | +0.613 (p 0.00) | +0.068 (p 0.76) | +0.126 (p 0.58) |
| 38.4 | +0.605 (p 0.00) | -0.029 (p 0.90) | +0.013 (p 0.96) |
| 49.4 | +0.535 (p 0.01) | -0.088 (p 0.70) | -0.054 (p 0.82) |
| 59.7 | +0.518 (p 0.01) | -0.137 (p 0.53) | -0.100 (p 0.66) |
| 72.7 | +0.438 (p 0.04) | -0.191 (p 0.39) | -0.156 (p 0.49) |
| 86.6 | +0.434 (p 0.04) | -0.186 (p 0.40) | -0.134 (p 0.54) |

Every endoscopy agreement sits inside +/-0.20 with p between 0.39 and 0.96. The
verdict statistic is a trend over seven numbers that are not distinguishable
from zero, so both endoscopy verdicts of -0.964 are monotone orderings of noise.
The Kvasir "hit" is not evidence for the curve and the sessile "miss" is not
evidence against it. Neither point is interpretable.

**The framing curve is therefore scoped to within-modality use, and the reason
is finding 1.** A verdict about which synthetic design a reference prefers only
exists where the benchmark and the reference agree about method ranking at all.
Across modality they do not, so there is no verdict to predict.

One observation worth keeping without claiming it: both endoscopy references
decline in the same direction as Medetec, from slightly positive at low contrast
to slightly negative at high. The sign pattern is consistent across two
independent endoscopy sets. With p >= 0.39 throughout it stays an observation.

## 3. What this changes in the paper

Gains:

- A measured three-level hierarchy of rank agreement, with the within-dataset
  control that makes the decline interpretable. This is a general result about
  benchmark transfer, not a wound result.
- A second modality, which was the stated gate on the higher venues.
- Honest scope for the framing curve instead of an assumed generality.

Costs:

- The framing curve is a within-modality statement. It cannot be claimed as a
  universal property of benchmark references on this evidence.
- Testing it in a second modality needs synthetic benchmarks built for that
  modality, so that the reference and the benchmark share ranking signal. That
  is a second paper, not a section of this one.

## 4. Caveats

- One endoscopy source. Kvasir and its sessile subset overlap in origin, which
  is why they agree at 0.864; they are not two independent endoscopy datasets.
- Kvasir frames were resampled to a 256 px long side to match Medetec's
  resolution and keep scoring tractable. Method ranking is not obviously
  resolution-invariant, and the project has already been bitten once by a
  resolution-dependent statistic.
- Kvasir masks are JPEG-compressed with anti-aliased edges, so they are
  thresholded at 127 rather than on any non-zero value as the wound masks are.
  The threshold was not swept.
- 120 frames per endoscopy reference against 150 for AZH and 152 for Medetec.
- The cross-modality agreements being indistinguishable from zero is itself
  measured at n = 22 methods, which is a small sample for a correlation. The
  conclusion is that there is no detectable agreement, not that agreement is
  provably zero.
