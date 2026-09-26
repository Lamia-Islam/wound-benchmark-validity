# Manuscript V3, Part 3: results

All agreement values are Spearman rank correlations of per-method mean Dice over
the 32-method pool unless stated otherwise.

---

## 7.1 Rank agreement falls in steps, and the decline is real

| pair | rho | p | 95% CI | relationship |
|---|---|---|---|---|
| Kvasir vs Kvasir-sessile | +0.952 | <0.0001 | [+0.871, +0.982] | same source |
| AZH test vs AZH train slice | +0.944 | <0.0001 | [+0.835, +0.984] | same dataset |
| FUSeg vs AZH | +0.660 | 0.0001 | [+0.430, +0.804] | same modality |
| Medetec vs AZH | +0.568 | 0.0012 | [+0.256, +0.781] | same modality |
| Medetec vs FUSeg | +0.461 | 0.0081 | [+0.085, +0.758] | same modality |
| Medetec vs Kvasir | +0.019 | 0.9172 | [-0.318, +0.360] | across modality |
| Medetec vs Kvasir-sessile | -0.042 | 0.8185 | [-0.397, +0.338] | across modality |
| AZH vs Kvasir | -0.191 | 0.2991 | [-0.506, +0.164] | across modality |
| AZH vs Kvasir-sessile | -0.277 | 0.1289 | [-0.579, +0.081] | across modality |
| FUSeg vs Kvasir | -0.477 | 0.0065 | [-0.706, -0.152] | across modality |
| FUSeg vs Kvasir-sessile | -0.552 | 0.0011 | [-0.741, -0.224] | across modality |

Collapsed: +0.944 within a dataset, +0.952 within a source, +0.563 on average
within a modality across sources, and -0.253 on average across modalities.

**The within-dataset control is what makes this interpretable.** Splitting one
dataset into two disjoint 150-image halves and scoring each independently gives
+0.944. The measurement therefore reproduces itself at 0.94, and the drop to
0.56 between different datasets of the same modality is genuine disagreement
between references rather than sampling noise in the measurement.

All three same-modality pairs are significant with confidence intervals
excluding zero. Over the earlier 22-method pool the weakest of them sat at
p = 0.0385 with an interval touching zero, so the enlarged pool strengthened
rather than weakened this structure.

## 7.2 The apparent cross-modality inversion is a composition artifact

The pooled across-modality mean of -0.253 invites the reading that methods good
at wound segmentation are worse than chance at colonoscopy. That reading is
wrong, and the way it fails is itself a result.

Splitting the pool:

| subset | same modality | across modality |
|---|---|---|
| all 32 methods | +0.563 | **-0.253** |
| 22 classical and pixel-classifier methods | +0.511 | **+0.108** |
| 10 modern architectures | +0.321 | **+0.236** |

The negative correlation exists only in the pooled comparison. Within either
subgroup it is weakly positive and not significant. The cause is visible in mean
Dice by subset:

| reference | 10 deep | 22 others | gap |
|---|---|---|---|
| Medetec | 0.374 | 0.295 | +0.079 |
| FUSeg | 0.581 | 0.132 | +0.449 |
| AZH | 0.396 | 0.330 | +0.066 |
| Kvasir | 0.062 | 0.200 | **-0.138** |
| Kvasir-sessile | 0.024 | 0.114 | **-0.091** |

The deep models were trained on wound photography. They beat the untrained
methods on all three wound references and collapse below them on both endoscopy
references. Two groups exchanging places between domains produces a negative
pooled rank correlation without any individual pair of methods inverting.

Two claims follow. First, within a homogeneous method pool there is no
detectable cross-modality rank agreement: +0.108 and +0.236, neither
significant, at a detection limit of 0.35. Second, and this is the transferable
warning, **in a pool mixing trained and untrained methods a cross-domain rank
comparison measures which methods had domain-matched training.** Its sign is set
by pool composition. A practitioner comparing benchmarks across domains with
such a pool will measure this artifact and can readily mistake it for a property
of the data.

## 7.3 Benchmark validity depends on the method pool

The clinical-to-clinical ceiling, measured on the two references no method was
trained on, is Medetec against AZH at **+0.568** (p = 0.0012).

Generator B against those same two references, over the 32-method pool:

| texture amplitude | measured texture | vs Medetec | vs AZH | mean | % of ceiling |
|---|---|---|---|---|---|
| 11 | 3.45 | +0.234 (p 0.193) | +0.058 (p 0.758) | +0.146 | 26% |
| 28 | 4.30 | +0.366 (p 0.040) | +0.157 (p 0.390) | +0.261 | 46% |
| 40 | 5.07 | +0.360 (p 0.043) | +0.194 (p 0.286) | +0.277 | **49%** |

Over the earlier 22-method pool, and against Medetec and FUSeg, the best
configuration of the same generator reached 98% of the then-measured ceiling.

Two changes account for the gap and both matter. FUSeg is excluded here because
eleven methods were trained on its training split. And the pool now contains
modern architectures whose relative ordering the synthetic benchmark does not
reproduce.

**A synthetic benchmark that appears to reach the clinical ceiling when assessed
over classical methods reaches about half of it once the methods people deploy
are in the pool.** This is the single most practically consequential number in
the paper. It says that validity figures for synthetic benchmarks are not
portable across method pools, and that assessing validity over convenient
classical baselines will overstate it.

The texture dose-response survives the pool change: agreement rises
monotonically with background texture, +0.146 to +0.261 to +0.277, which is the
same direction reported over the smaller pool.

## 7.4 Within one modality, lesion fraction predicts the direction of bias

A reference's contrast verdict is the slope of its rank agreement against the
benchmark's measured lesion-to-surround saturation contrast, per 10 saturation
units. Positive means the reference rewards high contrast.

The primary curve is built from AZH, the wound reference no method was trained
on. AZH frames were cropped around their own lesions to seven target lesion
fractions. Every crop uses the same 140 frames and the same annotations, so
lesion fraction is the only thing that differs between the seven references.

| lesion fraction | verdict, Dice | sd over 4 seeds | verdict, NSD | sd over 4 seeds |
|---|---|---|---|---|
| 0.0353 | +0.077 | 0.006 | +0.080 | 0.005 |
| 0.0550 | +0.029 | 0.009 | +0.061 | 0.005 |
| 0.0801 | -0.002 | 0.010 | +0.038 | 0.005 |
| 0.1101 | -0.037 | 0.009 | +0.004 | 0.005 |
| 0.1598 | -0.060 | 0.008 | -0.041 | 0.005 |
| 0.2194 | -0.078 | 0.007 | -0.056 | 0.006 |
| 0.2981 | -0.086 | 0.007 | -0.079 | 0.007 |

**Monotonicity against lesion fraction is Spearman -1.000 at p = 0.0005 in every
one of the four seeds, under both metrics.** Eight of eight seed and metric
combinations are perfectly monotonic. Seed standard deviation never exceeds
0.010 against a curve range of 0.163.

The boundary metric is the load-bearing control. Dice is biased by region size
and this experiment varies region size, so a Dice-only result would be
confounded by construction. Normalised surface Dice gives the same monotonic
relationship with the same sign.

### Replication on the contaminated reference

The same construction on FUSeg crops, spanning lesion fractions 0.008 to 0.160,
gives monotonicity between -0.881 and -1.000 across the same eight seed and
metric combinations. We report it as a replication rather than as the primary
result because eleven of the 32 methods were trained on FUSeg training images.
Removing that contamination sharpened the relationship rather than weakening it.

### The crossover is not a constant

| curve | metric | sign change |
|---|---|---|
| AZH-built | Dice | 0.055 to 0.080 |
| AZH-built | NSD | 0.110 to 0.160 |
| FUSeg-built | Dice | 0.110 to 0.160 |
| FUSeg-built | NSD | 0.110 to 0.160 |

Three of four agree. The crossover moves with the reference, with the metric,
and, as the 22-method version of this experiment showed, with the method pool.
The direction of the relationship moves with none of them. We report the
crossover as an interval observed under stated conditions and not as a threshold
for others to apply.

### Held-out placement, and a systematic offset

Medetec, lesion fraction 0.1024, placed on the AZH-built curve:

| metric | predicted | measured | sd over seeds | residual | as % of range |
|---|---|---|---|---|---|
| Dice | -0.028 | +0.038 | 0.004 | +0.066 | 40.6% |
| NSD | +0.013 | +0.056 | 0.006 | +0.043 | 27.0% |

The placement has the correct sign relative to the low-fraction end of the curve
but overshoots it. On the FUSeg-built curve the corresponding residuals for
Medetec and AZH were 12% to 19%.

Across two independently built curves and three held-out datasets, **every real
clinical dataset lands above the curve at its own lesion fraction**. Four of
four placements in one direction is systematic, and the mechanism is visible: a
crop taken to reach lesion fraction f is not the same object as a dataset
natively framed at f. Cropping to a high lesion fraction discards the
surrounding limb, background and clutter, whereas a natively framed dataset
still contains whatever context the photographer chose to include. Crop series
therefore overstate the framing effect, and real datasets fall short of the
curve built from them.

This restricts the claim rather than retracting it. The crop series establishes
the direction and the ordering, which is what is claimed. It is not a calibrated
predictor of a natively framed dataset's verdict.

## 7.5 Validity is not reducible to the measured texture statistic

If benchmark validity were a function of the background texture statistic, two
benchmarks matched on it would agree. Against Medetec and AZH, over 32 methods:

| benchmark | measured texture | vs Medetec | vs AZH | mean |
|---|---|---|---|---|
| generator A, uniform drape | 2.10 | -0.045 | -0.261 | **-0.153** |
| generator C, no noise | 2.34 | +0.116 | -0.102 | **+0.007** |
| generator C, noise 3.4 | 3.20 | +0.132 | -0.097 | **+0.017** |
| generator B, texture ablated | 3.23 | +0.201 | +0.063 | **+0.132** |

Two matched pairs. Generator C at 3.20 and generator B at 3.23 are matched to
within 0.03 on the statistic and differ by **0.115** in validity. Generator A at
2.10 and generator C at 2.34 are matched to within 0.24 and differ by 0.160.

For scale, the entire effect of sweeping texture within generator B across its
clinical range, from 3.45 to 5.07, is 0.131 (Section 7.3).

**The spread between generator families at a fixed value of the statistic is as
large as the whole within-family effect of the statistic.** The reduction fails.
The statistic captures something real, since texture predicts validity within a
family, but it is not a specification a benchmark designer could target.

## 7.6 Cross-modality verdicts are not interpretable

Kvasir-SEG's median lesion fraction is 0.106, close to Medetec's 0.102, so the
framing curve makes a prediction for it. Placed on the wound-built curve over
the 22-method pool, Kvasir's residual was 0.040 and the sessile subset's was
1.607 on a 1.93 scale, one apparent bullseye and one of the largest misses the
scale allows.

Neither is meaningful. The underlying agreements between the synthetic wound
benchmarks and the endoscopy references run from -0.19 to +0.13 with p-values
from 0.39 to 0.96. The verdict is a trend fitted through values that are not
distinguishable from zero.

This is a direct consequence of Section 7.1. A verdict about which synthetic
design a reference prefers exists only where the benchmark and the reference
agree about method ranking at all. Across modality they do not, so there is
nothing to predict. The framing result of Section 7.4 is scoped to
within-modality use for this reason, and not by assumption.

## 7.7 What did not reproduce

An earlier version of this work reported a significant rank inversion between
two synthetic benchmarks, rho = -0.526, p = 0.014. It was not reproduced during
reconstruction. Three configurations were tested and returned +0.456, +0.476 and
+0.423, and three candidate explanations were ruled out. It is withdrawn.

An earlier version reported that correcting a six-fold chroma error in generator
B improved benchmark validity from 94% to 99% of ceiling. Extending the
corrected generator from three seeds to the same five seeds as the uncorrected
one removes most of that: the seed-matched mean difference over twenty cells is
-0.001, with 10 of 20 improved. The correction is worth +0.023 at clinical-range
texture and costs -0.026 at low texture. It is reported as texture-dependent,
not as an improvement.
