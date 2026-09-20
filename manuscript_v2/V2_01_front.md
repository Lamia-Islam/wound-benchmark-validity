# MANUSCRIPT V2 — Part 1
## Title, Abstract, Introduction

**Target:** Journal of Imaging Informatics in Medicine (Springer). Hybrid,
subscription route at acceptance for zero cost. JIIM gives precedence to papers
reporting validation and implementation of informatics technologies over
stand-alone evaluation of image analysis tools, and this draft is framed for
that: it is about how to specify a validation dataset, evidenced through a
deployed consumer of the output.

**What changed from V1.** V1 led with the claim that two synthetic benchmarks
rank methods in significantly opposite orders. That result did not survive
reconstruction. V2 leads instead with a measured specification, a dose-response
relationship, and a framing result, none of which were in V1. Section 9 states
plainly what did not reproduce.

---

## Title

**How much realism does a synthetic segmentation benchmark need? A measured
specification from wound imaging**

Alternatives:

- *Matching background texture makes synthetic segmentation benchmarks
  predictive of clinical method ranking*
- *A measurable target for synthetic benchmark design in medical image
  segmentation*

---

## Abstract

**Purpose.** Synthetic images are widely used to compare medical image
segmentation methods when annotated clinical data is scarce. The practical
question is not whether synthetic data is realistic in general but how much of
which property it needs. We ask whether a benchmark's capacity to predict
clinical method ranking is governed by a property that can be measured on its
images, and what the target value is.

**Methods.** Twenty-two segmentation methods spanning thresholding, colour
channels, clustering, region growing, active contours, per-pixel learned
classifiers and a convolutional network were scored by Dice against ground truth
on synthetic benchmarks and on two independent clinical wound datasets (FUSeg,
Medetec). We define a background texture statistic computed from pixels, the
median local standard deviation of shading-corrected luminance in the background
region, and sweep a synthetic generator across it. Method rankings are compared
across benchmarks by Spearman correlation with permutation testing. A contrast
sweep serves as a null control, and a framing manipulation tests whether the
clinical reference itself is well defined.

**Results.** Rank agreement with clinical data rises with measured background
texture (Spearman 0.964, p = 0.0028 over seven levels at one seed) and peaks
where the synthetic value matches the clinical range. Across five generator
seeds at four levels the rise is confirmed with non-overlapping ranges
(Mann-Whitney p = 0.0040) and is a threshold rather than a gradual effect: small
amounts of texture change nothing, and the gain appears once the clinical range
is reached. The best setting attains rank agreement 0.536 averaged over seeds,
which is 94% of the 0.567 agreement between the two clinical datasets
themselves. A conventional uniform-background generator measures 2.10
on this statistic against a clinical range of 2.69 to 3.97, and operates at 76%
of that ceiling. Sweeping lesion-to-surround colour contrast produces no trend
(Spearman 0.143, p = 0.780), so the effect is not target visibility. The two
clinical datasets appear to disagree about which synthetic design is realistic,
but cropping one to the other's framing reverses its preference (Spearman +0.964
to -0.964), showing that the disagreement is an artifact of framing convention
rather than of clinic or anatomy. Finally, leave-one-out over the classifier's
input features identifies a single texture feature, local standard deviation of
the a* channel, whose removal collapses performance on a uniform-background
benchmark from 0.936 to 0.122 while leaving clinical performance largely intact.

**Conclusion.** Predictive validity of a synthetic segmentation benchmark is
governed by a property that can be measured on any image set, and the target is
to match the clinical value rather than to maximise realism. We give the
statistic, the clinical target range, and the achievable ceiling. We also show
that "validate against clinical data" is underspecified until the framing
convention of the clinical reference is stated.

**Keywords.** validation methodology; benchmark design; synthetic data; wound
segmentation; cross-dataset generalisation; imaging informatics

---

## 1. Introduction

Annotated clinical images are expensive, constrained by consent and governance,
and unevenly distributed across populations and care settings. Synthetic images
are the standard substitute, used to develop pipelines, tune parameters, and
choose among candidate methods before clinical data is available or in place of
it.

Nobody expects synthetic Dice to equal clinical Dice. What is assumed, usually
without statement, is that relative performance transfers: that the method which
looks better on the synthetic benchmark will be better clinically. That
assumption is what makes synthetic benchmarks useful for method selection.

The existing literature on validation in biomedical image analysis establishes
that method rankings are fragile, and attributes the fragility to metric
selection, aggregation scheme, annotator variability, and which test set is used
[1 to 3]. It does not say how to build a benchmark that avoids the problem, and
it offers no quantity a benchmark designer can measure on their own images to
find out where they stand.

That gap is the subject of this paper. We hold metric, aggregation and reference
annotations fixed, vary one measurable property of the benchmark images, and ask
how rank agreement with clinical data responds.

Three things follow, and the third was a surprise.

First, the relationship is a matching criterion rather than a monotone
preference for realism. Rank agreement rises with background texture up to the
clinical value and falls beyond it. Maximising realism is not the objective;
matching it is.

Second, there is a ceiling and it is nearly reached. The natural upper bound for
any benchmark is how well one clinical dataset predicts method ranking on
another. At matched texture the synthetic benchmark comes within 6% of it. The useful message about
synthetic validation is therefore not a warning but a specification.

Third, the clinical reference is not a single thing. Our two clinical datasets
appear to want opposite amounts of lesion-to-surround contrast. Cropping one to
the other's framing reverses its preference, so what looked like a property of
clinic or anatomy is a property of how the photographs were taken and cropped.
Any instruction to validate against clinical data has to name the framing
convention to be actionable.

### 1.1 Contributions

1. **A measured statistic and its clinical target.** Background texture,
   computed from pixels on any image set with or without masks. Clinical wound
   photography measures 2.69 to 3.97 at a stated resolution and window. A
   conventional uniform-background
   generator measures 2.10 (Section 4, Section 6.1).
2. **A dose-response relationship, peaking at the clinical value.** Rank
   agreement with clinical data rises monotonically with measured texture
   (Spearman 0.964, p = 0.0028) and peaks at the top of the clinical range
   (Section 6.2).
3. **The ceiling is nearly reached.** At matched texture, rank agreement over
   five seeds is 0.536 against a clinical-to-clinical ceiling of 0.567, or 94%,
   with seed ranges that do not overlap the uniform-background case
   (Section 6.2).
4. **A null control that rules out target visibility.** Sweeping
   lesion-to-surround colour contrast gives no trend (Spearman 0.143,
   p = 0.780). The effect is background structure, not how visible the target is
   (Section 6.3).
5. **Framing determines apparent realism preference.** Cropping one clinical
   dataset to the other's framing reverses its contrast preference from +0.964
   to -0.964. The clinical reference must be specified by framing convention
   (Section 6.4).
6. **A single-feature demonstration of the mechanism.** Leave-one-out over 15
   classifier input features identifies local standard deviation of a* as the
   feature whose removal collapses uniform-background performance from 0.936 to
   0.122 while clinical performance holds. On a uniform background the question
   "is there local texture here?" answers "is this the lesion?" exactly
   (Section 6.5).
7. **Released artefacts.** Both generators, all 22 methods, the texture
   statistic, every evaluation script and the per-image result files, under MIT
   licence with an archived DOI.

### 1.2 Scope and what this paper does not claim

We do not claim synthetic data is unsuitable for validation. The opposite: at a
measured and achievable setting it performs at the clinical-to-clinical ceiling.

We do not claim the texture statistic is the only property that matters. We
claim it is one that can be measured, that it has a target value, and that a
common design choice sits at roughly half of it.

We do not claim a clinically deployable wound assessment system. The clinical
datasets are chronic diabetic foot ulcers and the robotic pipeline of Section 7
is evaluated in simulation without clinical severity labels. It serves as an
instrumented consumer of segmentation output.

We report in Section 9 a result from an earlier version of this work that did
not survive reconstruction, and we do not rely on it.
