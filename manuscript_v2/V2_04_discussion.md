# MANUSCRIPT V2 — Part 4
## Sections 8 to 11: Discussion, Limitations, Reproduction note, Declarations

---

## 8. Discussion

### 8.1 A specification rather than a warning

The existing literature on validation in biomedical image analysis tells
practitioners that rankings are fragile [1 to 4]. That is correct and it is not
actionable on its own. A practitioner building a synthetic benchmark because
they have no annotated clinical data cannot apply advice that reduces to "be
careful" or "add realism."

This paper supplies the missing quantity. Background texture is computable on
any image set from pixels alone. Clinical wound photography measures 2.69 to 3.97 on
it, at a stated resolution and window. A benchmark that matches that range comes within 6% of the agreement between
two independent clinical datasets; a conventional uniform-background benchmark
measures 2.10 and operates at 76% of that ceiling.

A designer with no clinical data at all can still compute the statistic on their
own generator and compare it to a published target. A designer with a small
amount of clinical data, too little to use as a benchmark, can compute the
target themselves from those few images. The statistic requires no annotations
in its unmasked form.

### 8.2 Matching, not maximising

The relationship is not monotone in realism. Agreement peaks inside the clinical
range and falls beyond it. This matters practically, because the intuitive
response to a warning about unrealistic benchmarks is to add as much complexity
as possible, and that is the wrong move. It also matters conceptually: the
finding is about the benchmark resembling the deployment distribution, not about
the benchmark being difficult. A harder benchmark is not a more predictive one.

### 8.3 What the null control buys

Without the contrast sweep, the texture result would be open to the reading that
anything making the target harder to isolate improves rank transfer, which would
make the finding about difficulty rather than about realism. The contrast sweep
spans and exceeds both the clinical and the conventional-synthetic values and
produces no trend, so that reading is excluded.

We recommend the same control to anyone reporting a benchmark-design result. A
design property that improves rank transfer should be shown not to be a proxy
for target difficulty.

### 8.4 The framing result and what it costs the field

The result in Section 7.4 complicates every study that validates against a
single clinical dataset, including the ones this paper builds on.

Two datasets of the same pathology, annotated by experts, from different
clinics, give opposite answers about which synthetic design is more realistic.
Cropping one to the other's framing reverses its answer. The disagreement is
therefore not about anatomy or acquisition but about a convention in how the
photographs were taken and stored.

Two consequences follow. First, a reported agreement between a synthetic
benchmark and a clinical dataset is partly a statement about that dataset's
framing, and should say so. Second, where two clinical references are available
and disagree, the disagreement should be interrogated before it is averaged
away: in our case averaging produced a flat curve that concealed two strong and
opposite trends.

### 8.5 Recommendations

Each is supported by a measurement above.

**Report the background texture statistic of any synthetic benchmark used for
method selection, alongside the value for the clinical setting it stands in
for.** This is a two-line computation and it is the only way the failure
described here becomes visible from inside a single benchmark (Sections 4, 7.2).

**Target the clinical value rather than maximising realism.** Agreement peaks in
the clinical range and declines beyond it (Section 7.2).

**Show that a proposed design property is not a proxy for target difficulty.**
Sweep a visibility-like property as a control and report the null (Section 7.3).

**State the framing convention of any clinical reference.** Agreement with a
clinical dataset is partly agreement with its crop (Section 7.4).

**Check whether a single texture feature carries method performance on your
synthetic benchmark.** If it does, the benchmark is measuring foreground
detection rather than the task (Section 7.5).

---

## 9. What did not reproduce

An earlier version of this work, deposited as a preprint, reported that two
independently constructed synthetic benchmarks rank segmentation methods in
significantly opposite orders (Spearman -0.526, p = 0.014), and that one of them
was significantly negatively correlated with clinical ranking (-0.466,
p = 0.033).

Those results are not reproduced by the code released with this paper. Under
every configuration tested, the two synthetic benchmarks correlate positively:

| configuration | generator A vs generator B |
|---|---|
| 15-feature classifiers, default texture | +0.456 |
| 14-feature classifiers, default texture | +0.476 |
| 14-feature classifiers, texture matched to clinical | +0.423 |
| previously reported | -0.526 |

Three candidate explanations were tested and rejected. Excluding the learned
classifiers entirely makes the correlation more positive (+0.637), so that
family is not responsible. Predicting the entire frame would give Dice 0.241 on
the uniform-background benchmark, not the 0.166 previously reported, so
over-segmentation does not account for the earlier value. Raising the synthetic
benchmark's texture to the clinical value does not flip the sign.

The generator A versus FUSeg correlation does move substantially in the reported
direction once the classifier feature set is corrected as in Section 7.5, from
-0.099 to -0.304 against the reported -0.466, but does not reach it.

We therefore do not claim a rank inversion between synthetic benchmarks. The
supported statement is weaker and sufficient for this paper's argument: a
uniform-background benchmark does not predict clinical method ranking, with
correlations that are not significantly different from zero against either
clinical dataset.

This section is included because the earlier version is public. Readers who
encounter it should treat the rank-inversion claim as superseded.

---

## 10. Limitations

**Clinical domain.** Both clinical datasets are chronic diabetic foot ulcers.
The target values in Section 4 are for that imaging setting. Whether the
statistic's clinical target transfers to dermoscopy, endoscopy or radiology is
untested, and we expect the value to differ by modality even if the principle
holds.

**Number of clinical references.** Two. The ceiling in Section 7.1 is a single
pairwise measurement, and Section 7.4 shows that pair is not interchangeable
with any other pair at different framings.

**Seeds and sample size.** The cross-seed check covers five seeds at four
texture levels, 45 images per cell, and confirms the effect with non-overlapping
ranges (Table 1b). It also shows the single seed used for the full sweeps was
favourable: it put the peak at 100% of the ceiling where the seed average is
94%. The seven-level sweeps in Sections 7.2 and 7.3 remain single-seed and their
point values should be read with that in mind. The contrast null in Section 7.3
has not been repeated across seeds at all.

**The statistic's parameters, including resolution.** The 9 px window, the 31 px
shading sigma and the 512 px canonical side are choices. Resolution is the one
that bites: measured natively, Medetec and FUSeg rank in the opposite order to
the one they take at a common resolution, so any value reported without its
canonical side is uninterpretable. We resample to 512 and state it everywhere. A
sensitivity analysis over window and sigma has not been done, and until it is,
the target range should be read as specific to this parameterisation.

**Upper end of the texture sweep.** Above measured texture 5.6 the generated
backgrounds no longer read as skin on inspection, and those levels are excluded
from the reported trend. The decline beyond the clinical range is therefore
established over a short interval.

**Generator B's design dependency.** Its design targets, including the contrast
calibration in Section 5.2, were derived from the same clinical datasets used to
evaluate it. The targets are coarse and the texture sweep varies a property
rather than fitting one, but replication with a generator designed against
independent clinical data would strengthen it.

**Convolutional baseline.** Small, trained from random initialisation on CPU,
and well below published FUSeg performance. Included to extend the method pool
beyond classical approaches, not as a competitive method.

**Single metric.** Dice throughout, so that benchmarks remain comparable. Metric
selection is itself a source of ranking instability [1 to 3]; whether the
reported relationships persist under boundary-based metrics is untested.

**No physical system or clinical evaluation.** Section 7.7 is simulation. No
manipulator, no surgical task, no clinician, and neither clinical dataset
carries severity labels, so gate correctness cannot be assessed.

---

## 11. Declarations

### Data availability

All code, both generators, all twenty-two segmentation methods, the texture
statistic, trained models, fitted thresholds and per-image result files are
available at [REPOSITORY URL] under the MIT licence and archived at Zenodo
(DOI: [INSERT]). Synthetic benchmarks are regenerated deterministically from the
seeds stated in the text rather than distributed as image files.

The clinical datasets are third party and were not generated by the authors.
Both are publicly available at `github.com/uwm-bigdata/wound-segmentation`,
maintained by the Big Data Analytics and Visualization Laboratory, University of
Wisconsin-Milwaukee. The Foot Ulcer Segmentation Challenge dataset originates
from the AZH Wound and Vascular Center and was released for the MICCAI 2021
challenge; it is described in [6]. The Medetec subset derives from the Medetec
wound database. The authors obtained no special access privileges. No clinical
images are redistributed.

**This statement must not be submitted until the repository actually contains
the released artefacts.**

### Preprint

An earlier version of part of this work was posted as a preprint
(doi:10.21203/rs.3.rs-10362132/v1). The present manuscript reports different
claims and a substantially expanded evaluation. Section 9 states which result of
the earlier version is superseded.

### Competing interests

The authors declare no competing interests.

### Ethics approval

This study used only publicly available, previously published, de-identified
image datasets and required no new ethical approval.

### Funding

[Complete as applicable.]

---

## Figures

**Figure 1.** Rank agreement with clinical data against measured background
texture. Clinical range shaded; clinical-to-clinical ceiling as a horizontal
reference; generator A annotated at its measured value. This is the paper's
central exhibit. Already produced as `results/fig_main.png`, left panel.

**Figure 2.** The framing reversal. Contrast preference for Medetec, FUSeg, and
FUSeg re-cropped to Medetec's framing. Already produced as
`results/fig_main.png`, right panel.

**Figure 3.** Leave-one-out over classifier features, Dice on a
uniform-background benchmark, ordered by effect. One bar separated from the rest
by a factor of eight.

**Figure 4.** Example images: generator A, generator B at matched texture, and a
clinical image from each dataset, with measured texture annotated on each.
This lets a reader calibrate the statistic against appearance.

**Supplementary Table S1.** All 22 methods by 4 benchmarks, mean Dice with
bootstrap 95% confidence intervals.

---

## Pre-submission checklist

- [ ] Insert cross-seed numbers wherever [CROSS-SEED] appears
- [ ] Re-run Section 7.7 under the V2 configuration, or cut the section
- [ ] Fill the [SEARCH] citations in Section 2
- [ ] Verify reference [4] journal version, and whether [5] has a peer-reviewed
      version
- [ ] Produce Figures 3 and 4 and Supplementary Table S1
- [ ] Sensitivity analysis over the statistic's window and sigma, if time allows
- [ ] Push the repository, tag, mint the Zenodo DOI, insert into Section 11
- [ ] Confirm JIIM abstract word limit and structured-abstract headings
- [ ] Select the subscription route at acceptance
