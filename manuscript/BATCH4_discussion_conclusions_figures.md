# MANUSCRIPT — BATCH 4
## Sections 7, 9, 10 + figure specifications

> RECOVERED from conversation f7ee7590-a160-40ce-bc6c-e90dd81e1c63, turn 75.
> Original file path in the lost container: `/mnt/user-data/outputs/MANUSCRIPT_BATCH4.md`

Completes the draft. Section 8 (Limitations) is in Batch 3.

---

## 7. Discussion

### 7.1 What the measurements show

Three practices, each defensible in isolation, appear together in the pipeline
studied here.

The first is validating segmentation on synthetic imagery composited onto
uniform background. This rewards figure–ground separation, which on such images
is indistinguishable from lesion localisation. Methods that search for lesion
tissue within skin are penalised for declining to claim the whole foreground.
The consequence is a benchmark whose method ordering bears no reliable relation
to clinical ordering, and which on one of our two generators is inversely
related to it.

The second is validating a competence estimator on a held-out split of a single
dataset. Signals that track lesion size and framing convention are rewarded,
because within one dataset those properties are stable and correlate with
difficulty. Across sources they do not. The same seventeen signals and the same
model yield ρ = 0.948 within Medetec and ρ = 0.316 across sources.

The third is fitting safety thresholds against whichever perception module is
current. The thresholds encode the feature distributions that module produces,
so replacing it invalidates them. In our pipeline, improving segmentation
accuracy from 2.2% to 60.0% on the synthetic benchmark drove the intermediate
severity grade to 0 of 15 correct, with every borderline case reclassified as
healthy.

Individually each practice is a modest methodological compromise. Together they
produce a system reporting 100% three-grade accuracy under its own validation
while granting full autonomy on 62% of clinical frames where its segmentation
had failed.

### 7.2 Why background texture is the governing property

The ablation isolates a single mechanism rather than a set of contributing
factors. Background texture accounts for a change of 0.532 in mean rank
correlation; lesion size, placement, clutter and shape together account for less
than 0.08 each, and shape not at all.

This asymmetry is informative. Realism in the lesion — its size distribution,
irregular boundary, off-centre placement — does not make a benchmark predictive.
Realism in what surrounds the lesion does. The reason is structural: the
validity of a segmentation benchmark depends on whether the target region is
separable from its surroundings by the same cues that distinguish it clinically.
A uniform background supplies a cue that is absent in clinical images and
sufficient on its own, and no amount of lesion realism removes it.

The practical implication is narrow enough to act on. A synthetic segmentation
benchmark intended for method comparison must place the target within a
background that resembles the tissue it will be surrounded by clinically. This
is a weaker requirement than full image realism and does not require generative
modelling; our generator B uses low-frequency mottling, additive noise, a
directional illumination gradient, and simple hair strokes.

### 7.3 Relation to prior work on validation

That rankings in biomedical image analysis are unstable is established. Prior
analyses attribute instability to metric selection, aggregation scheme,
annotator variability, and choice of test set [1–3]. Our study holds metric,
aggregation and reference annotations fixed and varies one property of the
benchmark data, which localises a source of instability that has been noted in
aggregate but not isolated.

The relationship is complementary rather than competing. Metrics Reloaded [2]
guides metric selection given a problem; the present work concerns whether the
data on which those metrics are computed supports the conclusion being drawn.
Both are prerequisites for a valid comparison, and neither substitutes for the
other.

### 7.4 Recommendations

Stated as protocol requirements, each supported by a measurement reported above.

**Synthetic benchmarks used for method selection should have non-uniform
backgrounds resembling the clinical surround.** Uniform-background benchmarks
reward figure–ground separation. In our study this produced rank correlations of
−0.121 and −0.466 against clinical data, the latter significant.

**Competence and uncertainty estimators should be validated across sources, not
on held-out splits of one dataset.** Within-dataset validation overstated
performance by ρ = 0.948 against ρ = 0.316 for the identical model.

**Safety gate performance should be measured against a ground-truth perception
metric, not inferred from downstream task accuracy.** In our pipeline the two
diverged completely: synthetic classification accuracy of 100% coincided with an
unsafe-optimistic rate of 0.62.

**Thresholds coupling a safety layer to a perception module should be refitted
whenever that module changes, and the coupling documented.** Substituting one
segmentation for a more accurate one inverted the gate's behaviour on the
intermediate severity grade.

**Where a synthetic benchmark is used, its rank agreement with at least one
clinical dataset should be reported.** This is inexpensive relative to the
consequences of omitting it, and is the only means by which the failure
described here becomes visible.

---

## 9. Conclusions

Method rankings obtained on synthetic wound segmentation benchmarks do not
reliably transfer to clinical data. Two independently constructed synthetic
benchmarks ranked twenty-two segmentation methods in significantly opposite
orders, while two independent clinical datasets agreed. Feature-wise ablation
identified background texture as the governing property: its removal reduced
mean rank correlation with clinical data by 0.532 and reversed the sign, an
effect approximately seven times larger than that of lesion size, placement,
clutter or shape.

The mechanism is that compositing a lesion onto uniform background makes
figure–ground separation sufficient for a high segmentation score, which is not
the case clinically. A convolutional network achieving the best Dice of any
method on both clinical datasets scored 0.080 on the uniform-background
benchmark, near the bottom of the pool.

In a perception-gated autonomy pipeline consuming this segmentation, full
autonomy was granted on 27–62% of frames where segmentation had failed, and
threshold recalibration that raised synthetic accuracy from 60% to 100% did not
change this. No label-free competence signal, from any of four families, met a
prespecified criterion for predicting segmentation failure across clinical
sources.

These findings concern validation protocol rather than any individual method.
The practices implicated are common, the failures they produce are invisible
from within a single benchmark, and the corrections required are inexpensive.

---

## 10. Declarations

### Data availability

All code, both generators, all twenty-two segmentation methods, trained models,
fitted thresholds, and per-image result files are available at
`github.com/Lamia-Islam/surgical-vision-ros2` under the MIT licence and archived
at Zenodo (DOI: [INSERT]). Synthetic benchmarks are not distributed as image
files; they are regenerated deterministically from the seeds stated in the text
using the released generators.

> **RECOVERY NOTE:** This statement is currently FALSE. The repository contains
> only the Track A (SoftwareX) code. None of the 22 methods, the generators,
> the U-Net, or the result files are present. This statement cannot be made
> until the code is rebuilt and pushed.

The clinical datasets analysed are third party and were not generated by the
authors. Both are publicly available at
`github.com/uwm-bigdata/wound-segmentation`, maintained by the Big Data
Analytics and Visualization Laboratory, University of Wisconsin–Milwaukee. The
Foot Ulcer Segmentation Challenge dataset originates from the AZH Wound and
Vascular Center and was released for the MICCAI 2021 challenge
(`fusc.grand-challenge.org`); it is described in [5]. The Medetec subset derives
from the Medetec wound database (`medetec.co.uk`). The authors obtained no
special access privileges to either dataset. No clinical images are
redistributed.

### Competing interests

The authors declare no competing interests.

### Ethics approval

This study used only publicly available, previously published, de-identified
image datasets and required no new ethical approval.

### Preprint

An earlier version of part of this work was posted as a preprint
(doi:10.21203/rs.3.rs-10362132/v1). The present manuscript reports different
claims and a substantially expanded evaluation.

### Funding

[Complete as applicable.]

---

## Figure specifications

**Figure 1 — Pipeline architecture.** Four ROS 2 nodes with topic names and
message formats; the mapping from ABCDE total to verdict to autonomy level shown
as an inset table. Single column.

**Figure 2 — Synthetic against clinical Dice.** Two panels, generator A left,
generator B right. Each panel: 22 points, synthetic Dice on x, clinical Dice on
y, one marker shape per clinical dataset or two sub-panels. Annotate ρ and p in
each panel. Label the U-Net point in both. Reference diagonal in grey.

This is the paper's central exhibit. It should be legible at a glance: points
scattered without structure in the left panel, trending upward in the right.

**Figure 3 — Ablation.** Horizontal bars, mean ρ per ablation variant, ordered
by effect size, with the full-generator value as a reference line and zero
marked. The texture bar crosses zero; this should be visually unmistakable.

**Figure 4 — Within-dataset against cross-source competence prediction.** Two
panels: predicted against actual Dice under within-Medetec five-fold
cross-validation (ρ = 0.948), and the same model predicting FUSeg (ρ = 0.316).
Identical axes.

**Supplementary Table S1.** All 22 methods × 4 benchmarks, mean Dice with
bootstrap 95% confidence intervals. Generate from
`results/rank_validity_n22.json`.

---

## Pre-submission checklist (as written in August)

- [ ] Complete Table 4 from `results/rank_validity_n22.json`
- [ ] Fill remaining `[SEARCH]` citations in Section 2 (Batch 1 Rev A lists them
      in priority order)
- [ ] Verify reference [4] journal version (BIAS, *Medical Image Analysis* 2020),
      [18] author list, [19] volume and pages
- [ ] Check whether arXiv:2409.11968 has a peer-reviewed version
- [ ] Push repository, tag release, mint Zenodo DOI, insert into Section 10
- [ ] Generate Figures 1–4 and Supplementary Table S1
- [ ] Confirm JIIM abstract word limit and structured-abstract headings
- [ ] Select subscription (non–open access) route at acceptance

> **RECOVERY NOTE:** Items 1, 5 and 6 are now blocked on rebuilding the code,
> not merely on running it.
