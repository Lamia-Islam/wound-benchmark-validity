# MANUSCRIPT — BATCH 2 EDITS + BATCH 3 (RESULTS)

> RECOVERED from conversation f7ee7590-a160-40ce-bc6c-e90dd81e1c63, turn 73.
> Original file path in the lost container: `/mnt/user-data/outputs/MANUSCRIPT_BATCH3.md`

House rule applied throughout: the body states what was done and what was
found. Caveats live in Limitations. Nothing is hidden, it is placed where
reviewers expect it.

---

# PART A — Replacements for Batch 2

### Replace Section 3.8 with:

> **3.8 Size of the method pool**
>
> Spearman rank correlation over n = 6 methods requires |ρ| ≥ 0.83 to reach
> p < 0.05; over n = 22 it requires |ρ| ≥ 0.42. A pool of twenty-two is
> therefore the minimum at which moderate rank agreement can be distinguished
> from chance, and comparisons involving a handful of methods are not powered
> to support ranking conclusions.

### Replace the final paragraph of Section 4.3 with:

> Label encodings differ between the two datasets: Medetec masks are 0/1 valued
> and FUSeg masks 0/255. Both are binarised on any non-zero value.

### Replace Section 4.4's closing sentence with:

> The pipeline switches between levels 1 and 2 of the six-level taxonomy for
> medical robot autonomy [17]. It is an instrumented consumer of segmentation
> output, included so that the downstream consequences of a validation failure
> can be measured directly.

---

# PART B — Section 6: Results

## 6.1 Rank agreement across benchmarks

All twenty-two methods were scored by Dice against ground truth on four
benchmarks. Rank agreement between benchmarks is given in Table 3.

**Table 3.** Spearman rank correlation between benchmarks, n = 22 methods.
Permutation test, 20,000 permutations.

| comparison | ρ | p |
|---|---|---|
| generator A vs Medetec | −0.121 | 0.588 |
| generator A vs FUSeg | −0.466 | 0.033 |
| generator B vs Medetec | 0.439 | 0.040 |
| generator B vs FUSeg | 0.560 | 0.007 |
| **generator A vs generator B** | **−0.526** | **0.014** |
| **Medetec vs FUSeg** | **0.560** | **0.008** |

Two results carry the section. The two clinical datasets, which differ in
originating clinic, camera, resolution and framing, agree significantly
(ρ = 0.560, p = 0.008). The two synthetic benchmarks disagree significantly
(ρ = −0.526, p = 0.014). Agreement between independent clinical sources exceeds
agreement between synthetic benchmarks, and the synthetic benchmarks are
negatively related to one another.

Generator A does not predict clinical ranking. Its correlation with FUSeg is
significantly negative (ρ = −0.466, p = 0.033), meaning that on this benchmark a
method's synthetic rank is an inverted indicator of its clinical rank. Generator
B predicts clinical ranking significantly on both clinical datasets, at a
magnitude comparable to the agreement between the clinical datasets themselves.

## 6.2 A single-method illustration

The convolutional network achieves the highest Dice of all twenty-two methods on
both clinical datasets (0.670 FUSeg, 0.446 Medetec) and 0.680 on generator B. On
generator A it achieves **0.080**, placing it near the bottom of the pool.

The same weights, the same post-processing, and the same metric produce a method
that is best in class on clinical data and near-worst on a uniform-background
synthetic benchmark. Selecting a method on generator A would eliminate the
strongest available option.

Table 4 gives the contrast across the pool.

**Table 4.** Selected methods, mean Dice, with bootstrap 95% CI (5,000
resamples).

| method | generator A | generator B | Medetec | FUSeg |
|---|---|---|---|---|
| U-Net (117k params) | 0.080 | 0.680 | 0.446 | 0.670 |
| logistic regression (14 features) | 0.166 | — | 0.451 | 0.452 |
| GrabCut | 0.935 | — | — | — |
| Otsu on a\* | 0.924 | — | — | — |
| mean-shift + Otsu on a\* | 0.920 | — | — | — |
| HSV colour distance | 0.955 | — | 0.258 | 0.040 |
| gradient boosting (14 features) | 0.011 | — | 0.370 | — |

> [Complete this table from `results/rank_validity_n22.json` before submission.
> Full per-method values for all four benchmarks are in the released result files;
> consider a supplementary table with all 22 × 4 entries.]
>
> **RECOVERY NOTE:** `results/rank_validity_n22.json` was lost with the container.
> The full 22 × 4 matrix must be regenerated.

The three highest-scoring methods on generator A are all background-referenced
or global-threshold approaches. The two lowest are learned classifiers. On both
clinical datasets the ordering is approximately reversed.

## 6.3 Which design property governs rank preservation

Generator B was ablated one feature at a time, each reverting to generator A's
corresponding assumption, with all other features and the random seed held
fixed.

**Table 5.** Ablation of generator B. Rank correlation against each clinical
dataset, n = 22 methods.

| ablated feature | ρ Medetec | ρ FUSeg | mean ρ | change |
|---|---|---|---|---|
| none (full generator B) | 0.356 | 0.510 | 0.433 | — |
| grade-independent lesion size | 0.291 | 0.430 | 0.360 | −0.073 |
| off-centre placement | 0.260 | 0.448 | 0.354 | −0.079 |
| clutter | 0.303 | 0.436 | 0.369 | −0.064 |
| irregular lesion shape | 0.372 | 0.534 | 0.453 | +0.020 |
| **textured background** | **0.070** | **−0.268** | **−0.099** | **−0.532** |

Removing background texture reduces mean rank correlation by 0.532 and reverses
the sign of the FUSeg correlation, reproducing generator A's behaviour. The
largest effect of any other feature is 0.079, approximately one seventh as
large. Reverting lesion shape from irregular blobs to ellipses slightly improves
rank agreement, indicating that lesion shape realism is not material to this
outcome.

**Mechanism.** With a uniform background, the set of non-background pixels and
the set of lesion pixels coincide. Any method that separates figure from ground
therefore attains a high Dice, whether or not it localises pathology.
Corner-referenced and global-threshold methods are rewarded accordingly, while
methods that search for lesion tissue *within* skin are penalised for correctly
declining to claim the entire foreground. Introducing background texture
separates the two pixel sets and restores the intended task.

This follows from the compositing strategy rather than from any particular
generator. A synthetic benchmark constructed by compositing a lesion onto
uniform background will reward foreground detection regardless of how carefully
lesion size, shape and placement are modelled.

## 6.4 Downstream consequences for the autonomy gate

**Table 6.** Autonomy decisions on clinical images, learned segmentation.

| dataset | thresholds | full | reduced | hold | unsafe-optimistic |
|---|---|---|---|---|---|
| Medetec (82/152 frames failed) | hand-tuned | 82 | 35 | 35 | 28/82 = 0.34 |
| Medetec | recalibrated | 60 | 61 | 31 | 22/82 = 0.27 |
| FUSeg (66/120 frames failed) | hand-tuned | 84 | 21 | 15 | 40/66 = 0.61 |
| FUSeg | recalibrated | 88 | 18 | 14 | 41/66 = 0.62 |

On FUSeg, 62% of frames on which segmentation failed received full autonomy.

Threshold recalibration is the natural remedy and does not deliver. Fitting
thresholds on held-out synthetic seeds raises three-grade classification
accuracy on the synthetic benchmark from 60.0% to 100.0%, with a cross-seed mean
of 99.7% (SD 0.8) over eight unseen seeds. On the clinical safety metric the
unsafe-optimistic rate moves from 0.61 to 0.62. An intervention that appears
decisive under synthetic validation produces no clinical safety benefit.

The gate also inverts under improvement in perception. Replacing the
adaptive-threshold segmentation with HSV colour distance raises synthetic grade
accuracy from 2.2% to 60.0% while driving the intermediate grade from scattered
predictions to 0 of 15 correct: every borderline case classified as healthy.
Thresholds fitted against one segmentation's feature distributions do not
transfer to another, and the failure direction is toward proceeding.

End-to-end latency is 40.4 ms (24.8 Hz) on 512 × 512, single CPU core, excluding
ROS 2 transport, of which gate evaluation is 1.18 ms.

## 6.5 Competence signals

Seventeen label-free signals were evaluated against per-image Dice under the
protocol in Section 5.5.

**Table 7.** Best signal from each family.

| family | signal | ρ Medetec | ρ FUSeg |
|---|---|---|---|
| model confidence | fraction of decisive pixels | 0.435 | 0.411 |
| ensemble disagreement | mean IoU to learned mask | 0.666 | 0.328 |
| geometric plausibility | predicted area fraction | 0.633 | 0.314 |
| TTA instability | mask area coefficient of variation | −0.498 | −0.204 |

No signal met the |ρ| ≥ 0.6 criterion on both datasets. Several reversed sign
between datasets. Every family correlates moderately on Medetec and weakly on
FUSeg.

A learned combination of all seventeen signals does not transfer across clinical
sources.

**Table 8.** Cross-source prediction of per-image Dice from all seventeen
signals.

| training source | test source | ρ | R² | failure-detection AUROC |
|---|---|---|---|---|
| Medetec | FUSeg | 0.316 | −0.345 | 0.630 |
| FUSeg | Medetec | 0.337 | 0.062 | 0.687 |

Five-fold cross-validation within Medetec yields ρ = 0.948. The same features and
model evaluated across clinical sources yield ρ = 0.316. Within-dataset
validation therefore indicates a competence monitor of high accuracy, while
cross-source evaluation of the identical model indicates performance close to
chance.

Model confidence additionally increases on out-of-distribution input: mean
certainty is 0.942 (SD 0.010) on synthetic images against 0.902 on Medetec and
0.910 on FUSeg.

## 6.6 Cross-source performance of the learned baseline

**Table 9.** Learned baseline, effect of decision threshold.

| split | Dice @ 0.5 | Dice @ 0.9 | median @ 0.9 |
|---|---|---|---|
| FUSeg validation (n = 150) | 0.203 | 0.452 | 0.469 |
| Medetec (n = 152) | 0.557 | 0.451 | 0.482 |

The threshold was selected on held-out FUSeg training images and applied
unchanged to both clinical datasets. Under it, the two independent clinical
sources converge to within 0.001. Predicted lesion fraction on FUSeg tracks the
reference at 0.0136 against 0.0116.

At the uncorrected 0.5 threshold the two datasets differ by 0.354. The
difference is attributable to lesion fraction: Medetec's median lesion fraction
is 0.098 against FUSeg's 0.0068, so a threshold suited to one prior
over-segments under the other. Reporting Medetec alone at the uncorrected
threshold would overstate cross-source performance by approximately 24%.

Absolute performance remains below published deep-learning results on FUSeg,
which report Dice in the range 0.89–0.93.

---

# PART C — Section 8: Limitations

All caveats consolidated here.

**Clinical domain.** Both clinical datasets comprise chronic diabetic foot
ulcers. The autonomy pipeline is motivated by intraoperative wound assessment,
a domain for which no comparable annotated public dataset exists. The findings
concern the structure of perception-gated systems and the validity of synthetic
benchmarks; they are not validation of a surgical device, and transfer to
intraoperative imagery is not demonstrated.

**Gate correctness.** Neither clinical dataset carries severity labels, so
whether an assigned autonomy level is clinically appropriate cannot be assessed.
Only gate safety, defined against segmentation quality, is measured.

**No physical system or clinical evaluation.** The pipeline is evaluated in
simulation. There is no physical manipulator, no surgical task, and no clinician
in the loop. Reported latency is compute latency and excludes ROS 2 transport.

**Generator B's design dependency.** Generator B's design targets were derived
from measured properties of the same clinical datasets used to evaluate it. The
targets are coarse — lesion fraction range, placement distribution, presence of
background texture — rather than fitted, and the ablation identifies a
structural mechanism rather than a correlation. The dependency nonetheless
constrains how strongly the result should be read, and replication with a
generator designed against independent clinical data would strengthen it.

**Number of benchmarks.** Two synthetic generators and two clinical datasets is
a narrow basis for generalisation. Of four generator-to-clinical correlations,
three reach p < 0.05 and one does not.

**Convolutional baseline.** The U-Net is small, trained from random
initialisation on CPU without pretrained encoders or a full augmentation
schedule, and does not approach published FUSeg performance. It is included to
extend the method pool beyond classical approaches, not as a competitive
segmentation method. Whether the ranking failure holds for larger pretrained
architectures is untested.

**Ablation granularity.** Background texture was ablated as a binary presence or
absence. A dose-response analysis over texture amplitude would establish the
relationship more precisely than the present on/off comparison.

**Single metric.** Dice is used throughout so that benchmarks remain comparable.
Prior work shows that metric selection is itself a source of ranking instability
[1–3]; whether the reported rank disagreements persist under boundary-based or
detection-based metrics is not tested here.

---

## Notes for Batch 4 (carried forward)

- The Discussion should state the three compounding practices (synthetic
  validation, within-dataset competence validation, threshold coupling) and
  derive the four recommendations from the measurements rather than from
  general principle.
- Recommendations should be phrased as protocol requirements, since JIIM's
  audience is validation-oriented.
- Figures: (1) node graph and topic flow; (2) paired scatter of synthetic
  against clinical Dice, 22 points per panel, one panel per generator;
  (3) within-dataset against cross-source scatter for the competence
  combination. Figure 2 is the paper's central exhibit.
