# MANUSCRIPT — BATCH 1
## Title, Abstract, Introduction, Related Work

> RECOVERED from conversation f7ee7590-a160-40ce-bc6c-e90dd81e1c63, turn 67.
> Original file path in the lost container: `/mnt/user-data/outputs/MANUSCRIPT_BATCH1.md`
> NOTE: Section 2 (Related Work) here is the PLACEHOLDER version. It is superseded
> by BATCH1_REVA_related_work.md, which has verified citations.

**Target:** Journal of Imaging Informatics in Medicine (Springer). IF 3.1,
CiteScore 7.3, SCIE/Scopus/PubMed indexed. Hybrid — select the subscription
route at acceptance for zero cost.

**Framing note.** JIIM states that papers reporting *validation and
implementation* of informatics technologies take precedence over stand-alone
evaluation of image analysis tools. Every framing choice below serves that: the
paper is about how benchmark design determines validation outcomes, evidenced
through a deployed implementation. It is not pitched as a comparison of
segmentation algorithms.

---

## Title

**Benchmark design determines validation outcomes in wound image segmentation:
evidence from two synthetic benchmarks, two clinical datasets, and a deployed
autonomy pipeline**

Alternatives if the above runs long for the submission system:

- *Validation outcomes in wound segmentation depend on synthetic benchmark
  design: a study across 22 methods and four datasets*
- *When synthetic validation misleads: background texture determines whether
  wound segmentation benchmarks predict clinical performance*

---

## Abstract

**Structured, per JIIM convention.**

**Purpose.** Synthetic images are widely used to validate medical image
segmentation when annotated clinical data is scarce. We ask whether the
validation outcomes such benchmarks produce transfer to clinical data, which
design properties determine transfer, and what follows for a deployed system
that acts on segmentation output.

**Methods.** Twenty-two segmentation methods spanning thresholding, clustering,
region growing, active contours, per-pixel learned classifiers and a
convolutional neural network were scored by Dice against ground truth on four
benchmarks: two independently constructed synthetic wound generators, and two
independent clinical wound datasets (FUSeg, n = 1,010; Medetec, n = 152).
Method rankings were compared across benchmarks using Spearman correlation with
permutation testing. The synthetic generators were ablated feature by feature to
isolate the design property governing transfer. Consequences were measured in a
ROS 2 pipeline that gates manipulator autonomy on segmentation-derived wound
severity.

**Results.** The two synthetic benchmarks ranked methods in significantly
opposite orders (ρ = −0.526, p = 0.014), while the two clinical datasets agreed
(ρ = 0.560, p = 0.008). One synthetic benchmark was significantly *negatively*
correlated with clinical ranking (ρ = −0.466, p = 0.033). Ablation identified
background texture as the determining property: its removal reduced mean rank
correlation by 0.532 and reversed its sign, whereas removing grade-independent
lesion size, off-centre placement, clutter or irregular lesion shape cost at
most 0.079. A convolutional network achieving the best clinical Dice (0.670
FUSeg, 0.446 Medetec) scored 0.080 on the uniform-background benchmark. In the
deployed pipeline, full autonomy was granted on 27–62% of frames where
segmentation had failed, and threshold recalibration that raised synthetic
classification accuracy from 60% to 100% did not reduce this.

**Conclusion.** Synthetic benchmarks that composite a lesion onto uniform
background make foreground detection and lesion segmentation the same task, and
produce validation outcomes that do not transfer and may invert. Validation
protocols should require textured, non-uniform backgrounds, cross-source
evaluation, and direct measurement of downstream safety rather than inference
from segmentation accuracy.

**Keywords.** validation methodology; wound segmentation; synthetic data;
benchmark design; cross-dataset generalisation; imaging informatics; robotic
autonomy

[Word count ≈ 300. Check JIIM's abstract limit before submission; trim the
Results paragraph first if needed.]

---

## 1. Introduction

Validation is the point at which an imaging informatics tool earns clinical
trust. It is also the step most often performed on data that is convenient
rather than representative. Annotated clinical images are expensive to obtain,
constrained by consent and governance, and unevenly distributed across
populations and care settings. Synthetic images — generated or composited —
offer an appealing substitute, and are used throughout medical image analysis to
develop pipelines, tune parameters, and select among candidate methods before
clinical data is available, or in place of it. [VERIFY: 2–3 citations
establishing prevalence of synthetic data for validation in medical imaging;
a recent survey would be ideal.]

Absolute performance on synthetic data is not usually claimed to transfer.
Nobody expects synthetic Dice to equal clinical Dice. What is assumed, often
tacitly, is that *relative* performance transfers: that the method which
performs better synthetically will perform better clinically. This is the
assumption that makes synthetic benchmarks useful for method selection, and it
is rarely tested directly.

We test it, and it fails. Worse, it fails in a way that is invisible from inside
a single benchmark: two reasonable synthetic designs produce opposite
conclusions about which method to deploy, and only comparison against clinical
data reveals which is which.

The setting is wound image segmentation, chosen because it has an unusually
clean structure for this question. Two independent clinical datasets with expert
annotations are publicly available, from different clinics with different
cameras and framing conventions. Segmentation has a single unambiguous quality
metric. And the task has a well-defined downstream consumer: systems that act on
wound assessment, including the variable-autonomy robotic pipeline we
instrument here.

That downstream link matters for the argument. A validation failure that stays
inside a benchmark table is a methodological curiosity. The same failure
propagating into a system that decides how much autonomy a manipulator receives
is a safety property. We measure both.

### 1.1 Contributions

1. **Synthetic benchmark design determines the validation outcome.** Two
   independently constructed synthetic wound benchmarks rank 22 segmentation
   methods in significantly opposite orders (ρ = −0.526, p = 0.014), while two
   independent clinical datasets agree (ρ = 0.560, p = 0.008). One synthetic
   benchmark is significantly *negatively* correlated with clinical ranking.
2. **Background texture is the property responsible.** Feature-wise ablation
   shows removing background texture costs 0.532 in mean rank correlation and
   reverses its sign, while lesion size, placement, clutter and shape realism
   cost at most 0.079 each. We give the mechanism: uniform backgrounds collapse
   foreground detection and lesion segmentation into the same task.
3. **The failure propagates into a deployed system.** A ROS 2 pipeline gating
   manipulator autonomy on segmentation-derived severity grants full autonomy on
   27–62% of frames where segmentation has failed, and threshold recalibration
   that appears decisive under synthetic validation does not change this.
4. **Competence monitoring does not rescue it, and conventional validation hides
   that too.** Seventeen label-free competence signals fail to predict
   segmentation error across clinical sources; a learned combination reaches
   ρ = 0.948 under within-dataset cross-validation and ρ = 0.316 across sources.
5. **Released artefacts.** Both generators, all 22 methods, the pipeline, and
   every evaluation script, under MIT licence with archived DOI.

### 1.2 Scope and what this paper does not claim

We do not claim that synthetic data is unsuitable for validation. One of our two
generators predicts clinical ranking significantly on both clinical datasets;
the other does not. The claim is narrower and more actionable: a specific and
common design choice determines which of these you get, and the difference is
undetectable without clinical comparison.

We also do not claim a clinically deployable wound assessment system. The
clinical datasets are chronic diabetic foot ulcers; the robotic pipeline is
evaluated in simulation without clinical severity labels. The pipeline serves as
an instrumented consumer of segmentation output, which is what the argument
requires.

---

## 2. Related work (PLACEHOLDER — superseded by Rev A)

[Structure and content are set out below. Citation slots marked [VERIFY] must
be filled from a literature search — do not accept any reference in this draft
that is not written out in full. Target 25–35 references total.]

### 2.1 Synthetic and simulated data for validation in medical imaging

Synthetic imagery is used across medical image analysis to supplement scarce
annotated data, spanning physics-based simulation, compositing of segmented
lesions onto backgrounds, and generative modelling. [VERIFY: 4–6 citations. Aim
for one survey of synthetic data in medical imaging, one or two examples of
compositing-based generation for skin or wound imagery, and one or two examples
of generative augmentation. Search terms: "synthetic data medical image
segmentation survey", "lesion compositing augmentation dermoscopy", "GAN
augmentation wound segmentation".]

Work assessing whether synthetic training data transfers to real performance
exists primarily in the augmentation setting, where the question is whether
adding synthetic images improves a model. [VERIFY: 2–3 citations.] The distinct
question we address — whether synthetic *validation* preserves the *ordering* of
candidate methods — has received less attention, and we are not aware of a study
that ablates generator design properties against clinical rank agreement.
[VERIFY: search "benchmark validity segmentation ranking correlation synthetic
real" to confirm this gap before asserting it. If prior work exists, this
becomes a positioning paragraph rather than a gap claim.]

### 2.2 Wound image segmentation

Classical wound segmentation has drawn on thresholding, colour-space clustering,
region growing and active contours, with colour features motivated by the
clinical red–yellow–black tissue classification scheme. [VERIFY: 3–5 citations
for classical wound segmentation; search "wound segmentation colour features
review", "red yellow black wound classification".]

Deep segmentation now dominates. The Foot Ulcer Segmentation Challenge (FUSeg)
established a public benchmark with expert annotations, and reported Dice for
convolutional architectures is in the range 0.89–0.93.

> Wang C, Anisuzzaman DM, Williamson V, Dhar MK, Rostami B, Niezgoda J,
> Gopalakrishnan S, Yu Z. Fully automatic wound segmentation with deep
> convolutional neural networks. *Scientific Reports*. 2020;10:21897.
> doi:10.1038/s41598-020-78799-w

[VERIFY: add the FUSeg challenge report and 2–3 recent high-performing methods
on it, to substantiate the 0.89–0.93 range.]

### 2.3 Cross-dataset generalisation and evaluation validity

Performance drops under distribution shift are well documented in medical
imaging, and single-dataset evaluation is known to overstate generalisation.
[VERIFY: 3–4 citations; search "domain shift medical imaging generalisation
multi-site", "dataset shift segmentation clinical".] Our within-dataset versus
cross-source result (Section 7) is an instance of this in the specific setting
of competence estimation, where we find the discrepancy is large enough to
invert a practical conclusion.

Related critiques concern metric selection and benchmark construction in medical
image analysis. [VERIFY: search "metrics reloaded", "pitfalls medical image
analysis metrics", "benchmarking segmentation challenges analysis". There is a
substantial recent literature here and it is the closest prior work to this
paper — position carefully rather than claiming novelty too broadly.]

### 2.4 Uncertainty and competence estimation

Signals proposed for detecting model failure without labels fall into families
we test directly in Section 7: softmax or sigmoid confidence and its
calibration; ensemble disagreement; test-time-augmentation instability; and
geometric or structural plausibility of the prediction.

> Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural
> networks. *ICML*. 2017:1321–1330.

> Hendrycks D, Gimpel K. A baseline for detecting misclassified and
> out-of-distribution examples in neural networks. *ICLR*. 2017.

> Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive
> uncertainty estimation using deep ensembles. *NeurIPS*. 2017:6402–6413.

[VERIFY: add 2–3 medical-imaging-specific uncertainty papers, and at least one
on test-time augmentation for segmentation uncertainty; search "test time
augmentation uncertainty medical image segmentation".]

### 2.5 Perception-conditioned variable autonomy

Variable autonomy adjusts the division of control between operator and machine
according to conditions the system perceives, and has been studied in surgical
robotics, teleoperation and assistive robotics. [VERIFY: 4–5 citations; search
"variable autonomy shared control surgical robotics", "levels of autonomy
surgical robot", "adjustable autonomy teleoperation".] The safety case for such
systems rests on perception being either accurate or known to be inaccurate. We
are not aware of prior work measuring, on annotated clinical imagery, how often
such a gate grants full autonomy while its perception has in fact failed.
[VERIFY before asserting.]

---

## Reference list — verified entries only

These are canonical and safe to cite as written. Everything else in Section 2
must be filled by search.

**Segmentation methods used in the comparison**

> Otsu N. A threshold selection method from gray-level histograms. *IEEE
> Transactions on Systems, Man, and Cybernetics*. 1979;9(1):62–66.

> Canny J. A computational approach to edge detection. *IEEE Transactions on
> Pattern Analysis and Machine Intelligence*. 1986;8(6):679–698.

> Vincent L, Soille P. Watersheds in digital spaces: an efficient algorithm
> based on immersion simulations. *IEEE Transactions on Pattern Analysis and
> Machine Intelligence*. 1991;13(6):583–598.

> Comaniciu D, Meer P. Mean shift: a robust approach toward feature space
> analysis. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
> 2002;24(5):603–619.

> Chan TF, Vese LA. Active contours without edges. *IEEE Transactions on Image
> Processing*. 2001;10(2):266–277.

> Rother C, Kolmogorov V, Blake A. GrabCut: interactive foreground extraction
> using iterated graph cuts. *ACM Transactions on Graphics*.
> 2004;23(3):309–314.

> Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for
> biomedical image segmentation. *Medical Image Computing and Computer-Assisted
> Intervention (MICCAI)*. 2015:234–241.

**Metric**

> Dice LR. Measures of the amount of ecologic association between species.
> *Ecology*. 1945;26(3):297–302.

**Data**

> Wang C, Anisuzzaman DM, Williamson V, Dhar MK, Rostami B, Niezgoda J,
> Gopalakrishnan S, Yu Z. Fully automatic wound segmentation with deep
> convolutional neural networks. *Scientific Reports*. 2020;10:21897.

[VERIFY: Medetec wound database — cite the database itself
(https://www.medetec.co.uk/) and check whether a citable publication by
S. Thomas exists. Also cite the FUSeg challenge (https://fusc.grand-challenge.org/)
and confirm whether a challenge report paper is available.]

**Uncertainty**

> Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural
> networks. *ICML*. 2017:1321–1330.

> Hendrycks D, Gimpel K. A baseline for detecting misclassified and
> out-of-distribution examples in neural networks. *ICLR*. 2017.

> Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive
> uncertainty estimation using deep ensembles. *NeurIPS*. 2017:6402–6413.
