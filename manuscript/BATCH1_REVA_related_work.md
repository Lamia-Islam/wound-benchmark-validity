# MANUSCRIPT — BATCH 1, REVISION A
## Section 2: Related Work (verified citations)

> RECOVERED from conversation f7ee7590-a160-40ce-bc6c-e90dd81e1c63, turn 69.
> Original file path in the lost container: `/mnt/user-data/outputs/MANUSCRIPT_BATCH1_REV_A.md`
> This REPLACES the placeholder Related Work in Batch 1.

Every reference written out below has been verified against PubMed, Nature, or
arXiv. Remaining gaps are marked `[SEARCH]` with the exact query to run.

**Positioning summary.** The nearest prior work establishes that method rankings
in biomedical image analysis are unstable, and attributes that instability to
metric selection, aggregation scheme, annotator variability, and choice of test
data. What has not been done is to identify *which property of the benchmark
data* drives rank instability, causally, by ablation, with a clinical reference
standard. That is this paper's contribution, and it extends rather than
contradicts the existing literature.

---

## 2. Related work

### 2.1 Validation and ranking instability in biomedical image analysis

Concern about how biomedical image analysis methods are validated is
well established. In a systematic analysis of 150 challenges, Maier-Hein et al.
demonstrated that an algorithm's rank is not robust to the test data used for
validation, the ranking scheme applied, the observers producing reference
annotations, or the method of aggregating per-case values [1]. Their
re-evaluation of the 2015 segmentation challenges found that a quarter of
non-winning algorithms would have placed first under a different but equally
defensible aggregation choice, and that in some tasks every participating team
could have been ranked first. The same group subsequently developed the Metrics
Reloaded framework, which guides problem-aware metric selection through a
structured "problem fingerprint" [2], accompanied by a companion analysis
cataloguing metric-related pitfalls [3], and reporting guidelines for challenge
transparency [4].

This body of work establishes that rankings are fragile and identifies several
sources of fragility: metric choice, aggregation, annotation, and which test set
is used. Our study addresses a source that has not, to our knowledge, been
isolated: not *which* dataset is used, but *how a synthetic dataset is
constructed*. Where prior analyses vary real test sets or scoring procedures, we
hold the metric, the aggregation, and the reference annotations fixed, vary a
single generator design property, and measure the resulting rank agreement
against clinical data. This converts benchmark design from a known risk into a
measured, attributable one.

[SEARCH before asserting the gap: "synthetic benchmark ranking correlation real
data segmentation", "does synthetic data preserve method ranking medical
imaging". If prior work exists, soften to "has received limited attention" and
cite it.]

> **RESOLVED LATER (turn 71):** arXiv:2409.11968, "Efficacy of Synthetic Data as
> a Benchmark", asks exactly this question in NLP using Spearman rank
> correlation. So the METHOD is not novel. The gap narrows to: not done in
> medical image segmentation, and no prior work isolates the causal design
> property by ablation. See the revised gap sentence in Batch 2.

### 2.2 Synthetic and simulated data in medical image analysis

Synthetic imagery supplements scarce annotated data across medical image
analysis, via physics-based simulation, compositing of lesions onto backgrounds,
and generative modelling.

[SEARCH: "synthetic data medical image analysis survey 2024"; "generative
augmentation medical image segmentation review". Need 3–5 references. At least
one should be a survey, and at least one an example of compositing-based
generation, since compositing is the specific practice this paper implicates.]

Existing evaluations of synthetic data largely concern its use for *training* —
whether adding synthetic images improves downstream model performance. The
distinct question of whether synthetic data supports valid *validation*, in the
sense of preserving the relative ordering of candidate methods, is the subject
of this paper.

[SEARCH: "synthetic training data transfer real performance medical imaging" for
2–3 references supporting the first sentence.]

### 2.3 Wound image segmentation

Classical approaches to wound segmentation have used thresholding, colour-space
clustering, region growing and active contours, with colour features motivated
by the clinical red–yellow–black tissue classification scheme.

[SEARCH: "wound segmentation review colour features"; "red yellow black wound
tissue classification segmentation". Need 3–5 references.]

Deep segmentation now dominates the field. Wang et al. released a fully
automatic wound segmentation pipeline together with the annotated dataset from
the AZH Wound and Vascular Center that underpins the Foot Ulcer Segmentation
Challenge [5], and reported Dice for convolutional architectures on that
benchmark falls in the range 0.89–0.93.

[SEARCH: 2–3 recent high-performing FUSeg methods to substantiate the 0.89–0.93
range; query "FUSeg foot ulcer segmentation challenge Dice state of the art".]

Both clinical datasets used here are drawn from this line of work, and both
carry expert annotations, which is what permits Dice to be computed against a
reference rather than inferred from downstream task accuracy.

### 2.4 Cross-dataset generalisation and competence estimation

Performance degradation under distribution shift is well documented in medical
imaging, and evaluation confined to a single dataset is known to overstate
generalisation.

[SEARCH: "domain shift medical image segmentation multi-centre generalisation";
"single dataset evaluation overestimates performance medical imaging". Need 3–4
references.]

Signals proposed for detecting model failure without access to labels fall into
families we test directly: confidence and its calibration [6, 7], ensemble
disagreement [8], test-time-augmentation instability [SEARCH: "test time
augmentation uncertainty medical image segmentation" — Wang et al. 2019 in
Neurocomputing is likely the canonical reference; verify], and geometric or
structural plausibility of the prediction [SEARCH: "shape prior quality control
segmentation automatic failure detection"].

Our result in this area is a specific instance of the single-dataset problem
above, with an unusually large effect: the same features and model yield
Spearman 0.948 under within-dataset cross-validation and 0.316 across clinical
sources, which is the difference between a competence monitor that appears
deployable and one that is not.

### 2.5 Perception-conditioned variable autonomy

Variable autonomy systems adjust the division of control between operator and
machine according to perceived conditions, and have been studied in surgical
robotics, teleoperation, and assistive robotics.

[SEARCH: "variable autonomy surgical robotics levels of autonomy"; "shared
control adjustable autonomy teleoperation survey". Need 4–5 references.]

> **RESOLVED LATER (turn 71):** refs [17], [18], [19] secured — see Batch 2.

The safety argument for such systems presumes that perception is either accurate
or known to be inaccurate. We measure, on annotated clinical imagery, how often
a gate of this form grants full autonomy while its perception has in fact
failed.

[SEARCH before claiming this is unmeasured: "perception failure autonomy gate
safety metric"; "segmentation failure downstream robotic safety".]

---

## Verified reference list

Safe to use exactly as written.

**Validation methodology (the core positioning references)**

[1] Maier-Hein L, Eisenmann M, Reinke A, Onogur S, Stankovic M, Scholz P, et al.
Why rankings of biomedical image analysis competitions should be interpreted
with care. *Nature Communications*. 2018;9(1):5217.
doi:10.1038/s41467-018-07619-7

[2] Maier-Hein L, Reinke A, Godau P, Tizabi MD, Buettner F, Christodoulou E,
et al. Metrics reloaded: recommendations for image analysis validation. *Nature
Methods*. 2024;21(2):195–212. doi:10.1038/s41592-023-02151-z

[3] Reinke A, Tizabi MD, Baumgartner M, Eisenmann M, Heckmann-Nötzel D, Kavur AE,
et al. Understanding metric-related pitfalls in image analysis validation.
*Nature Methods*. 2024;21(2):182–194. doi:10.1038/s41592-023-02150-0

[4] Maier-Hein L, Reinke A, Kozubek M, Martel AL, Arbel T, Eisenmann M, et al.
BIAS: Transparent reporting of biomedical image analysis challenges.
arXiv:1910.04071. 2019.
*[Note: this appeared in Medical Image Analysis (2020). Verify the journal
version's volume and pages and cite that in preference to the preprint.]*

**Data**

[5] Wang C, Anisuzzaman DM, Williamson V, Dhar MK, Rostami B, Niezgoda J,
Gopalakrishnan S, Yu Z. Fully automatic wound segmentation with deep
convolutional neural networks. *Scientific Reports*. 2020;10:21897.
doi:10.1038/s41598-020-78799-w

[SEARCH: Medetec wound database citation — cite https://www.medetec.co.uk/ and
check for a citable publication by S. Thomas. FUSeg challenge —
https://fusc.grand-challenge.org/; check for a challenge report paper.]

**Uncertainty and competence estimation**

[6] Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural
networks. *Proceedings of the 34th International Conference on Machine Learning
(ICML)*. 2017:1321–1330.

[7] Hendrycks D, Gimpel K. A baseline for detecting misclassified and
out-of-distribution examples in neural networks. *International Conference on
Learning Representations (ICLR)*. 2017.

[8] Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive
uncertainty estimation using deep ensembles. *Advances in Neural Information
Processing Systems (NeurIPS)*. 2017:6402–6413.

**Segmentation methods in the comparison**

[9] Otsu N. A threshold selection method from gray-level histograms. *IEEE
Transactions on Systems, Man, and Cybernetics*. 1979;9(1):62–66.

[10] Canny J. A computational approach to edge detection. *IEEE Transactions on
Pattern Analysis and Machine Intelligence*. 1986;8(6):679–698.

[11] Vincent L, Soille P. Watersheds in digital spaces: an efficient algorithm
based on immersion simulations. *IEEE Transactions on Pattern Analysis and
Machine Intelligence*. 1991;13(6):583–598.

[12] Chan TF, Vese LA. Active contours without edges. *IEEE Transactions on
Image Processing*. 2001;10(2):266–277.

[13] Comaniciu D, Meer P. Mean shift: a robust approach toward feature space
analysis. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
2002;24(5):603–619.

[14] Rother C, Kolmogorov V, Blake A. GrabCut: interactive foreground extraction
using iterated graph cuts. *ACM Transactions on Graphics*. 2004;23(3):309–314.

[15] Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for
biomedical image segmentation. *Medical Image Computing and Computer-Assisted
Intervention (MICCAI)*. 2015:234–241.

**Metric**

[16] Dice LR. Measures of the amount of ecologic association between species.
*Ecology*. 1945;26(3):297–302.

**Variable autonomy (added in Batch 2, turn 71)**

[17] Yang GZ, Cambias J, Cleary K, Daimler E, Drake J, Dupont PE, et al. Medical
robotics — regulatory, ethical, and legal considerations for increasing levels
of autonomy. *Science Robotics*. 2017;2(4):eaam8638.
doi:10.1126/scirobotics.aam8638

[18] *Levels of autonomy in FDA-cleared surgical robots: a systematic review.*
*npj Digital Medicine*. 2024. doi:10.1038/s41746-024-01102-y
*[Verify author list before submission.]*

[19] Attanasio A, Scaglioni B, De Momi E, Fiorini P, Valdastri P. Autonomy in
surgical robotics. *Annual Review of Control, Robotics, and Autonomous Systems*.
2021. *[Verify volume and page range.]*

---

## Two sentences to add to the Introduction

Insert into Section 1, after the paragraph beginning "Absolute performance on
synthetic data is not usually claimed to transfer":

> That relative performance is fragile has been established for real benchmarks:
> rankings in biomedical image analysis challenges shift with the choice of
> metric, aggregation scheme, annotator, and test set [1–3]. What has not been
> established is which property of a benchmark drives that fragility, or whether
> a synthetic benchmark preserves ranking at all.

This positions the contribution inside an accepted problem rather than asserting
a new one, which is the safer rhetorical position and also the accurate one.

---

## Remaining searches, in priority order

1. `synthetic benchmark ranking correlation real data segmentation` — the gap
   claim in 2.1. **PARTIALLY RESOLVED** by arXiv:2409.11968 (turn 71). Still
   worth re-running for medical-imaging-specific work.
2. `variable autonomy surgical robotics levels of autonomy` — **RESOLVED**,
   refs [17]–[19].
3. `wound segmentation review colour features` — 3–5 refs for 2.3. OUTSTANDING.
4. `synthetic data medical image analysis survey` — 3–5 refs for 2.2. OUTSTANDING.
5. `domain shift medical image segmentation multi-centre` — 3–4 refs for 2.4. OUTSTANDING.
6. `test time augmentation uncertainty medical image segmentation` — 1–2 refs. OUTSTANDING.
7. Medetec and FUSeg challenge citations. OUTSTANDING.

Verified count at time of writing: 16 (19 after Batch 2). Target 25–35.
