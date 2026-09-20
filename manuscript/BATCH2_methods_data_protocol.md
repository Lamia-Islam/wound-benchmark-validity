# MANUSCRIPT — BATCH 2
## Sections 3–5: Methods, Data, Evaluation Protocol

> RECOVERED from conversation f7ee7590-a160-40ce-bc6c-e90dd81e1c63, turn 71.
> Original file path in the lost container: `/mnt/user-data/outputs/MANUSCRIPT_BATCH2.md`

Continues from Batch 1 (title, abstract, introduction) and Batch 1 Rev A
(related work). Two positioning corrections from the searches are folded in at
the end of this document — read those first, they change one sentence in the
Introduction.

---

## 3. Segmentation methods

Twenty-two methods were evaluated. They were selected to span the approaches a
practitioner might plausibly consider for wound segmentation, deliberately
including methods expected to perform poorly, since a benchmark that only
contains strong methods cannot reveal a ranking failure. All take a BGR image
and return a binary mask; no method received per-image tuning.

### 3.1 Background-referenced methods

Five methods estimate a background model from the four 20 × 20 image corners and
label pixels by their distance from it: grayscale adaptive thresholding in
Gaussian and mean variants, global Otsu thresholding on grayscale [9], Euclidean
distance from the median corner colour in BGR, and the same in HSV. All assume
the lesion is not at the image corners.

### 3.2 Colour-referenced methods

Four methods threshold a colour channel chosen for physiological relevance
rather than for contrast with background: Otsu on the CIELAB a\* channel, the
92nd percentile of a\*, the redness ratio R/(G+B), and Otsu on HSV saturation.
The a\* channel encodes red–green opponency and separates erythematous tissue
from perilesional skin independently of luminance.

### 3.3 Clustering methods

Four methods: k-means in CIELAB at k = 3 and k = 5, taking the modal corner
cluster as background; k-means at k = 4 retaining the cluster with the highest
mean a\*, which asks which cluster resembles wound tissue rather than which is
not background; and pyramid mean-shift filtering [13] followed by Otsu on a\*.

### 3.4 Region and contour methods

Five methods: marker-based watershed [11] seeded from a\* extremes; GrabCut [14]
initialised from a central rectangle covering 70% of the frame; flood-fill
region growing from the a\* maximum; a two-phase Chan–Vese style region
competition on a\* without an edge term [12]; and Canny edge detection [10] with
morphological closure and contour filling.

### 3.5 Learned methods

Three per-pixel classifiers over fourteen features: CIELAB, HSV and BGR
channels; local mean and standard deviation of a\* and of grayscale over a 9 × 9
window; and a\* and saturation centred on each image's own median. The last two
express "redder and more saturated than this patient's skin" rather than an
absolute colour, which is what permits transfer across skin tones and
illumination. Pixel coordinates are deliberately excluded: both clinical
datasets centre their lesions to some degree, and a classifier given position
learns framing convention rather than appearance.

Variants: logistic regression over all fourteen features, gradient boosting over
all fourteen, and logistic regression over the nine colour features only. All
trained on 264,292 pixels sampled from 250 FUSeg training images, balanced
between lesion and non-lesion, in approximately one second on a single CPU core.

### 3.6 Convolutional network

A three-level U-Net [15] with 16, 32 and 64 channels (117,681 parameters),
trained from random initialisation on 393 FUSeg training images at 128 × 128 for
30 epochs with combined Dice and binary cross-entropy loss, horizontal flip
augmentation, and Adam at learning rate 3 × 10⁻³. Training took 470 s on a
single CPU core.

This is not intended to approach published FUSeg performance, which uses GPUs,
pretrained encoders, and full training schedules. It is included so the method
pool is not confined to classical approaches, and so the ranking analysis
includes a method of a fundamentally different family.

### 3.7 Post-processing

All methods receive identical post-processing: morphological closing then
opening with an elliptical kernel, and removal of connected components below
0.05% of frame area. Learned methods additionally receive Gaussian smoothing of
the probability field before thresholding, rather than morphological cleaning of
a binary mask, because morphology on a noisy binary mask preferentially erases
small lesions — the failure mode that matters most in this application.

### 3.8 On the size of the method pool

Twenty-two is a methodological requirement, not thoroughness for its own sake.
Spearman rank correlation over n = 6 methods requires |ρ| ≥ 0.83 to reach
p < 0.05; over n = 22 it requires |ρ| ≥ 0.42. Ranking conclusions drawn from a
handful of methods cannot be distinguished from noise, and we note this because
comparisons of that size are common. Our own earlier analysis at n = 6 produced
correlations that appeared substantially larger and none of which were
significant.

> **REVISION issued later in the same turn — replace 3.8 with:**
>
> **3.8 Size of the method pool**
>
> Spearman rank correlation over n = 6 methods requires |ρ| ≥ 0.83 to reach
> p < 0.05; over n = 22 it requires |ρ| ≥ 0.42. A pool of twenty-two is
> therefore the minimum at which moderate rank agreement can be distinguished
> from chance, and comparisons involving a handful of methods are not powered
> to support ranking conclusions.

---

## 4. Data

### 4.1 Generator A: lesion composited onto uniform background

Generator A synthesises 512 × 512 images in three severity grades. A centred
ellipse (semi-axes 90 × 85 px for the healthy grade) is composited onto a
near-uniform surgical drape with a mild radial illumination gradient. Sutures
are drawn for the two less severe grades. Grade is encoded through lesion size,
tissue colour composition, and boundary irregularity.

This design is conventional: a lesion of controlled geometry placed on clean
background, with severity encoded in appearance. It is representative of
generators used when clinical data is unavailable.

### 4.2 Generator B: lesion embedded in textured skin

Generator B was constructed independently, targeting measured properties of
clinical wound photographs rather than the weaknesses of any segmentation
method. Building a generator to defeat corner sampling would be as circular as
building one that rewards it.

Design targets:

| property | generator B | rationale |
|---|---|---|
| lesion fraction | log-uniform, 0.005–0.12 of frame, sampled independently of grade | spans measured clinical range; size carries no label information |
| placement | uniform over the central 60% of frame | clinical photographs are not centred by construction |
| background | textured skin: low-frequency mottling, fine noise, directional lighting gradient, hair | clinical backgrounds are not uniform |
| skin tone | varied per image over a plausible BGR range | multiple patients |
| clutter | gauze and instrument-like regions in a minority of frames | operative and clinical scenes contain non-anatomical objects |
| lesion shape | radially perturbed polygon | lesions are not elliptical |
| specularity | highlights over moist tissue | wet tissue reflects |

Grade is defined by tissue composition (granulation, slough, necrosis
proportions) and boundary irregularity, never by size.

Both generators emit exact ground-truth masks — the mask used to paint each
image — and are deterministic given a seed. Both are released.

### 4.3 Clinical datasets

Two publicly available datasets with expert annotations, both distributed at
`github.com/uwm-bigdata/wound-segmentation`:

| | FUSeg | Medetec |
|---|---|---|
| source | AZH Wound and Vascular Center [5] | Medetec wound database |
| annotated images used | 1,010 (400 train, 150 validation, 120 test) | 152 |
| resolution | 512 × 512 | 224 × 224 |
| median lesion fraction | 0.0068 | 0.098 |
| framing | full-limb photographs | cropped around lesion |

The fourteen-fold difference in median lesion fraction is not incidental. It
means the two datasets pose measurably different problems, and it is used in
Section [Results] to show that reporting either alone gives a misleading
estimate of segmentation performance.

Label encodings differ between datasets — Medetec masks are 0/1 valued and FUSeg
0/255 — and are binarised on any non-zero value. We note this because a fixed
mid-range threshold silently empties the Medetec masks and yields a Dice of
exactly zero for every method, an error we made and corrected during this work.

> **REVISION issued later in the same turn — replace the final paragraph of 4.3 with:**
>
> Label encodings differ between the two datasets: Medetec masks are 0/1 valued
> and FUSeg masks 0/255. Both are binarised on any non-zero value.

### 4.4 The perception-gated autonomy pipeline

A ROS 2 Humble pipeline of four nodes. A detector node segments an incoming
image and scores it against a surgical adaptation of the dermatological ABCDE
rule (asymmetry via contour solidity, border via Laplacian variance on a
boundary strip, colour via tissue-zone ratios, diameter via minimum enclosing
circle, evolution via inter-frame area change). Each criterion contributes 0, 1
or 2. A behaviour manager maps the total to a verdict and thence to a
manipulator autonomy level, and a bridge node publishes corresponding motion
constraints.

| total | verdict | autonomy | speed factor |
|---|---|---|---|
| 0–1 | PASS | full | 1.00 |
| 2 | CAUTION | reduced, per-waypoint confirmation | 0.50 |
| ≥ 3 | FAIL | hold | 0.00 |

This corresponds to transitions between levels 1 and 2 in the widely used
six-level taxonomy for medical robot autonomy [17]. The pipeline is not a
clinical device and is not proposed as one; it is an instrumented consumer of
segmentation output, included so the consequences of a validation failure can be
measured rather than argued.

> **REVISION issued later in the same turn — replace 4.4's closing sentence with:**
>
> The pipeline switches between levels 1 and 2 of the six-level taxonomy for
> medical robot autonomy [17]. It is an instrumented consumer of segmentation
> output, included so that the downstream consequences of a validation failure
> can be measured directly.

---

## 5. Evaluation protocol

### 5.1 Metric

Dice similarity coefficient [16] against ground truth, computed per image and
averaged. The same metric is used on synthetic and clinical data, which is what
makes the four benchmarks directly comparable. Where a method produces an empty
mask and the reference is also empty, Dice is defined as 1.

Metric choice is held fixed throughout by design. Prior work has shown that
metric selection and aggregation scheme are themselves sources of ranking
instability [1–3]; by fixing both, any ranking difference we observe is
attributable to the data rather than to the scoring procedure.

### 5.2 Rank agreement

Method rankings are compared between benchmarks using Spearman rank correlation,
following the approach used for the analogous question in natural language
processing [SEARCH — cite arXiv:2409.11968, "Efficacy of Synthetic Data as a
Benchmark"; verify whether a peer-reviewed version now exists and prefer it].
Significance is assessed by permutation test with 20,000 permutations of one
ranking, reporting the two-sided proportion of permuted correlations at least as
extreme as observed. Bootstrap 95% confidence intervals on mean Dice use 5,000
resamples.

### 5.3 Generator ablation

Generator B is ablated one design feature at a time, each reverting to generator
A's corresponding assumption, with all other features held at generator B
settings and the random seed fixed. For each variant, all 22 methods are scored
and the resulting ranking is correlated against both clinical datasets. A
feature that governs predictive validity should produce a large drop when
removed.

### 5.4 Threshold selection for learned methods

Balanced pixel sampling induces a prior mismatch: the classifier learns under a
50% lesion prior while clinical images are 0.7–10% lesion. Applying a 0.5
decision threshold under that mismatch over-segments severely.

The decision threshold was selected by sweeping candidate values on FUSeg
*training* images 250–400, which were not used to fit the classifier and are
disjoint from every evaluation split. The selected value (0.9) was then applied
unchanged to both clinical datasets and both synthetic benchmarks. No evaluation
split influenced any threshold.

### 5.5 Competence signal protocol

Seventeen label-free signals were evaluated as predictors of per-image Dice,
spanning four families: model confidence, ensemble disagreement, test-time
augmentation instability, and geometric plausibility.

The acceptance criterion was fixed before any signal was computed: a signal is
usable only if |Spearman ρ| ≥ 0.6 against per-image Dice on **both** clinical
datasets, with consistent sign. All seventeen signals are reported regardless of
outcome. This protocol is stated here rather than in the results to make clear
that it was a criterion rather than a post-hoc filter; selecting the
best-correlating signal after inspection would be selection on the outcome.

### 5.6 Gate safety metric

Severity labels do not exist for either clinical dataset, so gate *correctness*
cannot be measured. Gate *safety* can, because per-image Dice against expert
annotation is known.

An **unsafe-optimistic** decision is defined as full autonomy granted on a frame
where segmentation failed (Dice < 0.5). A **conservative** decision is reduced
autonomy or hold on a frame where segmentation succeeded. The former is a safety
failure; the latter is a cost. We report both, alongside an oracle gate that
grants full autonomy exactly when Dice ≥ 0.5, as an unreachable upper bound.

### 5.7 Reproducibility

All experiments are deterministic given the stated seeds. Synthetic benchmarks
are regenerated from seed rather than distributed as image files. Code, models,
fitted thresholds and per-image result files are released under MIT licence with
an archived DOI. No clinical images are redistributed; both clinical datasets
are obtained from their original public sources.

---

## Positioning corrections from the literature searches

**Correction 1 — the rank-correlation method is not novel, and that helps.**
The question of whether synthetic data preserves method ranking, measured by
Spearman correlation against real data, has been posed in natural language
processing (arXiv:2409.11968). Cite it in Section 5.2 as methodological
precedent. This strengthens the paper: rank correlation becomes an established
instrument rather than a choice needing defence.

Revise the Introduction gap sentence to:

> Whether synthetic benchmarks preserve method ranking has been examined in
> natural language processing, where rank correlation against real benchmarks
> is the established instrument. To our knowledge it has not been examined for
> medical image segmentation, and no prior work isolates *which* property of a
> synthetic benchmark governs rank preservation.

The second clause is the contribution and it remains intact.

**Correction 2 — verified variable autonomy references.**

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

Yang et al. define six levels from 0 (no autonomy) to 5 (full autonomy), with
level 1 robot assistance, level 2 task autonomy, level 3 conditional autonomy,
and level 4 high autonomy. The gate switches between levels 1 and 2.

---

## Notes for Batch 3 (carried forward)

- Lead the results with the two significant, well-powered comparisons
  (A vs B at ρ = −0.526, p = 0.014; Medetec vs FUSeg at ρ = 0.560, p = 0.008).
  The single-method illustration — the U-Net at Dice 0.080 on generator A and
  0.670 on FUSeg — belongs immediately after, as it is the most legible
  statement of the finding.
- Report the two non-significant generator-to-clinical correlations plainly
  rather than burying them.
- The Medetec 0.557 versus cross-source 0.451 discrepancy for the learned
  baseline is a self-directed methodological caution and should be presented as
  such, since it is an instance of the paper's own thesis.

---

## References cited in this batch (partial list from turn 71)

**Methods**
- Canny J. A computational approach to edge detection. *IEEE TPAMI*. 1986;8(6):679–698.
- Vincent L, Soille P. Watersheds in digital spaces: an efficient algorithm based on immersion simulations. *IEEE TPAMI*. 1991;13(6):583–598.
- Comaniciu D, Meer P. Mean shift: a robust approach toward feature space analysis. *IEEE TPAMI*. 2002;24(5):603–619.
- Chan TF, Vese LA. Active contours without edges. *IEEE TIP*. 2001;10(2):266–277.
- Rother C, Kolmogorov V, Blake A. GrabCut: interactive foreground extraction using iterated graph cuts. *ACM TOG*. 2004;23(3):309–314.
- Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for biomedical image segmentation. *MICCAI*. 2015:234–241.

**Metric**
- Dice LR. Measures of the amount of ecologic association between species. *Ecology*. 1945;26(3):297–302.

**Data**
- Wang C, Anisuzzaman DM, Williamson V, Dhar MK, Rostami B, Niezgoda J, Gopalakrishnan S, Yu Z. Fully automatic wound segmentation with deep convolutional neural networks. *Scientific Reports*. 2020;10:21897.
- [VERIFY] Medetec wound database — cite the database itself (https://www.medetec.co.uk/) and check whether a citable publication by S. Thomas exists. Also cite the FUSeg challenge (https://fusc.grand-challenge.org/) and confirm whether a challenge report paper is available.

**Uncertainty**
- Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. *ICML*. 2017:1321–1330.
- Hendrycks D, Gimpel K. A baseline for detecting misclassified and out-of-distribution examples in neural networks. *ICLR*. 2017.
- Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*. 2017:6402–6413.
