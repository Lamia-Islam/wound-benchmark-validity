# MANUSCRIPT V2 — Part 2
## Sections 2 to 5: Related work, methods, data, evaluation protocol

---

## 2. Related work

### 2.1 Validation and ranking instability in biomedical image analysis

In a systematic analysis of 150 challenges, Maier-Hein et al. showed that an
algorithm's rank is not robust to the test data used for validation, the ranking
scheme applied, the observers producing reference annotations, or the method of
aggregating per-case values [1]. Their re-evaluation of the 2015 segmentation
challenges found that a quarter of non-winning algorithms would have placed
first under a different but equally defensible aggregation choice. The same
group developed the Metrics Reloaded framework for problem-aware metric
selection [2], a companion analysis of metric-related pitfalls [3], and
reporting guidelines for challenge transparency [4].

That work establishes that rankings are fragile and names several sources. It
does not supply a quantity a benchmark designer can compute on their own images
to find out whether their benchmark is one of the fragile ones. Our study holds
metric, aggregation and reference annotations fixed, varies one measurable
property of the images, and reports the response. This converts benchmark design
from a known risk into a measured one with a target value.

### 2.2 Whether synthetic data preserves method ranking

The question of whether synthetic data preserves the relative ranking of methods
seen on real data, measured by Spearman correlation, has been posed in natural
language processing [5]. Rank correlation is therefore an established instrument
for this question rather than a choice needing defence. To our knowledge it has
not been applied to medical image segmentation, and no prior work identifies a
measurable image property that governs the outcome or reports its target value.

[SEARCH before submission: "synthetic benchmark ranking correlation medical
image segmentation"; "does synthetic data preserve method ranking medical
imaging". If medical-imaging prior work exists, this becomes positioning.]

### 2.3 Synthetic data in medical image analysis

Synthetic imagery supplements scarce annotated data via physics-based
simulation, compositing of lesions onto backgrounds, and generative modelling.
Existing evaluation largely concerns use for training, whether adding synthetic
images improves a model. Whether synthetic data supports valid validation, in
the sense of preserving the ordering of candidate methods, is the subject here.

[SEARCH: "synthetic data medical image analysis survey"; "generative
augmentation medical image segmentation review". Need 3 to 5 references,
including at least one example of compositing-based generation, since
compositing is the specific practice this paper addresses.]

### 2.4 Wound image segmentation

Classical wound segmentation has used thresholding, colour-space clustering,
region growing and active contours, with colour features motivated by the
clinical red-yellow-black tissue scheme. Deep segmentation now dominates. Wang
et al. released a fully automatic pipeline together with the annotated dataset
from the AZH Wound and Vascular Center that underpins the Foot Ulcer
Segmentation Challenge [6]; reported Dice for convolutional architectures on
that benchmark is in the range 0.89 to 0.93.

[SEARCH: 3 to 5 classical wound segmentation references; 2 to 3 recent
high-performing FUSeg methods to substantiate the range.]

### 2.5 Cross-dataset generalisation

Performance degradation under distribution shift is well documented in medical
imaging, and evaluation confined to a single dataset is known to overstate
generalisation. Our framing result in Section 6.4 is a sharper form of this: two
datasets from different clinics do not merely differ in difficulty, they invert
each other's verdict on which synthetic design is realistic, and the inversion
is removed by a crop.

[SEARCH: "domain shift medical image segmentation multi-centre generalisation".
Need 3 to 4 references.]

### 2.6 Perception-conditioned variable autonomy

Variable autonomy systems adjust the division of control between operator and
machine according to perceived conditions [7 to 9]. The safety argument presumes
perception is either accurate or known to be inaccurate. Section 7 measures, on
annotated clinical imagery, how often a gate of this form grants full autonomy
while its perception has in fact failed.

---

## 3. Segmentation methods

Twenty-two methods, selected to span the approaches a practitioner might
plausibly consider, deliberately including methods expected to perform poorly,
since a benchmark containing only strong methods cannot reveal a ranking
failure. All take a BGR image and return a binary mask. No method receives
per-image tuning.

### 3.1 Background-referenced (5)

A background model is estimated from the four 20 x 20 image corners and pixels
labelled by distance from it: grayscale adaptive thresholding in Gaussian and
mean variants, global Otsu on grayscale [10], Euclidean distance from the median
corner colour in BGR, and the same in HSV. All assume the lesion is not at the
image corners.

### 3.2 Colour-referenced (4)

Four methods threshold a colour channel chosen for physiological relevance
rather than for contrast with background: Otsu on the CIELAB a* channel, the
92nd percentile of a*, the redness ratio R/(G+B), and Otsu on HSV saturation.
The a* channel encodes red-green opponency and separates erythematous tissue
from perilesional skin independently of luminance.

The redness ratio is unbounded where G+B approaches zero. It is clipped to the
1st to 99th percentile before thresholding; without this the rescaling collapses
the image to a handful of levels and the method returns an empty mask on every
image, which would place it in a rank it did not earn.

### 3.3 Clustering (4)

k-means in CIELAB at k = 3 and k = 5 taking the modal corner cluster as
background; k-means at k = 4 retaining the cluster with the highest mean a*,
which asks which cluster resembles wound tissue rather than which is not
background; and pyramid mean-shift filtering [11] followed by Otsu on a*.

### 3.4 Region and contour (5)

Marker-based watershed [12] seeded from a* extremes; GrabCut [13] initialised
from a central rectangle covering 70% of the frame; flood-fill region growing
from the a* maximum; a two-phase Chan-Vese style region competition on a*
without an edge term [14]; and Canny edge detection [15] with morphological
closure and contour filling.

### 3.5 Learned per-pixel classifiers (3)

Fifteen per-pixel features: CIELAB, HSV and BGR channels (9); local mean and
standard deviation of a* and of grayscale over a 9 x 9 window (4); and a* and
saturation centred on each image's own median (2). The last two express "redder
and more saturated than this patient's skin" rather than an absolute colour,
which is what permits transfer across skin tones and illumination. Pixel
coordinates are deliberately excluded: both clinical datasets centre their
lesions to some degree, and a classifier given position learns framing
convention rather than appearance.

Variants: logistic regression over all features, gradient boosting over all
features, and logistic regression over the nine colour features only. All
trained on approximately 248,000 pixels sampled from 250 FUSeg training images,
balanced between lesion and non-lesion, in about ten seconds on a single CPU
core.

Section 6.5 reports a leave-one-out analysis over these fifteen features and its
consequences for benchmark design.

### 3.6 Convolutional network (1)

A three-level U-Net [16] with 16, 32 and 64 channels, 117,681 parameters,
trained from random initialisation on 393 FUSeg training images at 128 x 128 for
30 epochs with combined Dice and binary cross-entropy loss, horizontal flip
augmentation, and Adam at learning rate 3e-3. Training takes under 500 s on a
single CPU core.

This is not intended to approach published FUSeg performance, which uses GPUs,
pretrained encoders and full training schedules. It is included so the method
pool is not confined to classical approaches and so the ranking analysis
contains a method of a different family.

### 3.7 Post-processing

Identical for all methods: morphological closing then opening with an elliptical
kernel, and removal of connected components below 0.05% of frame area. Learned
methods instead receive Gaussian smoothing of the probability field before
thresholding, because morphology on a noisy binary mask preferentially erases
small lesions, the failure mode that matters most here.

### 3.8 Size of the method pool

Spearman rank correlation over n = 6 methods requires an absolute rho of at
least 0.83 to reach p < 0.05; over n = 22 it requires 0.42. A pool of
twenty-two is therefore close to the minimum at which moderate rank agreement
can be distinguished from chance, and comparisons involving a handful of methods
are not powered to support ranking conclusions.

---

## 4. The background texture statistic

The property we vary has to be measurable on any image set, or it is a knob in
our generator rather than a specification.

**Definition.** For one image, convert to luminance, divide by a heavily blurred
copy of itself (Gaussian, sigma 31 px) so the measurement is contrast-like
rather than intensity-like and insensitive to overall illumination, compute the
local standard deviation over a 9 x 9 window, and take the median over the
background region. The background excludes a margin of three window widths
around the lesion so the lesion boundary does not contribute its own step. For a
set of images, take the median over images.

Two variants are reported: masked, using the ground-truth mask to define
background, and unmasked over the whole frame for image sets with no masks to
hand. The two agree closely on every set we measured.

The window and the shading sigma are choices. Section 8 reports what is not yet
established about them.

**Measured values.**

| image set | masked | unmasked |
|---|---|---|
| Clinical Medetec (n = 152) | 2.69 | 3.10 |
| Clinical FUSeg (n = 120) | 3.97 | 4.09 |
| Generator A (uniform background) | 2.10 | 2.13 |
| Generator B, texture ablated | 2.14 | 2.16 |
| Generator B, default | 2.45 | 2.49 |
| Generator B, matched setting | 4.40 | 4.56 |

The clinical range is 2.69 to 3.97. A conventional uniform-background generator
measures 2.10, below the clinical floor.

**Resolution is part of the definition.** A fixed-pixel window measures texture
at a fixed pixel scale, so the same scene at two resolutions gives two answers.
Measured natively, Medetec (224 px) reads 5.07 and FUSeg (512 px) reads 3.97,
and the ordering of the two datasets is the reverse of what it is at a common
resolution. Every image is therefore resampled to 512 px before measuring, and
any reported value must state the canonical side and the window. This is not a
convenience: an earlier version of this analysis compared native resolutions and
obtained a clinical range that was an artifact of the comparison.

---

## 5. Data

### 5.1 Generator A: lesion composited onto uniform background

512 x 512 images in three severity grades. A centred ellipse (semi-axes 90 x 85
px for the healthy grade) composited onto a near-uniform surgical drape with a
mild radial illumination gradient. Sutures for the two less severe grades. Grade
encoded through lesion size, tissue colour composition and boundary
irregularity. This design is conventional and representative of generators used
when clinical data is unavailable. Measured background texture 2.10.

### 5.2 Generator B: lesion embedded in textured skin

Constructed to target measured properties of clinical wound photographs rather
than the weaknesses of any segmentation method. Building a generator to defeat
corner sampling would be as circular as building one that rewards it.

| property | setting | rationale |
|---|---|---|
| lesion fraction | log-uniform, 0.005 to 0.12 of frame, independent of grade | spans measured clinical range; size carries no label information |
| placement | uniform over the central 60% of frame | clinical photographs are not centred by construction |
| background | textured skin: low-frequency luminance mottling, fine noise, directional lighting gradient, hair strokes | clinical backgrounds are not uniform |
| skin tone | one depth parameter along a fixed skin chromaticity, so R > G > B always holds | multiple patients, without producing non-skin hues |
| clutter | gauze and instrument-like regions in a minority of frames | operative and clinical scenes contain non-anatomical objects |
| lesion shape | radially perturbed polygon | lesions are not elliptical |
| tissue palette | derived from each image's own skin tone, at a calibrated saturation offset | see below |
| specularity | highlights over moist tissue | wet tissue reflects |

Grade is defined by tissue composition (granulation, slough, necrosis
proportions) and boundary irregularity, never by size.

**Tissue palette calibration.** A fixed absolute palette produces whatever
lesion-to-surround contrast happens to fall out of the skin tone draw. Measured
against a 41 px perilesional ring, clinical images give a saturation delta of
+29.6 (Medetec) and +36.8 (FUSeg), so approximately +33. Generator A sits at
+55.8. The tissue colours are therefore derived from each image's sampled skin
tone at a calibrated offset, giving +38.4. This targets a measured clinical
property, not any method's behaviour.

Both generators emit exact ground-truth masks, the mask used to paint each
image, and are deterministic given a seed. Both are released.

### 5.3 Clinical datasets

Two publicly available datasets with expert annotations, both distributed at
`github.com/uwm-bigdata/wound-segmentation`.

| | FUSeg | Medetec |
|---|---|---|
| source | AZH Wound and Vascular Center [6] | Medetec wound database |
| annotated images | 1,010 (810 train, 200 validation) | 152 |
| evaluation slice used | validation images 0 to 120 | all 152 |
| resolution | 512 x 512 | 224 x 224 |
| median lesion fraction | 0.0072 | 0.0980 |
| framing | full-limb photographs | cropped around lesion |
| background texture (at 512 px) | 3.97 | 2.69 |

The fourteen-fold difference in median lesion fraction is not incidental. It is
the variable manipulated in Section 6.4.

Label encodings differ: Medetec masks are 0/1 valued and FUSeg 0/255. Both are
binarised on any non-zero value. A fixed mid-range threshold silently empties
the Medetec masks and yields Dice of exactly zero for every method.

---

## 6. Evaluation protocol

### 6.1 Metric

Dice similarity coefficient [17] against ground truth, computed per image and
averaged. Where a method produces an empty mask and the reference is also empty,
Dice is 1. The same metric is used on synthetic and clinical data, which is what
makes the benchmarks comparable. Metric and aggregation are held fixed by design
so that any ranking difference is attributable to the data [1 to 3].

### 6.2 Rank agreement

Method rankings are compared between benchmarks by Spearman rank correlation,
following [5]. Significance by permutation test, 20,000 permutations of one
ranking, two-sided. Bootstrap 95% confidence intervals on mean Dice use 5,000
resamples.

### 6.3 Sweeps

Both sweeps are one factor at a time over generator B, with all other design
properties held at their default and the seed fixed. At each level the benchmark
is generated, all 22 methods are scored, and the resulting ranking is correlated
against each clinical dataset separately. The measured texture statistic of the
generated images is reported alongside the nominal setting, so the reported axis
is an image property rather than a configuration field.

The contrast sweep is a null control. If rank agreement responded to any change
in target distinguishability, the texture result would admit a trivial reading.

### 6.4 The ceiling

The natural upper bound for a benchmark's rank agreement with clinical data is
how well one clinical dataset predicts ranking on another. We report Medetec
against FUSeg for this purpose and express synthetic results as a percentage of
it.

### 6.5 Reproducibility

All experiments are deterministic given the stated seeds. Synthetic benchmarks
are regenerated from seed rather than distributed as image files. Code, models,
fitted thresholds and per-image result files are released under MIT licence with
an archived DOI. No clinical images are redistributed; both clinical datasets
are obtained from their original public sources.
