# Manuscript V3, Part 2: methods

---

## 3. Method pool

Thirty-two segmentation methods in eight families. Every method is run through
one interface with one post-processing step, so nothing in the ranking reflects
differences in how the outputs were cleaned up.

| family | n | methods |
|---|---|---|
| background-referenced | 5 | adaptive Gaussian, adaptive mean, Otsu on grey, BGR corner distance, HSV corner distance |
| colour-referenced | 4 | Otsu on a*, 92nd percentile of a*, redness ratio, Otsu on saturation |
| clustering | 4 | k-means in Lab at k = 3, 4 and 5, mean shift with Otsu on a* |
| region and contour | 5 | watershed on a*, GrabCut from centre, flood fill on a*, Chan-Vese on a*, Canny with fill |
| pixel classifiers | 3 | logistic regression on all features, gradient boosting, logistic regression on colour only |
| small CNN | 1 | three-level U-Net, 117,681 parameters |
| deep encoder-decoder | 5 | U-Net, U-Net++, LinkNet on ResNet-18; U-Net on MobileNetV2 and EfficientNet-B0 |
| deep other | 5 | DeepLabV3+, PSPNet, FPN, MAnet on ResNet-18; SegFormer on MiT-B0 |

### 3.1 The classical methods

Twenty-one algorithms with no learned parameters. Each produces a binary mask
from a single image. They are included because they span a wide range of
segmentation principles at a wide range of performance, which gives the rank
correlations something to order, and because being untrained they carry no
dependence on any dataset.

Two implementation details matter for reproduction. The redness ratio
R / (G + B) is unbounded, and pixels where G + B approaches zero send it to
approximately 1e6, which collapses the subsequent 8-bit rescale to a handful of
levels and empties the mask; it is clipped to the 1st and 99th percentiles
before thresholding. Methods that return a mostly-foreground mask are inverted,
since several of these operators have no intrinsic polarity.

### 3.2 The learned methods

Three pixel classifiers are fitted on 264,292 pixels sampled from 250 FUSeg
training images, balanced between lesion and non-lesion.

The small U-Net is a three-level network with 16, 32 and 64 channels, trained on
393 FUSeg training images at 128 x 128 from random initialisation. Convolution
bias is enabled, giving exactly 117,681 parameters.

### 3.3 The modern architectures

Ten architectures from segmentation_models_pytorch, chosen to vary along the two
axes modern segmentation varies along:

- seven decoders on one encoder, ResNet-18: U-Net, U-Net++, DeepLabV3+, FPN,
  PSPNet, LinkNet, MAnet
- three alternative encoders: U-Net on MobileNetV2, U-Net on EfficientNet-B0,
  and SegFormer on MiT-B0, which is a transformer and brings its own decoder

**No pretrained weights are used anywhere in this study.** Every architecture is
randomly initialised, so no method enters the pool carrying knowledge of an
external dataset, and the pool is not confounded by differences in pretraining
corpora.

All ten see identical data under an identical protocol, copied from the small
U-Net so that the new and old methods are comparable: the same 393 FUSeg
training images, 128 x 128, Dice plus binary cross-entropy, Adam, horizontal
flip augmentation, batch size 8, 30 epochs. The single deliberate difference is
a learning rate of 1e-3 rather than 3e-3, because 3e-3 with Adam is not stable
for models of 14 to 22 million parameters. One image per epoch is dropped
because 393 images at batch size 8 leaves a final batch of one, and DeepLabV3+
applies batch normalisation to a 1 x 1 feature map in its pooling branch, which
is undefined for a single sample. The drop is applied to every architecture so
the protocol stays identical across the pool.

Decision thresholds are swept per model on FUSeg training images 400 to 550, a
slice disjoint from the 393 used for training and from every evaluation split.
Held-out Dice at the selected threshold ranges from 0.507 to 0.641, so the pool
has genuine spread rather than ten near-identical models.

### 3.4 A caveat on pool composition that affects the results

Eleven of the 32 methods are trained, and all eleven are trained on FUSeg
training images. Section 6.2 shows that this makes FUSeg unsuitable as a neutral
clinical reference for this pool: the trained methods beat the untrained ones by
+0.449 mean Dice on FUSeg against +0.079 and +0.066 on the other two wound
datasets. Headline agreement figures in this paper are therefore reported
against Medetec and AZH, which no method was trained on.

---

## 4. Clinical reference sets

| reference | n scored | modality | source | median lesion fraction |
|---|---|---|---|---|
| Medetec | 152 | wound photography | Medetec archive | 0.102 |
| FUSeg validation | 120 | wound photography | AZH Wound and Vascular Center | 0.011 |
| AZH test | 150 | wound photography | AZH Wound and Vascular Center | 0.029 |
| AZH train slice | 150 | wound photography | AZH Wound and Vascular Center | 0.026 |
| Kvasir-SEG | 120 | colonoscopy | Kvasir collection | 0.106 |
| Kvasir-SEG sessile | 120 | colonoscopy | Kvasir collection | 0.043 |

Five distinct reference sets from three independent sources. The AZH train slice
is a disjoint 150-image slice of the same dataset as AZH test and exists solely
as a within-dataset control.

FUSeg and AZH come from the same clinic: the Foot Ulcer Segmentation Challenge
was built from the AZH Wound and Vascular Center data. Template matching 25 AZH
patches against all 1,010 FUSeg frames at six scales gives a maximum normalised
cross-correlation of 0.927 with no exact matches, so they are distinct images
under different curation conventions rather than re-crops of one another. We
treat them as one source throughout.

Kvasir-SEG and its sessile subset likewise share a source. The cross-modality
conclusions therefore rest on one endoscopy source, which is the study's main
data limitation.

### 4.1 Mask handling

Medetec masks are 0/1 and FUSeg masks 0/255, so wound masks are binarised on any
non-zero value. A fixed mid-range threshold silently empties the Medetec masks
and produces a Dice of exactly zero for every method.

Kvasir masks are JPEG-compressed and carry anti-aliased edge values of 1, 2, 3
and so on, so binarising on any non-zero value inflates every lesion. Kvasir is
thresholded at 127. The threshold was not swept.

Kvasir frames vary in size and are resampled so the long side is 256 px,
matching Medetec's resolution.

---

## 5. Synthetic benchmarks

Three generator families, built on different principles. Each produces images
with ground-truth lesion masks and exposes its design properties as explicit
parameters, which is why parametric generators are used here rather than a
diffusion model: the ablation and dose-response experiments require that
individual design properties be switched off and swept independently, and a
learned generator does not offer those controls.

**Generator A** places a lesion on a uniform drape background. It has no
background texture parameter.

**Generator B** renders textured skin: a shared luminance mottling field with a
small independent chroma residual, lesion tissue drawn from a clinically
calibrated palette in ordered skin-tone depth, plus clutter such as hair and
gauze. Its background texture amplitude is the swept parameter for the
dose-response experiments.

**Generator C** shares no code path with generator B and differs in every
mechanism: 1/f^beta spectral-filtered noise instead of upsampled Gaussian
mottling, a radial vignette instead of a linear ramp, speckle and grain instead
of hair and gauze, distance-field alpha blending instead of filled polygons, a
CIELAB Gaussian mixture instead of an HSV offset from skin tone, and a spatial
tissue gradient instead of blob patches.

We describe these as constructed on different principles rather than as
independently constructed. They were written by the same authors and share
whatever unstated assumptions we hold about what a wound photograph looks like.
Generator C exists specifically to break the shared-implementation assumption
and is the instrument for the non-reducibility test in Section 6.5.

### 5.1 Background texture statistic

The median local standard deviation of shading-corrected luminance in a 9 px
window, measured over background pixels only.

Every image is resampled to a canonical 512 px before measurement. This is not
cosmetic. Measured natively, Medetec at 224 px reads 5.07 and FUSeg at 512 px
reads 3.88; resampled to a common 512 px they read 2.69 and 3.88, which reverses
their order. A statistic that changes the ordering of two datasets depending on
the resolution they happen to be distributed at is not a property of the images.

---

## 6. Statistical protocol

**Agreement.** Spearman rank correlation between two vectors of per-method mean
Dice over the same method pool. Two-sided permutation p-values at 20,000
permutations. Confidence intervals by bootstrap over methods at 5,000 resamples.

**Detection limit.** At n = 32 methods, a Spearman requires approximately
|rho| >= 0.35 to reach p < 0.05; at n = 22 the requirement is 0.43. Where we
report an absence of agreement we state it as an absence of detectable
agreement at this sample size, not as a demonstration that agreement is zero.

**Metrics.** Every relationship that this paper claims is computed twice: with
Dice, and with normalised surface Dice at a 2 px tolerance. The second is
included because Dice carries a region-size bias and the framing experiment in
Section 6.4 varies region size, so the two would otherwise be confounded by
construction. Both metrics treat two empty masks as perfect agreement.

**Seeds.** Every synthetic benchmark used in a reported relationship is built at
four generator seeds (2024, 7, 101, 555) and the relationship is computed
independently for each. Single-seed results overturned two conclusions during
this work and are not reported as findings anywhere in this paper.

**The verdict statistic.** A reference's preference over a design property is
the slope of its rank agreement against the measured value of that property,
fitted by ordinary least squares over the sweep and expressed per 10 units.

An earlier version of this work used the Spearman correlation of agreement
against the property instead. That statistic saturates: once the underlying
relationship is monotone it returns exactly +1 or -1 regardless of magnitude. At
n = 32 it returned exactly +1.000 for five of eight conditions where the
underlying agreements ranged from -0.33 to +0.36, discarding the signal it was
meant to measure. The slope is continuous and does not saturate. Results
computed with the rank version are not reported.
