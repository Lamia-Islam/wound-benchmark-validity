# Findings

Everything below is regenerable from the code in this repository against the
public FUSeg and Medetec datasets. Numbers that do not reproduce the original
manuscript are stated as such.

---

## 1. A portable specification for synthetic benchmark design

The original claim was "use textured backgrounds." That is advice. This is a
specification, because the quantity is measured on images rather than read off
a generator's config.

**Statistic.** `background_texture` = median over images of the median local
standard deviation of luminance in the background region, 9x9 window, after
dividing out low-frequency shading so it is contrast-like rather than
intensity-like. Implemented in `woundbench/imagestats.py`. Computable on any
image set, synthetic or clinical, with or without masks.

**Measured values.**

| image set | background texture |
|---|---|
| Clinical Medetec (n=152) | **5.07** |
| Clinical FUSeg (n=120) | **3.97** |
| generator A (uniform background) | 2.10 |
| generator B, amp 0 (texture ablated) | 2.14 |
| generator B, amp 11 | 2.45 |
| generator B, amp 28 | 3.50 |
| generator B, amp 40 | 4.40 |
| generator B, amp 55 | 5.61 |
| generator B, amp 75 | 7.07 |

A conventional uniform-background generator measures **2.10**, roughly half the
clinical floor of 3.97.

## 2. Predictive validity peaks where generator texture MATCHES clinical texture

Rank agreement with clinical data, against the measured statistic:

| background texture | mean rho vs clinical | % of ceiling |
|---|---|---|
| 2.14 (uniform, generator A's assumption) | 0.434 | 76% |
| 2.17 | 0.464 | 82% |
| 2.24 | 0.456 | 80% |
| 2.45 | 0.496 | 87% |
| 2.88 | 0.529 | 93% |
| 3.50 | 0.560 | 99% |
| **4.40** | **0.570** | **100%** |
| 5.61 | 0.542 | 95% |

Monotone rise to a peak at 4.40, which sits inside the clinical range
(3.97-5.07), then decline. **Spearman +0.964, p = 0.0028** over the rising
range where the generated images still read as skin.

The ceiling is how well one clinical dataset predicts ranking on another,
Medetec vs FUSeg, **rho = 0.567**. At matched texture the synthetic benchmark
reaches 0.570. A synthetic benchmark whose background texture matches clinical
predicts clinical method ranking as well as a second clinical dataset does.

This is a matching criterion, not a monotone "more is better": beyond the
clinical range agreement falls again, and by amplitude 55 the images no longer
read as skin.

## 3. Target visibility is not the mechanism (null control)

Sweeping lesion-to-surround colour contrast across seven levels spanning the
measured clinical value (~33 saturation units), generator A's value (~56) and
beyond gives **Spearman +0.143, p = 0.780**. Flat.

The texture effect is specifically about background structure, not about how
visible the target is. Without this control the texture result would be open to
the trivial reading.

## 4. A dataset's apparent realism preference is an artifact of its framing

Underneath that flat mean, the two clinical datasets pull in opposite
directions:

| reference | contrast preference | trend |
|---|---|---|
| Medetec (cropped, lesion fraction 0.098) | falls with contrast | Spearman **-0.893**, p 0.0132 |
| FUSeg (full-limb, lesion fraction 0.0068) | rises with contrast | Spearman **+0.964**, p 0.0034 |
| **FUSeg cropped to Medetec framing** | **falls with contrast** | Spearman **-0.964**, p 0.0030 |

Cropping FUSeg around its own lesions to Medetec's median lesion fraction
(0.0076 -> 0.0979, n=115) **reverses its preference**, from +0.964 to -0.964.
Same images, same anatomy, same clinic. Only the framing convention changed.

So "validate against clinical data" is underspecified. Which clinical data, and
how it is cropped, determines which synthetic design looks realistic. This is a
methodological finding in its own right and it is what makes the contrast mean
flat in section 3.

## 5. The mechanism, demonstrated at the level of a single input feature

The manuscript says its pixel classifiers used FOURTEEN features but enumerates
FIFTEEN. Leave-one-out over all fifteen identifies the missing one uniquely.

| dropped feature | Dice on generator A |
|---|---|
| none (all 15) | 0.936 |
| any of the 9 colour channels | 0.933 - 0.936 |
| a_localmean, gray_localmean, gray_localsd | 0.937 - 0.943 |
| a_centred, S_centred | 0.937 - 0.938 |
| **a_localsd** | **0.122** |

Manuscript reports 0.166. Gap 0.044 for `a_localsd`, versus ~0.77 for all
fifteen alternatives.

`a_localsd` is the local standard deviation of the CIELAB a\* channel: a pure
texture feature. On generator A the background measures 2.10, so local a\*
standard deviation is near zero everywhere outside the lesion, and the question
"is there local texture here?" answers "is this the lesion?" exactly. The
classifier gets the label for free.

This is the paper's figure-ground collapse thesis, demonstrated on one input
feature rather than argued. It is independent mechanistic evidence for the
central claim, and it was not in the original.

Refitting on the identified 14-feature set reproduces the manuscript's learned
scores on generator A closely: logistic regression 0.122 against 0.166,
gradient boosting 0.158 against 0.011, U-Net unchanged at 0.042 against 0.080.

## 6. What does NOT reproduce

The headline. The manuscript reports the two synthetic benchmarks ranking
methods in significantly opposite orders, rho = -0.526, p = 0.014.

| attempt | generator A vs generator B |
|---|---|
| 15-feature classifiers, generator B at amp 11 | +0.456 |
| 14-feature classifiers, generator B at amp 11 | +0.476 |
| 14-feature classifiers, generator B at amp 40 (clinical texture) | +0.423 |
| manuscript | **-0.526** |

Two independently motivated corrections, each of which fixed its own diagnostic,
and the sign does not flip. Ruled out along the way:

- the learned family is not the swing factor (excluding it entirely makes the
  correlation MORE positive, +0.637)
- over-segmentation does not explain the manuscript's 0.166 (predicting the
  whole frame gives 0.241 on generator A)
- generator B's texture being too low does not explain it (raising it to the
  clinical value barely moves the correlation)

Generator A vs FUSeg did improve substantially, from -0.099 to -0.304 with the
14-feature set, against the reported -0.466. But generator A vs generator B
stays positive under every configuration tested.

**Conclusion: this rebuild does not reproduce the reported rank inversion
between the two synthetic benchmarks.** What it supports is the weaker and
still-useful statement that generator A does not predict clinical ranking
(non-significant against both clinical datasets), plus everything in sections
1 to 5, which is more than the original established.

## 7. Bugs found and fixed during the rebuild

- **Rainbow background.** Low-frequency mottling drawn independently per colour
  channel decorrelates chroma and produces pastel blotches, making the
  background more saturated than the lesion. Fixed by carrying most of the
  amplitude in a shared luminance field.
- **Green skin.** Sampling B, G and R independently let blue exceed red. Fixed
  by drawing one depth parameter along a fixed skin chromaticity; R > G > B now
  holds on all images.
- **Degenerate redness ratio.** R/(G+B) is unbounded; near-zero G+B sends it to
  ~1e6 and a min-max rescale collapses the image to five grey levels, returning
  an empty mask on every image. Fixed by percentile clipping. Dice 0.000 ->
  0.430 on Medetec.

## 8. Verified against the manuscript

| check | manuscript | rebuild |
|---|---|---|
| U-Net parameter count | 117,681 | **117,681** |
| learned decision threshold | 0.9 | **0.90** (independently swept) |
| Medetec median lesion fraction | 0.098 | **0.0980** |
| mask encodings | 0/1 and 0/255 | confirmed |
| Medetec vs FUSeg rank agreement | 0.560, p 0.008 | **0.567, p 0.007** |
| generator B vs Medetec | 0.439, p 0.040 | 0.605, p 0.003 |
| ablation: texture dominates | -0.532 vs max 0.079 | -0.316 vs max 0.154 |

The U-Net count was diagnostic: bias=False gives 117,361, and the 320 shortfall
is exactly the ten convolution bias vectors.

## 9. Caveats

- Cross-seed variance is being measured (5 seeds x 5 texture levels); until it
  lands, every dose-response number is single-seed at 45 images per level.
- Amplitudes 55 and 75 are visually off-distribution and excluded from the
  headline trend.
- The gradient boosting score on generator A (0.158) does not match the
  manuscript's 0.011 as closely as logistic regression does.
- Several method free parameters were never recorded and are marked
  `REBUILD CHOICE` in the source.
- The texture statistic's window (9px) and shading sigma (31px) are choices. A
  sensitivity check over them would strengthen the specification.
- Sections 1 to 4 are findings of this rebuild, not reproductions. They should
  be reported as new work.
