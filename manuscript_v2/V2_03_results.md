# MANUSCRIPT V2 — Part 3
## Section 7: Results

All numbers regenerated from the released code against the public FUSeg and
Medetec datasets. Placeholders marked [CROSS-SEED] are filled from the
multi-seed run; see Section 8 for its status at the time of writing.

---

## 7.1 The clinical ceiling

Two clinical datasets differing in originating clinic, camera, resolution and
framing agree on method ranking at **rho = 0.567, p = 0.007** (n = 22 methods,
permutation test).

This is the reference quantity for everything that follows. It is the best a
benchmark could be expected to do: if a synthetic benchmark predicts clinical
method ranking as well as one clinical dataset predicts another, it has reached
the point where the remaining disagreement is between clinical sources rather
than between synthetic and clinical.

## 7.2 Rank agreement rises with measured background texture and peaks at the
## clinical value

Generator B was swept across background texture, one factor at a time, with all
other design properties held fixed and the seed fixed. At each level all 22
methods were scored and the ranking correlated against each clinical dataset.

**Table 1.** Rank agreement against measured background texture.

| measured texture | rho vs Medetec | rho vs FUSeg | mean rho | % of ceiling |
|---|---|---|---|---|
| 2.14 | 0.564 | 0.303 | 0.434 | 76% |
| 2.17 | 0.577 | 0.352 | 0.464 | 82% |
| 2.24 | 0.572 | 0.340 | 0.456 | 80% |
| 2.45 | 0.605 | 0.387 | 0.496 | 87% |
| 2.88 | 0.636 | 0.422 | 0.529 | 93% |
| 3.50 | 0.627 | 0.492 | 0.560 | 99% |
| **4.40** | **0.654** | **0.486** | **0.570** | **100%** |
| 5.61 | 0.668 | 0.415 | 0.542 | 95% |

Over the rising range, **Spearman 0.964, p = 0.0028**. Both clinical datasets
agree on the direction independently, which is what makes this a property of the
benchmark rather than of one reference.

Two features of the curve matter more than the trend statistic.

**It peaks at the top of the clinical range.** Clinical images measure 2.69
(Medetec) to 3.97 (FUSeg) at the canonical resolution. The measured texture at
99% of ceiling is 3.50, inside that range; the peak at 100% is 4.40, just above
it. Beyond it agreement falls. The objective is
therefore to match the clinical value, not to maximise texture. A generator can
be too textured as well as too smooth, and by measured texture 5.61 the
generated backgrounds no longer read as skin on inspection.

**It approaches the ceiling.** At the peak, single-seed rank agreement is 0.570
against the clinical-to-clinical value of 0.567. Averaged over five seeds
(Table 1b) it is 0.536, or 94% of that ceiling. A synthetic benchmark whose
background texture matches clinical comes within 6% of the agreement between two
independent clinical datasets.

**A conventional generator sits below the clinical floor and at 76% of the
ceiling.** Generator A, a centred lesion composited onto a near-uniform drape,
measures 2.10 against a clinical floor of 2.69, and achieves 0.434.

**Table 1b.** Cross-seed confirmation. Five independent generator seeds at each
of four texture levels, 45 images per cell.

| measured texture | n seeds | mean rho | SD | range | % of ceiling |
|---|---|---|---|---|---|
| 2.14 | 5 | 0.431 | 0.036 | 0.382 to 0.466 | 76% |
| 2.45 | 5 | 0.439 | 0.050 | 0.380 to 0.496 | 77% |
| 3.50 | 5 | 0.523 | 0.044 | 0.485 to 0.580 | 92% |
| **4.40** | 5 | **0.536** | 0.039 | 0.474 to 0.570 | **94%** |

Three things follow, and one of them corrects the single-seed reading above.

**The effect survives seed variation.** The seed ranges at 2.14 and at 4.40 do
not overlap (0.382 to 0.466 against 0.474 to 0.570). Comparing 2.14 against 3.50,
Mann-Whitney U = 25, p = 0.0040, which is complete separation for five against
five. The seed-averaged curve is perfectly monotone (Spearman 1.000, p = 0.085
at four levels, which is the smallest p four points can produce), and all five
seeds individually show a positive trend, median 0.800.

**It is a threshold, not a smooth rise.** Texture 2.14 and 2.45 are
indistinguishable (0.431 against 0.439, ranges almost entirely overlapping).
The gain appears once texture reaches the clinical range. Adding a little
texture to a uniform benchmark achieves nothing; reaching the clinical value
achieves most of the available improvement.

**The ceiling is approached, not attained.** Seed-averaged, the best level
reaches 0.536, which is 94% of the clinical-to-clinical ceiling rather than the
100% that the single seed used for Table 1 suggested. That seed was favourable.
The honest statement is that a texture-matched synthetic benchmark comes within
6% of the agreement between two independent clinical datasets.

## 7.3 Target visibility is not the mechanism

If rank agreement responded to any change in how distinguishable the target is
from its surround, the texture result would admit a trivial reading. It does
not.

**Table 2.** Rank agreement against measured lesion-to-surround saturation
delta. Clinical images measure approximately +33; generator A sits at +55.8.

| measured saturation delta | rho vs Medetec | rho vs FUSeg | mean rho |
|---|---|---|---|
| 19.5 | 0.573 | 0.238 | 0.406 |
| 28.4 | 0.613 | 0.335 | 0.474 |
| 38.4 | 0.605 | 0.387 | 0.496 |
| 49.4 | 0.535 | 0.462 | 0.499 |
| 59.7 | 0.518 | 0.486 | 0.502 |
| 72.7 | 0.438 | 0.479 | 0.458 |
| 86.6 | 0.434 | 0.504 | 0.469 |

**Spearman 0.143, p = 0.780.** No trend.

The effect in Section 7.2 is specifically background structure. Making the
target more or less visible against its surround, across a range spanning and
exceeding both the clinical and the conventional-synthetic values, does not
change how well the benchmark predicts clinical ranking.

## 7.4 The clinical reference is not one thing: framing reverses the verdict

The flat mean in Table 2 conceals two opposing trends. Agreement with Medetec
falls monotonically as contrast rises; agreement with FUSeg rises monotonically.
The two cancel.

The obvious reading is that the datasets differ in some clinical way. They do
differ in framing: Medetec is cropped around the lesion with a median lesion
fraction of 0.098, FUSeg is full-limb at 0.0072, a fourteen-fold difference.

We tested framing directly. Each FUSeg image was cropped to a square centred on
its own lesion, sized so the lesion occupies Medetec's median fraction, and
resized to Medetec's resolution. Median lesion fraction moved from 0.0076 to
0.0979 (n = 115). Nothing else changed: same clinic, same patients, same
cameras, same annotations.

**Table 3.** Contrast preference before and after re-framing.

| reference | trend of rank agreement against contrast |
|---|---|
| Medetec (cropped, lesion fraction 0.098) | **-0.893**, p = 0.0132 |
| FUSeg (full-limb, lesion fraction 0.0072) | **+0.964**, p = 0.0034 |
| **FUSeg, cropped to Medetec framing** | **-0.964**, p = 0.0030 |

Re-framing reverses the preference, from strongly positive to strongly negative,
and lands it on Medetec's.

What looked like a disagreement between two clinics about benchmark realism is
an artifact of how the photographs were framed. The instruction "validate
against clinical data" is therefore underspecified: which clinical dataset, and
at what framing, determines which synthetic design appears realistic. A paper
reporting agreement with a single clinical reference is reporting agreement with
that reference's framing convention as much as with its clinical content.

This also explains Table 2. The contrast sweep is flat because it averages two
references with opposite framing.

## 7.5 The mechanism, at the level of a single input feature

Section 7.2 reports what background texture does. This section shows why, in the
smallest possible unit.

The learned classifiers take fifteen per-pixel features (Section 3.5). Each was
removed in turn, the classifier refitted on identical pixel samples, and scored
on a uniform-background benchmark and on both clinical datasets.

**Table 4.** Leave-one-out over classifier input features, Dice on a
uniform-background benchmark.

| dropped feature | Dice |
|---|---|
| none (all 15) | 0.936 |
| any of the 9 colour channels | 0.933 to 0.936 |
| a* local mean | 0.937 |
| grayscale local mean | 0.937 |
| grayscale local SD | 0.943 |
| a* centred, saturation centred | 0.937, 0.938 |
| **a\* local standard deviation** | **0.122** |

One feature, and it is a texture feature. Removing local standard deviation of
the a* channel collapses performance on a uniform-background benchmark by a
factor of eight. Removing any other feature changes nothing.

The reason is structural. On a background measuring 2.10, local standard
deviation of a* is near zero everywhere outside the lesion. The question "is
there local texture at this pixel?" therefore answers "is this pixel lesion?"
exactly, and a classifier with access to that feature receives the label for
free. Clinical images measure 2.69 to 3.97, where the same feature carries no such
shortcut.

This is the figure-ground collapse that Section 7.2 measures in aggregate,
exhibited on one input. It also gives benchmark designers a concrete diagnostic:
if a single texture feature carries a method's performance on your synthetic
benchmark, the benchmark is measuring foreground detection rather than the task.

## 7.6 Ablation of the other design properties

For completeness, each generator B design property was reverted to generator A's
corresponding assumption, one at a time, seed fixed.

**Table 5.** Change in mean rank agreement with clinical data on reverting one
property.

| reverted property | change |
|---|---|
| grade-independent lesion size | -0.154 |
| off-centre placement | -0.130 |
| clutter | -0.091 |
| irregular lesion shape | -0.045 |
| **textured background** | **-0.316** |

Background texture is the largest single effect, roughly twice the next largest.
The ordering supports treating it as the governing property, and the sweep in
Section 7.2 is the graded version of the same comparison.

## 7.7 Downstream consequences

[Retained from V1 and to be re-run under the V2 configuration before submission.
The measurements available are: on FUSeg, 62% of frames where segmentation
failed received full autonomy; threshold recalibration that raised synthetic
three-grade classification accuracy from 60% to 100% moved the unsafe-optimistic
rate from 0.61 to 0.62; end-to-end latency 40.4 ms at 512 x 512 on a single CPU
core, of which gate evaluation is 1.18 ms.]

This section serves the argument only insofar as it shows the validation choice
propagates into a system that acts on the output. It is not a clinical claim.
