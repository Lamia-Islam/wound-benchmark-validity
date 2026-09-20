# The clinical reference's verdict is a function of its framing

The framing result used to be a single reversal between two datasets. It is now
a monotonic curve with two held-out clinical datasets sitting on it.

---

## 1. What was measured

A clinical reference's "contrast verdict" is the trend of its rank agreement
with a synthetic benchmark as that benchmark's lesion-to-surround contrast is
swept. Positive means the reference rewards high contrast, negative means it
rewards low contrast. The seven contrast levels are the cached generator B
sweep, unchanged from earlier work, so nothing about the benchmarks varies here.

FUSeg was cropped around its own lesions to eight target lesion fractions
spanning the clinical range. Every crop uses the same 120 frames and the same
annotations, so lesion fraction is the only thing that differs between them.
Each crop was scored by all 22 methods and given its own verdict.

## 2. The curve

| target | achieved lesion fraction | n | contrast verdict | p |
|---|---|---|---|---|
| 0.005 | 0.0077 | 116 | +0.964 | 0.0034 |
| 0.010 | 0.0101 | 116 | +0.964 | 0.0034 |
| 0.020 | 0.0200 | 116 | +0.821 | 0.0347 |
| 0.035 | 0.0350 | 116 | +0.821 | 0.0347 |
| 0.055 | 0.0549 | 116 | +0.360 | 0.4268 |
| 0.080 | 0.0798 | 115 | -0.893 | 0.0132 |
| 0.110 | 0.1099 | 115 | -0.929 | 0.0073 |
| 0.160 | 0.1597 | 114 | -0.964 | 0.0034 |

Monotonicity of verdict against lesion fraction: **Spearman -0.988, p = 0.0001**.

The verdict changes sign between lesion fraction **0.055 and 0.080**. Below that
the reference rewards high lesion-to-surround contrast; above it, low contrast.
Nothing about the wounds, the imaging or the annotations differs across these
eight references. Only how much of the frame the lesion occupies.

## 3. Two held-out clinical datasets land on it

The curve is built entirely from cropped FUSeg. Medetec and AZH were not used to
build it. Their lesion fractions were measured, the curve was read at those
values, and only then were their verdicts computed.

| dataset | lesion fraction | predicted | measured | residual |
|---|---|---|---|---|
| FUSeg | 0.0109 | +0.953 | +0.964 | +0.011 |
| **AZH** | **0.0285** | **+0.821** | **+1.000** | **+0.179** |
| **Medetec** | **0.1024** | **-0.920** | **-0.893** | **+0.027** |

FUSeg is not an independent check, since the curve is made of FUSeg crops.

**Medetec is.** It is a different archive with different patients, different
photography and different annotators. A curve built by cropping one dataset
predicts its contrast verdict to within **0.027** on a scale spanning 1.93
units, and gets the sign right across the sign change.

**AZH is the second.** Residual +0.179, right sign, right region. It measures
+1.000, which is the maximum the statistic can take, so the curve cannot be
credited or blamed past that point.

## 4. Why this matters more than the reversal did

The old claim was: cropping a dataset can reverse its verdict. That is an
existence proof, and a reader can dismiss it as a quirk of one crop.

The new claim is: the verdict is a monotonic function of one measurable property
of the reference, the function crosses zero in the middle of the clinical range,
and two clinical datasets that had no part in building it fall on it.

That has predictive content. Given any clinical dataset's lesion fraction, the
curve states in advance which synthetic design it will call realistic. It also
says that a benchmark-design conclusion drawn against a single clinical
reference is a statement about that reference's curation, not about wounds, and
it gives the number at which the conclusion flips.

## 5. Caveats

- The verdict statistic is a Spearman over seven contrast levels, so it is
  coarse and saturates at +/-1. AZH sits at the saturation point, which is why
  its residual is not directly comparable to Medetec's.
- The seven contrast benchmarks are single-seed. Every point on the curve
  inherits whatever seed sensitivity that sweep has, and seed count has already
  overturned two conclusions in this project. The curve should be repeated
  across seeds before the manuscript states the sign-change value as a number.
- AZH and FUSeg come from the same clinic. Medetec does not, and Medetec is the
  stronger of the two held-out points for that reason.
- Crops above about 0.11 discard most of the surrounding limb, so the top of the
  curve is further from anything a clinician would photograph than the bottom.
- All of this is chronic foot ulcer photography. Nothing here establishes that
  lesion fraction is the controlling variable in another modality.
