# External validation broke the specification claim

I built generator C specifically to test whether the texture specification is a
property of images or a property of generator B's code. It is the latter. This
document records the test and what survives it.

---

## The test

Generator C shares no code path with generator B and differs in every mechanism:
1/f^beta spectral-filtered noise instead of upsampled Gaussian mottling, radial
vignette instead of a linear ramp, speckle and grain instead of hair and gauze,
distance-field alpha blending instead of filled polygons, a CIELAB Gaussian
mixture instead of an HSV offset from skin tone, and a spatial tissue gradient
instead of blob patches.

Both generators were swept over their own texture knob and both measured with
the same statistic. The comparison is on measured texture, not on matching knob
values, which is the whole point.

## Result 1: three benchmarks at matched texture disagree wildly

| benchmark | measured texture | mean rho vs clinical |
|---|---|---|
| generator A (uniform drape) | 2.10 | **0.121** |
| generator B, texture ablated | 2.14 | **0.434** |
| generator C, no noise | 2.34 | **0.189** |

Spread at matched texture: **0.312**.

For scale, the entire effect of sweeping texture within generator B, from 2.14
to 5.61, is **0.224**.

The between-generator spread at fixed texture is larger than the within-generator
effect of the property itself.

## Result 2: texture has no effect at all within generator C

Three seeds per level.

| measured texture | mean rho |
|---|---|
| 2.37 | 0.190 |
| 2.72 | 0.171 |
| 3.31 | 0.172 |
| 4.07 | 0.178 |
| 5.77 | 0.163 |

Flat across a texture range that, inside generator B, produced a
seed-confirmed rise from 0.431 to 0.536.

## Why

Generator C's top-scoring methods are `otsu_a` (0.750), `otsu_saturation`
(0.737) and `redness_ratio` (0.574). The last of those scores 0.048 on Medetec.
C rewards colour thresholding, because its background chroma is nearly constant
while its lesion chroma is distinct, so any chroma threshold separates them
regardless of how much luminance texture is present.

Measuring chroma texture separately makes this explicit:

| image set | luminance texture | chroma texture |
|---|---|---|
| Clinical Medetec | 2.60 | 0.50 |
| Clinical FUSeg | 3.88 | 0.45 |
| generator A | 2.10 | 2.44 |
| generator B (all levels) | 2.09 to 4.25 | 2.74 to 2.85 |
| generator C (all levels) | 2.33 to 5.67 | 0.00 to 0.52 |

Generator B's texture knob moves luminance texture while leaving chroma texture
pinned near 2.8, five times the clinical value. Generator C has chroma texture
close to clinical but no chroma structure at the scale that matters. Neither
generator resembles clinical images in both channels, and the single-channel
statistic cannot see the difference.

## What this means for the claim

**Withdrawn:** that background texture is a portable specification, that a
designer can measure it on their own benchmark and read off predictive validity,
and that the clinical target range is a sufficient design target. Three
benchmarks at matched texture span 0.121 to 0.434. The statistic does not
transfer across generator families.

**Retained, now correctly scoped:** within a fixed generator family, background
texture has a real effect. In generator B it moves rank agreement from 0.431 to
0.536 across five seeds with non-overlapping ranges (Mann-Whitney p = 0.0040).
That is a genuine within-family dose-response and it is stated as such.

**New, and stronger than what it replaces:** benchmark predictive validity is
not reducible to a single measured image statistic. We attempted the reduction
deliberately, with a statistic chosen for portability, and measured the residual:
the between-generator spread at fixed texture exceeds the within-generator
effect. This is a quantitative extension of the established finding that
rankings are fragile. Where prior work shows rank depends on the test set, this
shows that matching a measured realism property of the test set is not
sufficient to fix it.

## What is unaffected

- **The framing reversal.** Cropping FUSeg to Medetec's framing flips its
  contrast preference from +0.964 to -0.964. This concerns clinical references,
  not generators, and is untouched.
- **The single-feature mechanism.** Dropping local standard deviation of a*
  collapses uniform-background performance from 0.936 to 0.122 while clinical
  performance holds. This is a statement about generator A specifically and is
  untouched.
- **The clinical ceiling.** Medetec against FUSeg, rho = 0.567, p = 0.007.
- **The cross-seed texture effect within generator B**, correctly scoped.
- **The contrast null within generator B**, correctly scoped, and still
  un-repeated across seeds.

## What I would do next

1. Define the statistic per channel and per scale, as a profile rather than a
   number, and test whether profile distance to clinical predicts validity
   across generator families. The chroma table above suggests the profile is
   where the signal is; a single luminance number at one scale is not.
2. Fix generator B's chroma texture, which sits at five times the clinical
   value at every level and was never noticed because the statistic did not
   measure it.
3. Only then reconsider whether any portable specification is available.

## Honest note on what this cost

The specification claim was the strongest-sounding result in this project and it
did not survive the first test designed to break it. Finding that here is worth
more than finding it in review, but it does mean the paper's headline has to be
a negative result about reducibility rather than a positive result about a
target value.
