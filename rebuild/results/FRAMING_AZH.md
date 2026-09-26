# The framing curve, rebuilt on the uncontaminated reference

The curve in FRAMING_V2.md is built from crops of FUSeg validation frames.
Eleven of the 32 methods were trained on FUSeg training images. The evaluation
images are disjoint but the distribution is not, which made that the weakest of
the paper's three main results.

This rebuilds it from AZH, the wound reference no method was trained on. The
contrast sweep is reused unchanged, four seeds, so only the reference differs.

---

## 1. The curve is cleaner on AZH than on FUSeg

Slope of rank agreement against contrast, per 10 saturation units. Seven crop
levels of AZH, four generator seeds.

| lesion fraction | Dice | sd | NSD | sd |
|---|---|---|---|---|
| 0.0353 | +0.077 | 0.006 | +0.080 | 0.005 |
| 0.0550 | +0.029 | 0.009 | +0.061 | 0.005 |
| 0.0801 | -0.002 | 0.010 | +0.038 | 0.005 |
| 0.1101 | -0.037 | 0.009 | +0.004 | 0.005 |
| 0.1598 | -0.060 | 0.008 | -0.041 | 0.005 |
| 0.2194 | -0.078 | 0.007 | -0.056 | 0.006 |
| 0.2981 | -0.086 | 0.007 | -0.079 | 0.007 |

Monotonicity against lesion fraction: **Spearman -1.000, p = 0.0005, in every
one of the four seeds, under both metrics.** Eight of eight seed and metric
combinations are perfectly monotonic.

The FUSeg-built curve gave -0.881 to -1.000 across the same eight combinations.
So removing the contamination did not weaken the relationship, it sharpened it.

Coverage differs. AZH ships as 224 px patches at a median lesion fraction of
0.029, and cropping can only raise a lesion fraction, so this curve runs from
0.035 to 0.298 where the FUSeg curve ran from 0.008 to 0.160. The two overlap
across 0.035 to 0.160, which contains both crossovers.

## 2. The crossover is metric-dependent as well as pool-dependent

| curve | metric | sign change |
|---|---|---|
| FUSeg-built | Dice | 0.110 to 0.160 |
| FUSeg-built | NSD | 0.110 to 0.160 |
| AZH-built | Dice | 0.055 to 0.080 |
| AZH-built | NSD | 0.110 to 0.160 |

Three of the four agree on 0.110 to 0.160. The AZH curve under Dice crosses
earlier.

This strengthens rather than complicates the paper's position, because that
position is already that the crossover is not a portable constant. It moves with
the method pool, it moves with the reference, and it moves with the metric. The
direction of the relationship does not move at all.

## 3. Held-out placement is worse, and the reason is informative

Medetec, lesion fraction 0.1024, placed on the AZH-built curve:

| metric | predicted | measured | sd over seeds | residual | as % of range |
|---|---|---|---|---|---|
| Dice | -0.028 | +0.038 | 0.004 | +0.066 | 40.6% |
| NSD | +0.013 | +0.056 | 0.006 | +0.043 | 27.0% |

On the FUSeg-built curve the same residuals were 16.6% and 19.4%. So the
uncontaminated curve predicts Medetec less well.

Medetec sits **above** both curves. So did AZH and FUSeg when placed on the
FUSeg-built curve. Across two independently built curves and three held-out
datasets, every real clinical dataset lands above the cropped-reference curve at
its own lesion fraction. Four of four placements, one direction.

That is systematic and it has a plain explanation. A crop taken to reach lesion
fraction f is not equivalent to a dataset natively framed at f. Cropping to a
high lesion fraction throws away the surrounding limb, background and clutter,
whereas a dataset that is natively framed tightly still contains whatever
context the photographer chose to include. Crops therefore overstate the effect
of the framing convention, and real datasets fall short of the curve built from
them.

The consequence for the paper is a restriction rather than a retraction. The
crop series establishes the direction and the ordering, which is what the result
claims. It is not a calibrated predictor of a natively framed dataset's verdict,
and the offset is now characterised rather than unexplained.

## 4. What changes in the manuscript

- Section 7.4 should be built on the AZH curve, with the FUSeg curve reported
  alongside as the contaminated replication.
- The monotonicity figure improves from -0.881 to -1.000 as the worst case, and
  becomes exact in all eight seed and metric combinations.
- The one-directional offset, previously listed as unexplained, becomes a stated
  limitation of crop-built curves with a mechanism.
- The held-out residual should be quoted as 27% to 41% of range on the
  uncontaminated curve, not 17% to 19% on the contaminated one.

## 5. Caveats

- AZH cannot reach lesion fractions below about 0.035, so the low-fraction end
  of the FUSeg curve has no uncontaminated counterpart.
- Only one held-out dataset is available for the AZH curve. FUSeg at 0.011 is
  below the range and is not placed.
- The crops use ground-truth mask centroid and area. That is unchanged from the
  FUSeg version and remains oracle localisation.
- Seven crop levels, 140 usable images each, four seeds, seven contrast levels.
