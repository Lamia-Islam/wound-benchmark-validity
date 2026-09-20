# Rebuild status after run 2 (four benchmarks + ablation)

All 22 methods, both generators, the five ablations and the evaluation harness
are rebuilt and running. Datasets are the real public FUSeg and Medetec.

---

## 1. Exact verifications

| check | manuscript | rebuild |
|---|---|---|
| U-Net parameter count | 117,681 | **117,681** |
| learned decision threshold | 0.9 | **0.90** (independently swept) |
| Medetec median lesion fraction | 0.098 | **0.0980** |
| Medetec mask encoding | 0/1 | 0/1 |
| FUSeg mask encoding | 0/255 | 0/255 |
| Medetec / FUSeg resolution | 224 / 512 | 224 / 512 |
| Medetec vs FUSeg rank agreement | rho 0.560, p 0.008 | **rho 0.567, p 0.007** |

The U-Net count was diagnostic rather than cosmetic. With bias=False on the
convolutions the model totals 117,361. The 320 shortfall is exactly the ten
convolution bias vectors, so the original used biased convolutions.

## 2. Table 3, rank agreement across four benchmarks

| comparison | manuscript | rebuild | verdict |
|---|---|---|---|
| Medetec vs FUSeg | +0.560, p 0.008 | **+0.567, p 0.007** | reproduced |
| generator B vs Medetec | +0.439, p 0.040 | **+0.605, p 0.003** | reproduced, stronger |
| generator B vs FUSeg | +0.560, p 0.007 | +0.387, p 0.076 | same sign, weaker, not significant |
| generator A vs Medetec | −0.121, p 0.588 | +0.342, p 0.121 | not reproduced (both non-significant) |
| generator A vs FUSeg | −0.466, p 0.033 | −0.099, p 0.663 | same sign, much weaker |
| **generator A vs generator B** | **−0.526, p 0.014** | **+0.456, p 0.034** | **not reproduced, sign flipped** |

## 3. Table 5, the ablation. The central claim reproduces.

Change in mean rank correlation against clinical data when one generator B
feature is reverted to generator A's assumption:

| ablated feature | rebuild | manuscript |
|---|---|---|
| grade-independent lesion size | −0.154 | −0.073 |
| off-centre placement | −0.130 | −0.079 |
| clutter | −0.091 | −0.064 |
| irregular lesion shape | −0.045 | +0.020 |
| **textured background** | **−0.316** | **−0.532** |

Background texture is the largest single effect, and removing it drives the
FUSeg correlation negative (−0.056, manuscript −0.268). The ordering the paper
depends on holds: texture dominates, and no other feature comes close.

Differences: the effect is about 60% of the reported magnitude, and texture
leads the next-largest feature by 2x rather than 7x.

## 4. Two attributed discrepancies

### 4a. The learned classifiers score far too high on generator A

The three pixel classifiers score 0.80 to 0.93 on generator A. The manuscript
records 0.166 for logistic regression and 0.011 for gradient boosting.

Counterfactual, substituting the manuscript's learned rows into this rebuild's
generator A scores and recomputing:

| comparison | rebuild | counterfactual | manuscript |
|---|---|---|---|
| generator A vs FUSeg | −0.099 | **−0.359** | −0.466 |
| generator A vs Medetec | +0.342 | +0.159 | −0.121 |
| generator A vs generator B | +0.456 | +0.259 | −0.526 |

So this one discrepancy accounts for roughly half the generator-A gap. Ruled
out as causes: over-segmentation (predicting the whole frame gives Dice 0.241
on generator A, not 0.166) and the self-referenced features (the colour-only
variant also scores 0.957). The 14-versus-15 feature discrepancy flagged during
recovery remains the leading suspect and is now known to be load-bearing rather
than cosmetic.

### 4b. Generator B still rewards some figure-ground methods

Even with the learned rows substituted, generator A vs generator B stays
positive. Leaving the learned family out entirely makes it more positive
(+0.637), so they are not the cause of the sign flip.

Generator B does work directionally. Family mean Dice:

| family | gen A | gen B | Medetec | FUSeg |
|---|---|---|---|---|
| background-referenced | 0.448 | **0.179** | 0.231 | 0.049 |
| clustering | 0.778 | 0.258 | 0.310 | 0.046 |
| colour-referenced | 0.922 | 0.489 | 0.276 | 0.065 |
| region/contour | 0.683 | 0.342 | 0.349 | 0.086 |
| CNN | 0.042 | **0.596** | 0.272 | 0.626 |

The corner-referenced family collapses and the U-Net recovers, both as the paper
describes. What remains: GrabCut holds 0.859 on generator B against 0.057 on
FUSeg, and mean-shift holds 0.759 against 0.068. Those two keep generator B's
ranking closer to generator A's than the original's apparently was.

## 5. Generator B calibration, and why it is not method-tuning

The first Generator B had HIGHER lesion-to-skin colour contrast than generator
A, which let simple colour thresholding beat it. The contrast was recalibrated
against measured clinical values rather than against any method's behaviour,
per section 4.2's stated design principle.

Lesion versus 41px perilesional ring:

| source | HSV saturation delta | CIELAB a* delta |
|---|---|---|
| Medetec (n=152) | +29.6 | +4.7 |
| FUSeg (n=114) | +36.8 | +10.3 |
| **clinical target** | **~+33** | **~+8** |
| generator A | +55.8 | +16.5 |
| generator B, first attempt | +72.6 | +18.5 |
| generator B, calibrated | **+38.4** | **+12.8** |

## 6. Bugs found and fixed in the rebuild

- **Rainbow background.** Generating the low-frequency mottling independently
  per colour channel decorrelates chroma and produces pastel rainbow blotches,
  making the background more saturated than the lesion. Fixed by carrying most
  of the amplitude in a shared luminance field.
- **Green skin.** Sampling B, G and R independently let blue exceed red. Fixed
  by drawing one depth parameter along a fixed skin chromaticity, so R > G > B
  holds on all 45 images.
- **Degenerate redness ratio.** R/(G+B) is unbounded; pixels where G+B is near
  zero send it to ~1e6 and a min-max rescale collapses the image to five grey
  levels, returning an empty mask on every image. Fixed by clipping to the
  1st-99th percentile. Dice went 0.000 to 0.430 on Medetec.

## 7. Where the paper's claims now stand

| claim | status |
|---|---|
| Two independent clinical datasets agree on method ranking | **reproduced** (0.567 vs 0.560) |
| Generator B predicts clinical ranking | **reproduced** on Medetec (0.605, p 0.003); same sign but not significant on FUSeg (0.387, p 0.076) |
| Background texture is the design property governing rank transfer | **reproduced qualitatively**, at ~60% of the reported magnitude |
| A uniform-background benchmark inverts the CNN's standing | **reproduced** (U-Net 0.042 gen A, 0.596 gen B, 0.626 FUSeg) |
| Generator A is significantly negatively correlated with clinical ranking | **not reproduced** (−0.099, p 0.663; −0.359 under the counterfactual) |
| The two synthetic benchmarks rank methods in significantly opposite orders | **not reproduced**, sign flipped (+0.456) |

## 8. Next, in order

1. Resolve 4a. Run the 14-feature ablation: drop each of the 15 features in turn,
   refit, and score on generator A. If one drop sends the learned methods to
   ~0.166 on generator A while holding their clinical scores, that identifies
   the original feature set.
2. Resolve 4b. Diagnose why GrabCut and mean-shift survive generator B. Both are
   region-growing from a central or smoothed initialisation, and generator B's
   lesions may still be too compact and too centrally placed relative to
   clinical wounds.
3. Only then decide what the paper can claim. On current evidence the headline
   would have to be restated: generator A fails to predict clinical ranking
   (non-significant, both datasets) rather than inverting it.
