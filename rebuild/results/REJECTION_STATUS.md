# Rejection analysis: what has been closed

Status against REJECTION_ANALYSIS.md, after the work of 25 and 26 September.

## Tier 1

| # | objection | status |
|---|---|---|
| 1 | method pool is obsolete, 21 of 22 classical | **closed**. Pool is 32, with ten modern architectures trained from random initialisation under the project's own protocol. Deep models are the top performers on every wound reference. |
| 2 | circularity, generators tuned on the same references they are validated against | **partly closed**. AZH is now the neutral reference no design decision touched, and headline numbers can be reported against it. Generator C and the untuned profile descriptor remain the other defences. |
| 3 | n = 22, confidence intervals swallow the claims | **closed**. n = 32, detection threshold falls from 0.428 to 0.352, and all three same-modality pairs are now significant with intervals excluding zero. |
| 4 | the framing curve is a saturating statistic wearing a trend | **closed, and it was worse than the objection said.** At n = 32 the rank statistic pinned at exactly +1.000 for five of eight crop levels. Replaced with the OLS slope of agreement on contrast, which is continuous. The effect survives and is cleaner. |
| 5 | every framing result rests on one seed | **closed**. Four seeds, standard deviation never above 0.008 on a range of 0.154, monotonic at p < 0.01 in every seed. |

## Tier 2

| # | objection | status |
|---|---|---|
| 6 | Dice only, and Dice's region-size bias is confounded with the framing experiment | **closed**. Normalised surface Dice at 2 px gives the same curve, the same monotonicity and the same sign-change interval. This was the most dangerous objection and it is answered. |
| 7 | the crop uses the ground-truth mask, which is oracle localisation | **acknowledged, not fixable**. Stated as a construction choice rather than a proposed pipeline. |
| 8 | five references are really three sources | **acknowledged**. Reported as three. A second endoscopy source is still wanted. |
| 9 | the three generator families are not independent | **open**. Language needs softening in the manuscript. |
| 10 | no comparison with generative models | **open**. Needs one honest paragraph on why parametric generators, not a diffusion model. |

## What the enlarged pool changed in the findings

Two conclusions moved, both caught rather than published.

**The cross-modality inversion is a composition artifact.** The pooled
across-modality correlation of -0.253 does not exist within either subgroup
(+0.108 classical, +0.236 deep). Deep models trained on wound photography beat
classical methods on wounds and collapse below them on endoscopy, and two groups
swapping places produces a negative pooled correlation without any pair of
methods inverting. The surviving claim is the original one, no detectable
cross-modality agreement within a homogeneous pool, plus a new warning that
mixed pools measure training-domain match rather than anything about the data.

**The framing crossover is not a portable constant.** It moved from the 0.055 to
0.080 interval at n = 22 to the 0.110 to 0.160 interval at n = 32. The
relationship is robust; the threshold value is a property of the method pool as
well as the references and must not be quoted for others to apply.

**The held-out placement is weaker than claimed.** Medetec's residual was
reported as 0.027 on a 1.93 scale. Under the non-saturating statistic it is 17%
of the curve's range under Dice and 19% under NSD. Still a real out-of-sample
placement with the right sign and magnitude, no longer a bullseye.

## Remaining, in order

1. FUSeg is no longer a neutral reference for this pool, since eleven of the 32
   methods were trained on its training split. Headline agreement should be
   quoted as Medetec versus AZH, +0.568.
2. A second, independent endoscopy source.
3. Rewrite the manuscript around the cross-modality hierarchy, with the
   synthetic benchmark as the case study, and with the framing result as the
   within-modality finding.
4. Soften the independence language on the generator families, and add the
   paragraph on why parametric generators rather than a diffusion model.
