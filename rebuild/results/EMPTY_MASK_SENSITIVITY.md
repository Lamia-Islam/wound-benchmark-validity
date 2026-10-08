# Empty ground-truth masks: do they move the agreement hierarchy?

Every clinical reference was rescored with all 32 methods, keeping the Dice and NSD (tau = 2 px) of every method on every image (results/perimage32/). Rank agreement between two references is the Spearman correlation of the 32 per-method mean scores. p is a two-sided permutation test (20000 permutations). The 95% CI is a bootstrap over methods (5000 resamples, seed 0).

Two corrections are built in. First, the two Kvasir references no longer share images: Kvasir (disjoint) is the first 120 Kvasir-SEG images that are not in the sessile subset, so it shares no image with Kvasir sessile. The old selection shared 31. Second, FUSeg and AZH come from the same clinic, so FUSeg vs AZH is now labelled same source, not same modality.

Three image filters are compared: all images; excluding images whose ground-truth mask is empty; and also excluding images whose lesion covers less than 0.1% of the image.

## 1. How many empty and near-empty masks there are

| reference | n | empty GT | lesion fraction in (0, 0.001) | n after both exclusions | median lesion fraction |
|---|---|---|---|---|---|
| Medetec | 152 | 0 | 0 | 152 | 0.0980 |
| FUSeg | 120 | 4 | 8 | 108 | 0.0076 |
| AZH test | 150 | 10 | 3 | 137 | 0.0190 |
| AZH train slice | 150 | 8 | 3 | 139 | 0.0159 |
| Kvasir (disjoint) | 120 | 0 | 0 | 120 | 0.1278 |
| Kvasir sessile | 120 | 0 | 0 | 120 | 0.0427 |

## 2. Corrected hierarchy under Dice

| level | pair | all images: rho, p, CI | excluding empty GT: rho, p, CI | excluding empty and < 0.001: rho, p, CI |
|---|---|---|---|---|
| same dataset | AZH test vs AZH train slice | +0.944, <0.0001, [+0.835, +0.984] | +0.920, <0.0001, [+0.784, +0.975] | +0.920, <0.0001, [+0.784, +0.975] |
| same source | FUSeg vs AZH test | +0.660, 0.0001, [+0.430, +0.804] | +0.368, 0.0408, [+0.044, +0.630] | +0.360, 0.0451, [+0.033, +0.621] |
| same source | Kvasir (disjoint) vs Kvasir sessile | +0.950, <0.0001, [+0.864, +0.980] | +0.950, <0.0001, [+0.864, +0.980] | +0.950, <0.0001, [+0.864, +0.980] |
| different source, same modality | Medetec vs FUSeg | +0.461, 0.0081, [+0.085, +0.758] | +0.478, 0.0067, [+0.118, +0.761] | +0.466, 0.0083, [+0.102, +0.757] |
| different source, same modality | Medetec vs AZH test | +0.568, 0.0012, [+0.256, +0.781] | +0.510, 0.0038, [+0.165, +0.750] | +0.510, 0.0038, [+0.165, +0.750] |
| across modality | Medetec vs Kvasir (disjoint) | +0.032, 0.8589, [-0.310, +0.373] | +0.032, 0.8589, [-0.310, +0.373] | +0.032, 0.8589, [-0.310, +0.373] |
| across modality | Medetec vs Kvasir sessile | -0.042, 0.8185, [-0.397, +0.338] | -0.042, 0.8185, [-0.397, +0.338] | -0.042, 0.8185, [-0.397, +0.338] |
| across modality | FUSeg vs Kvasir (disjoint) | -0.481, 0.0058, [-0.705, -0.157] | -0.476, 0.0061, [-0.708, -0.147] | -0.484, 0.0053, [-0.716, -0.150] |
| across modality | FUSeg vs Kvasir sessile | -0.552, 0.0011, [-0.741, -0.224] | -0.547, 0.0016, [-0.743, -0.210] | -0.551, 0.0015, [-0.752, -0.210] |
| across modality | AZH test vs Kvasir (disjoint) | -0.203, 0.2680, [-0.512, +0.159] | +0.026, 0.8876, [-0.327, +0.381] | +0.026, 0.8876, [-0.327, +0.381] |
| across modality | AZH test vs Kvasir sessile | -0.277, 0.1289, [-0.579, +0.081] | -0.051, 0.7814, [-0.403, +0.324] | -0.051, 0.7814, [-0.403, +0.324] |

Level means:

| level | all images | excluding empty GT | excluding empty and < 0.001 |
|---|---|---|---|
| same dataset | +0.9439 | +0.9197 | +0.9197 |
| same source | +0.8048 | +0.6587 | +0.6549 |
| different source, same modality | +0.5143 | +0.4940 | +0.4879 |
| across modality | -0.2537 | -0.1762 | -0.1781 |

## 3. Corrected hierarchy under NSD

| level | pair | all images: rho, p, CI | excluding empty GT: rho, p, CI | excluding empty and < 0.001: rho, p, CI |
|---|---|---|---|---|
| same dataset | AZH test vs AZH train slice | +0.960, <0.0001, [+0.890, +0.982] | +0.962, <0.0001, [+0.897, +0.983] | +0.964, <0.0001, [+0.903, +0.982] |
| same source | FUSeg vs AZH test | +0.782, <0.0001, [+0.533, +0.931] | +0.712, <0.0001, [+0.430, +0.891] | +0.713, <0.0001, [+0.432, +0.890] |
| same source | Kvasir (disjoint) vs Kvasir sessile | +0.937, <0.0001, [+0.847, +0.974] | +0.937, <0.0001, [+0.847, +0.974] | +0.937, <0.0001, [+0.847, +0.974] |
| different source, same modality | Medetec vs FUSeg | +0.461, 0.0087, [+0.074, +0.757] | +0.460, 0.0088, [+0.073, +0.764] | +0.465, 0.0078, [+0.087, +0.763] |
| different source, same modality | Medetec vs AZH test | +0.580, 0.0007, [+0.232, +0.816] | +0.631, 0.0002, [+0.318, +0.833] | +0.631, 0.0002, [+0.316, +0.832] |
| across modality | Medetec vs Kvasir (disjoint) | +0.266, 0.1392, [-0.069, +0.549] | +0.266, 0.1392, [-0.069, +0.549] | +0.266, 0.1392, [-0.069, +0.549] |
| across modality | Medetec vs Kvasir sessile | +0.353, 0.0462, [+0.038, +0.625] | +0.353, 0.0462, [+0.038, +0.625] | +0.353, 0.0462, [+0.038, +0.625] |
| across modality | FUSeg vs Kvasir (disjoint) | -0.284, 0.1169, [-0.628, +0.082] | -0.282, 0.1195, [-0.629, +0.084] | -0.279, 0.1244, [-0.625, +0.085] |
| across modality | FUSeg vs Kvasir sessile | -0.132, 0.4658, [-0.477, +0.238] | -0.132, 0.4665, [-0.482, +0.238] | -0.128, 0.4809, [-0.474, +0.241] |
| across modality | AZH test vs Kvasir (disjoint) | -0.029, 0.8764, [-0.359, +0.274] | +0.095, 0.6096, [-0.237, +0.390] | +0.097, 0.6001, [-0.237, +0.396] |
| across modality | AZH test vs Kvasir sessile | +0.084, 0.6457, [-0.240, +0.373] | +0.215, 0.2341, [-0.098, +0.481] | +0.217, 0.2286, [-0.098, +0.487] |

Level means:

| level | all images | excluding empty GT | excluding empty and < 0.001 |
|---|---|---|---|
| same dataset | +0.9604 | +0.9619 | +0.9637 |
| same source | +0.8596 | +0.8248 | +0.8253 |
| different source, same modality | +0.5203 | +0.5456 | +0.5478 |
| across modality | +0.0430 | +0.0858 | +0.0878 |

## 4. Every pair of references

Under Dice:

| pair | all images | excluding empty GT | excluding empty and < 0.001 |
|---|---|---|---|
| Medetec vs FUSeg | +0.461 (p 0.0081; [+0.085, +0.758]) | +0.478 (p 0.0067; [+0.118, +0.761]) | +0.466 (p 0.0083; [+0.102, +0.757]) |
| Medetec vs AZH test | +0.568 (p 0.0012; [+0.256, +0.781]) | +0.510 (p 0.0038; [+0.165, +0.750]) | +0.510 (p 0.0038; [+0.165, +0.750]) |
| Medetec vs AZH train slice | +0.571 (p 0.0008; [+0.231, +0.797]) | +0.524 (p 0.0028; [+0.160, +0.774]) | +0.524 (p 0.0028; [+0.160, +0.774]) |
| Medetec vs Kvasir (disjoint) | +0.032 (p 0.8589; [-0.310, +0.373]) | +0.032 (p 0.8589; [-0.310, +0.373]) | +0.032 (p 0.8589; [-0.310, +0.373]) |
| Medetec vs Kvasir sessile | -0.042 (p 0.8185; [-0.397, +0.338]) | -0.042 (p 0.8185; [-0.397, +0.338]) | -0.042 (p 0.8185; [-0.397, +0.338]) |
| FUSeg vs AZH test | +0.660 (p 0.0001; [+0.430, +0.804]) | +0.368 (p 0.0408; [+0.044, +0.630]) | +0.360 (p 0.0451; [+0.033, +0.621]) |
| FUSeg vs AZH train slice | +0.536 (p 0.0021; [+0.265, +0.722]) | +0.288 (p 0.1090; [-0.061, +0.577]) | +0.282 (p 0.1170; [-0.068, +0.573]) |
| FUSeg vs Kvasir (disjoint) | -0.481 (p 0.0058; [-0.705, -0.157]) | -0.476 (p 0.0061; [-0.708, -0.147]) | -0.484 (p 0.0053; [-0.716, -0.150]) |
| FUSeg vs Kvasir sessile | -0.552 (p 0.0011; [-0.741, -0.224]) | -0.547 (p 0.0016; [-0.743, -0.210]) | -0.551 (p 0.0015; [-0.752, -0.210]) |
| AZH test vs AZH train slice | +0.944 (p <0.0001; [+0.835, +0.984]) | +0.920 (p <0.0001; [+0.784, +0.975]) | +0.920 (p <0.0001; [+0.784, +0.975]) |
| AZH test vs Kvasir (disjoint) | -0.203 (p 0.2680; [-0.512, +0.159]) | +0.026 (p 0.8876; [-0.327, +0.381]) | +0.026 (p 0.8876; [-0.327, +0.381]) |
| AZH test vs Kvasir sessile | -0.277 (p 0.1289; [-0.579, +0.081]) | -0.051 (p 0.7814; [-0.403, +0.324]) | -0.051 (p 0.7814; [-0.403, +0.324]) |
| AZH train slice vs Kvasir (disjoint) | -0.130 (p 0.4795; [-0.462, +0.250]) | +0.071 (p 0.7016; [-0.325, +0.449]) | +0.071 (p 0.7016; [-0.325, +0.449]) |
| AZH train slice vs Kvasir sessile | -0.214 (p 0.2445; [-0.538, +0.154]) | -0.002 (p 0.9916; [-0.406, +0.381]) | -0.002 (p 0.9916; [-0.406, +0.381]) |
| Kvasir (disjoint) vs Kvasir sessile | +0.950 (p <0.0001; [+0.864, +0.980]) | +0.950 (p <0.0001; [+0.864, +0.980]) | +0.950 (p <0.0001; [+0.864, +0.980]) |

Under NSD:

| pair | all images | excluding empty GT | excluding empty and < 0.001 |
|---|---|---|---|
| Medetec vs FUSeg | +0.461 (p 0.0087; [+0.074, +0.757]) | +0.460 (p 0.0088; [+0.073, +0.764]) | +0.465 (p 0.0078; [+0.087, +0.763]) |
| Medetec vs AZH test | +0.580 (p 0.0007; [+0.232, +0.816]) | +0.631 (p 0.0002; [+0.318, +0.833]) | +0.631 (p 0.0002; [+0.316, +0.832]) |
| Medetec vs AZH train slice | +0.621 (p 0.0002; [+0.303, +0.833]) | +0.688 (p <0.0001; [+0.411, +0.851]) | +0.680 (p <0.0001; [+0.395, +0.849]) |
| Medetec vs Kvasir (disjoint) | +0.266 (p 0.1392; [-0.069, +0.549]) | +0.266 (p 0.1392; [-0.069, +0.549]) | +0.266 (p 0.1392; [-0.069, +0.549]) |
| Medetec vs Kvasir sessile | +0.353 (p 0.0462; [+0.038, +0.625]) | +0.353 (p 0.0462; [+0.038, +0.625]) | +0.353 (p 0.0462; [+0.038, +0.625]) |
| FUSeg vs AZH test | +0.782 (p <0.0001; [+0.533, +0.931]) | +0.712 (p <0.0001; [+0.430, +0.891]) | +0.713 (p <0.0001; [+0.432, +0.890]) |
| FUSeg vs AZH train slice | +0.761 (p <0.0001; [+0.487, +0.926]) | +0.653 (p 0.0001; [+0.328, +0.864]) | +0.663 (p 0.0001; [+0.346, +0.867]) |
| FUSeg vs Kvasir (disjoint) | -0.284 (p 0.1169; [-0.628, +0.082]) | -0.282 (p 0.1195; [-0.629, +0.084]) | -0.279 (p 0.1244; [-0.625, +0.085]) |
| FUSeg vs Kvasir sessile | -0.132 (p 0.4658; [-0.477, +0.238]) | -0.132 (p 0.4665; [-0.482, +0.238]) | -0.128 (p 0.4809; [-0.474, +0.241]) |
| AZH test vs AZH train slice | +0.960 (p <0.0001; [+0.890, +0.982]) | +0.962 (p <0.0001; [+0.897, +0.983]) | +0.964 (p <0.0001; [+0.903, +0.982]) |
| AZH test vs Kvasir (disjoint) | -0.029 (p 0.8764; [-0.359, +0.274]) | +0.095 (p 0.6096; [-0.237, +0.390]) | +0.097 (p 0.6001; [-0.237, +0.396]) |
| AZH test vs Kvasir sessile | +0.084 (p 0.6457; [-0.240, +0.373]) | +0.215 (p 0.2341; [-0.098, +0.481]) | +0.217 (p 0.2286; [-0.098, +0.487]) |
| AZH train slice vs Kvasir (disjoint) | +0.076 (p 0.6798; [-0.268, +0.382]) | +0.227 (p 0.2083; [-0.109, +0.503]) | +0.217 (p 0.2319; [-0.120, +0.496]) |
| AZH train slice vs Kvasir sessile | +0.177 (p 0.3303; [-0.158, +0.467]) | +0.339 (p 0.0583; [+0.026, +0.579]) | +0.330 (p 0.0659; [+0.013, +0.575]) |
| Kvasir (disjoint) vs Kvasir sessile | +0.937 (p <0.0001; [+0.847, +0.974]) | +0.937 (p <0.0001; [+0.847, +0.974]) | +0.937 (p <0.0001; [+0.847, +0.974]) |

## 5. Ties

Number of methods whose mean score is exactly equal to at least one other method's mean score (Spearman gives tied methods the average rank).

| reference | Dice, all | Dice, excl. empty | Dice, excl. both | NSD, all | NSD, excl. empty | NSD, excl. both |
|---|---|---|---|---|---|---|
| Medetec | 0 | 0 | 0 | 0 | 0 | 0 |
| FUSeg | 0 | 0 | 0 | 0 | 0 | 0 |
| AZH test | 0 | 0 | 0 | 0 | 0 | 0 |
| AZH train slice | 0 | 0 | 0 | 0 | 0 | 0 |
| Kvasir (disjoint) | 0 | 0 | 0 | 0 | 0 | 0 |
| Kvasir sessile | 0 | 0 | 0 | 0 | 0 | 0 |

What the methods do on the empty-GT images themselves (on such an image Dice and NSD are 1 if the method predicts nothing and 0 otherwise):

| reference | empty-GT images | methods that never leave one empty | methods that always leave them empty | distinct per-image score patterns across the 32 methods |
|---|---|---|---|---|
| Medetec | 0 | | | |
| FUSeg | 4 | 20 | 0 | 5 |
| AZH test | 10 | 18 | 6 | 9 |
| AZH train slice | 8 | 16 | 7 | 10 |
| Kvasir (disjoint) | 0 | | | |
| Kvasir sessile | 0 | | | |

## 6. The pool split, repeated on the new data

Mean Spearman over the six across-modality pairs (each wound reference against each Kvasir reference), within each part of the pool. The earlier overlapping-Kvasir values were -0.253 (all 32), +0.108 (22 non-deep) and +0.236 (10 deep).

Under Dice:

| subset | all images | excluding empty GT | excluding empty and < 0.001 |
|---|---|---|---|
| all 32 | -0.2537 | -0.1762 | -0.1781 |
| non-deep 22 | +0.1137 | +0.1171 | +0.1171 |
| deep 10 | +0.2545 | +0.2384 | +0.2040 |

Under NSD:

| subset | all images | excluding empty GT | excluding empty and < 0.001 |
|---|---|---|---|
| all 32 | +0.0430 | +0.0858 | +0.0878 |
| non-deep 22 | +0.4581 | +0.4589 | +0.4598 |
| deep 10 | +0.2687 | +0.2182 | +0.2364 |

All four levels by subset (Dice, all images):

| subset | same dataset | same source | different source, same modality | across modality |
|---|---|---|---|---|
| all 32 | +0.9439 | +0.8048 | +0.5143 | -0.2537 |
| non-deep 22 | +0.9639 | +0.7216 | +0.5093 | +0.1137 |
| deep 10 | +0.9394 | +0.5939 | +0.3515 | +0.2545 |

## 7. Verdict

**Where empty masks are.** FUSeg has 4 of 120, AZH test has 10 of 150, AZH train slice has 8 of 150. Medetec and both Kvasir references have none, and no Kvasir lesion is below 0.001 of the image.

**Yes, excluding empty masks changed one conclusion, under Dice only.** The wound same-source pair, FUSeg vs AZH test, falls from +0.660 (p 0.0001) to +0.368 (p 0.0408). That puts it below both different-source wound pairs (Medetec vs FUSeg +0.478 (p 0.0067), Medetec vs AZH test +0.510 (p 0.0038)), so under Dice the claim that same-source wound references agree more than different-source ones holds only while the empty masks are left in. FUSeg vs AZH train slice falls from +0.536 (p 0.0021) to +0.288 (p 0.1090) and stops being significant. Under NSD the same pair moves less (+0.782 (p <0.0001) to +0.712 (p <0.0001)) and stays above both different-source pairs (+0.460 (p 0.0088) and +0.631 (p 0.0002)).

**Why.** On an empty-GT image a method scores 1 if it predicts nothing and 0 otherwise. On AZH test, 10 of the 10 deep models leave at least 70% of the 10 empty images empty, while 16 of the 18 untrained classical methods never leave one empty. That is a bonus of up to 0.0667 mean Dice (10 of 150 images) that goes almost entirely to the trained methods. The trained methods also lead on FUSeg (the deep models were trained on FUSeg training images), so the bonus pushes the AZH ranking toward the FUSeg ranking. Remove the empty images and the deep models on AZH drop back among the classical ones. The empty images create no exact ties in any mean (section 5); their effect is a shared shift, not a tie.

**The four-level order itself survives.** Under Dice the level means go from +0.9439, +0.8048, +0.5143, -0.2537 (all images) to +0.9197, +0.6587, +0.4940, -0.1762 (excluding empty GT), for same dataset, same source, different source same modality, and across modality. The order is unchanged, but the same-source level now rests on the Kvasir pair; its wound half sits below the level beneath it.

**Across modality.** The Dice mean moves from -0.2537 to -0.1762, because the AZH test vs Kvasir pairs move from weakly negative to about zero (-0.203 (p 0.2680) to +0.026 (p 0.8876); -0.277 (p 0.1289) to -0.051 (p 0.7814)). No across-modality pair crosses p = 0.05. The two FUSeg vs Kvasir pairs stay significantly negative (-0.476 (p 0.0061), -0.547 (p 0.0016)).

**The pool split still holds.** Across-modality mean under Dice, all images: all 32 -0.2537, 22 non-deep +0.1137, 10 deep +0.2545. Excluding empty GT: -0.1762, +0.1171, +0.2384. Under Dice the pooled value is negative and both subsets are positive in all three conditions, so the negative pooled number is still a composition effect.

**Clean Kvasir same-source value.** With the 31 shared images removed it is +0.9498 (p <0.0001, 95% CI [+0.864, +0.980]) over 32 methods, against +0.9523 with the overlap (CI [+0.871, +0.982], recomputed here from the old cache). The overlap inflated it by 0.0026, which is negligible. Kvasir has no empty masks, so this value is the same in every condition.

**Tiny lesions add nothing.** Also dropping lesions below 0.001 of the image moves no pairwise rho by more than 0.0121 beyond the empty-mask exclusion.

## Limitations

- The deep-only subset has 10 methods, so a single rho there has a wide null band (about |rho| >= 0.64 for p < 0.05).
- The bootstrap CI resamples methods, not images. It describes uncertainty from the choice of method pool, not from the choice of images.
- Lesion fraction is measured after resampling for Kvasir and at native size for the wound sets. The 0.001 cut is a share of pixels, not a clinical size.
- Which condition is "right" is a design choice. Empty masks are part of the published AZH and FUSeg test sets, and a method that correctly predicts nothing on them is doing its job. The point here is only that the FUSeg vs AZH Dice agreement leans on them.
