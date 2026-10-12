# SAM checks and phase-2 run checks (V7)

Script: `scripts/v7/sam_checks_v7.py`. Numbers: `results/v7/sam_checks_v7.json`. Exclusions as in perimage_v6 (8 FUSeg, 47 control images) unless a table says all images. Image order of every Kaggle file was checked against perimage32 (gt_fraction identical, 36 files).

## Definitions

- Dice and NSD (tau 2 px) per image, dataset mean over all kept images (empty references included).
- nnU-Net = run-1 scores (the column in M11 and M19). nnU-Net rescore P0 = phase-2 re-prediction of the same checkpoint; its masks are the ones the nnbox prompts were drawn from.
- M19 range = lowest and highest dataset mean among the 19 M19 methods (18 seed-averaged architectures + nnU-Net).
- Paired unit bootstrap as practical_v6: units = photograph for Medetec and AZH (and control), image otherwise; B = 2000; one weight matrix per dataset from default_rng([20261010, k]), k = dataset index in REFS + control, shared by all contrasts and both metrics. Interval = 2.5 and 97.5 percentiles. The bootstrap helper is copied from practical_v6 and the agreement code from stats_v6 (importing either runs a full analysis); the agreement copy reproduces stats_v6 group M exactly (max abs diff 0.0e+00 over 12 values).

## Summary

- DICE, official MedSAM preprocessing (oracle boxes) minus run-1 MedSAM: higher in 6 of 12 sets (medetec, co2wounds, azh, kvasir_disjoint, kvasir_sessile, azh_train_slice), lower in 3 (etis_larib, ph2, tn3k_test), differences -0.253 to +0.352. MedSAM fixed stays below run-1 SAM oracle in 12 of 12 sets.
- NSD, official MedSAM preprocessing (oracle boxes) minus run-1 MedSAM: higher in 6 of 12 sets (medetec, co2wounds, azh, kvasir_disjoint, kvasir_sessile, azh_train_slice), lower in 3 (fuseg, etis_larib, ph2), differences -0.148 to +0.153. MedSAM fixed stays below run-1 SAM oracle in 12 of 12 sets.
- DICE, sam_nnbox minus nnU-Net on the 4 wound sets + control: range -0.015 to +0.009; interval above 0 in none, below 0 in none.
- NSD, sam_nnbox minus nnU-Net on the 4 wound sets + control: range -0.046 to +0.117; interval above 0 in ['azh', 'azh_train_slice'], below 0 in ['medetec'].
- DICE, medsam_fixed_nnbox minus nnU-Net on the 4 wound sets + control: range -0.175 to -0.043; interval above 0 in none, below 0 in ['medetec', 'co2wounds', 'fuseg', 'azh'].
- NSD, medsam_fixed_nnbox minus nnU-Net on the 4 wound sets + control: range -0.358 to +0.010; interval above 0 in none, below 0 in ['medetec', 'co2wounds', 'fuseg', 'azh'].
- No nnU-Net box (prediction empty, so nnbox = nnU-Net exactly): at most 35 of 564 wound images and at most 581 of 690 colonoscopy images (analysis sets); exactly confirmed only 1 (empty references).
- DICE agreement, M19 (19) vs M19 + both nnbox (21): wound pairs 0.211 -> 0.332, modality_diff 0.225 -> 0.316 (+sam_nnbox only: 0.237, 0.278; +medsam_fixed_nnbox only: 0.314, 0.273).
- NSD agreement, M19 (19) vs M19 + both nnbox (21): wound pairs 0.473 -> 0.524, modality_diff 0.359 -> 0.428 (+sam_nnbox only: 0.449, 0.377; +medsam_fixed_nnbox only: 0.548, 0.420).
- Dice rank among M19 plus the variant (of 20) on medetec, co2wounds, fuseg, azh: sam_nnbox [3, 5, 11, 2], medsam_fixed_nnbox [19, 20, 20, 19]. A method that is last on every wound set adds concordant ranks to every wound pair, so the rise in wound-pair agreement from medsam_fixed_nnbox is mechanical, not evidence that the wound datasets agree more.
- nnU-Net rescore vs run 1: 172 of 1712 images differ in Dice beyond rounding (max 0.003382), 46 in NSD (max 0.005401); largest dataset-mean change 2.6e-05 Dice, 8.3e-05 NSD; no mean changes at the third decimal; no rank change; same pipeline in both runs, so GPU floating-point non-determinism, not a bug; no reported number uses the rescore nnU-Net values except the rescore sensitivity family and the nnbox prompts.
- Diverged trainings: only MAnet (manet_r18 seed 2, manet_r50 seeds 0 and 2); first non-finite losses in epochs 11 to 20, 11 to 20, 1 to 10; recipe identical to run 1 apart from the seed (AdamW 3e-4, wd 1e-4, cosine over 40 epochs, fp16 AMP with GradScaler, batch 16). No saved model has a threshold at the grid edge or a non-finite step; sweep Dice below 0.76 only for pspnet_r18_in (run 1), pspnet_r18_in_s1, pspnet_r18_in_s2.
- Consistency: pretrained_s0, kvasir_nets and kvasir_unet reproduce run 1 to the 5e-6 rounding floor (recomputed independently); only nnU-Net differs (see above).
- nnU-Net held-out FUSeg (n = 148): mean Dice 0.8328 at 0.5; the sweep threshold cannot be recovered (probability maps not kept). On the test sets themselves the best grid threshold would add at most +0.031 Dice (descriptive, optimistic).

## Part 1. SAM and MedSAM

### Coverage

- Phase-2 groups sam_v2_oracle and sam_v2_nnbox: 13 sets each, synthetic sets: 0 and 0. The 11 references and the control are all present; the only extra set is fuseg_r256 (not used here). Run-1 SAM has 33 sets (12 clinical + 21 synthetic). So the phase-2 SAM variants have no synthetic scores and cannot enter the benchmark-validity comparison.

### 1a. MedSAM preprocessing, code reading

`kaggle2/k2_sam.py` reproduces the official MedSAM_Inference.py steps: RGB image resized to 1024 x 1024 (cubic, anti-aliased, back to uint8), min-max scaled to [0, 1], no mean/std normalisation (the Hugging Face SamModel does not normalise pixel_values), boxes scaled to the 1024 frame, sigmoid of the 256 x 256 low-resolution logits bilinearly upsampled to the native size. Reading the code I found no remaining preprocessing difference, so the MedSAM fixed numbers below should be taken as MedSAM's real behaviour on these images. sam_nnbox uses SamProcessor exactly as run-1 SAM.

### 1a. Dataset means, DICE

| dataset | n img (units) | SAM r1 oracle | MedSAM r1 oracle | MedSAM fixed oracle | SAM nnbox | MedSAM fixed nnbox | nnU-Net | nnU-Net rescore | M19 mean | M19 range |
|---|---|---|---|---|---|---|---|---|---|---|
| medetec | 152 (38) | 0.885 | 0.376 | 0.728 | 0.762 | 0.589 | 0.764 | 0.764 | 0.709 | 0.568 to 0.771 |
| co2wounds | 150 (150) | 0.792 | 0.552 | 0.659 | 0.460 | 0.360 | 0.474 | 0.474 | 0.438 | 0.405 to 0.474 |
| fuseg | 112 (112) | 0.896 | 0.752 | 0.728 | 0.800 | 0.653 | 0.794 | 0.794 | 0.798 | 0.738 to 0.827 |
| azh | 150 (118) | 0.907 | 0.585 | 0.664 | 0.641 | 0.518 | 0.631 | 0.631 | 0.588 | 0.508 to 0.713 |
| kvasir_disjoint | 120 (120) | 0.932 | 0.792 | 0.839 | 0.041 | 0.042 | 0.044 | 0.044 | 0.063 | 0.013 to 0.106 |
| kvasir_sessile | 120 (120) | 0.885 | 0.719 | 0.752 | 0.013 | 0.014 | 0.010 | 0.010 | 0.022 | 0.001 to 0.045 |
| cvc_clinicdb | 150 (150) | 0.904 | 0.736 | 0.724 | 0.055 | 0.037 | 0.056 | 0.056 | 0.068 | 0.048 to 0.091 |
| cvc_colondb | 150 (150) | 0.886 | 0.675 | 0.686 | 0.003 | 0.009 | 0.003 | 0.003 | 0.007 | 0.001 to 0.014 |
| etis_larib | 150 (150) | 0.896 | 0.798 | 0.750 | 0.021 | 0.021 | 0.020 | 0.020 | 0.023 | 0.013 to 0.030 |
| ph2 | 150 (150) | 0.887 | 0.814 | 0.561 | 0.471 | 0.283 | 0.440 | 0.440 | 0.048 | 0.006 to 0.440 |
| tn3k_test | 150 (150) | 0.764 | 0.747 | 0.698 | 0.131 | 0.109 | 0.140 | 0.140 | 0.015 | 0.000 to 0.140 |
| azh_train_slice | 103 (89) | 0.887 | 0.558 | 0.660 | 0.549 | 0.505 | 0.548 | 0.548 | 0.549 | 0.429 to 0.710 |

Paired differences with 95% unit-bootstrap intervals (DICE). * = interval excludes 0.

| dataset | MedSAM fixed oracle - MedSAM r1 oracle | SAM nnbox - nnU-Net | MedSAM fixed nnbox - nnU-Net | SAM nnbox - nnU-Net rescore | MedSAM fixed nnbox - nnU-Net rescore |
|---|---|---|---|---|---|
| medetec | +0.3519 [+0.2930, +0.4083] * | -0.0019 [-0.0153, +0.0128] | -0.1749 [-0.2077, -0.1411] * | -0.0020 [-0.0153, +0.0128] | -0.1749 [-0.2077, -0.1411] * |
| co2wounds | +0.1069 [+0.0624, +0.1528] * | -0.0146 [-0.0335, +0.0013] | -0.1138 [-0.1419, -0.0869] * | -0.0146 [-0.0336, +0.0013] | -0.1138 [-0.1419, -0.0869] * |
| fuseg | -0.0239 [-0.0539, +0.0043] | +0.0057 [-0.0050, +0.0178] | -0.1413 [-0.1684, -0.1143] * | +0.0057 [-0.0050, +0.0178] | -0.1412 [-0.1684, -0.1143] * |
| azh | +0.0797 [+0.0404, +0.1179] * | +0.0091 [-0.0134, +0.0298] | -0.1134 [-0.1524, -0.0754] * | +0.0091 [-0.0134, +0.0298] | -0.1134 [-0.1524, -0.0755] * |
| kvasir_disjoint | +0.0472 [+0.0179, +0.0748] * | -0.0033 [-0.0112, +0.0026] | -0.0024 [-0.0136, +0.0109] | -0.0033 [-0.0112, +0.0026] | -0.0024 [-0.0136, +0.0109] |
| kvasir_sessile | +0.0328 [+0.0053, +0.0620] * | +0.0028 [-0.0021, +0.0119] | +0.0033 [-0.0076, +0.0171] | +0.0028 [-0.0021, +0.0119] | +0.0033 [-0.0076, +0.0171] |
| cvc_clinicdb | -0.0120 [-0.0479, +0.0239] | -0.0014 [-0.0072, +0.0048] | -0.0194 [-0.0332, -0.0071] * | -0.0014 [-0.0072, +0.0048] | -0.0194 [-0.0332, -0.0071] * |
| cvc_colondb | +0.0106 [-0.0243, +0.0411] | -0.0006 [-0.0020, +0.0007] | +0.0059 [-0.0018, +0.0170] | -0.0006 [-0.0020, +0.0007] | +0.0059 [-0.0018, +0.0170] |
| etis_larib | -0.0480 [-0.0735, -0.0237] * | +0.0010 [-0.0011, +0.0031] | +0.0008 [-0.0024, +0.0043] | +0.0010 [-0.0011, +0.0031] | +0.0008 [-0.0024, +0.0043] |
| ph2 | -0.2526 [-0.2969, -0.2109] * | +0.0309 [+0.0185, +0.0436] * | -0.1569 [-0.1866, -0.1289] * | +0.0309 [+0.0185, +0.0436] * | -0.1569 [-0.1866, -0.1289] * |
| tn3k_test | -0.0491 [-0.0757, -0.0214] * | -0.0085 [-0.0207, +0.0004] | -0.0312 [-0.0470, -0.0159] * | -0.0085 [-0.0206, +0.0004] | -0.0312 [-0.0470, -0.0159] * |
| azh_train_slice | +0.1019 [+0.0510, +0.1505] * | +0.0012 [-0.0260, +0.0277] | -0.0427 [-0.0925, +0.0047] | +0.0012 [-0.0260, +0.0277] | -0.0427 [-0.0925, +0.0047] |

### 1a. Dataset means, NSD

| dataset | n img (units) | SAM r1 oracle | MedSAM r1 oracle | MedSAM fixed oracle | SAM nnbox | MedSAM fixed nnbox | nnU-Net | nnU-Net rescore | M19 mean | M19 range |
|---|---|---|---|---|---|---|---|---|---|---|
| medetec | 152 (38) | 0.566 | 0.096 | 0.249 | 0.365 | 0.141 | 0.411 | 0.411 | 0.403 | 0.291 to 0.500 |
| co2wounds | 150 (150) | 0.548 | 0.292 | 0.357 | 0.260 | 0.139 | 0.257 | 0.257 | 0.252 | 0.220 to 0.288 |
| fuseg | 112 (112) | 0.808 | 0.446 | 0.374 | 0.683 | 0.307 | 0.666 | 0.666 | 0.578 | 0.390 to 0.666 |
| azh | 150 (118) | 0.879 | 0.303 | 0.423 | 0.516 | 0.258 | 0.423 | 0.423 | 0.494 | 0.378 to 0.620 |
| kvasir_disjoint | 120 (120) | 0.709 | 0.388 | 0.478 | 0.023 | 0.020 | 0.017 | 0.017 | 0.032 | 0.004 to 0.059 |
| kvasir_sessile | 120 (120) | 0.653 | 0.311 | 0.362 | 0.011 | 0.007 | 0.008 | 0.008 | 0.014 | 0.001 to 0.034 |
| cvc_clinicdb | 150 (150) | 0.764 | 0.357 | 0.366 | 0.036 | 0.013 | 0.032 | 0.032 | 0.037 | 0.027 to 0.052 |
| cvc_colondb | 150 (150) | 0.684 | 0.298 | 0.323 | 0.001 | 0.003 | 0.001 | 0.001 | 0.007 | 0.001 to 0.015 |
| etis_larib | 150 (150) | 0.832 | 0.585 | 0.495 | 0.007 | 0.005 | 0.006 | 0.006 | 0.019 | 0.006 to 0.031 |
| ph2 | 150 (150) | 0.319 | 0.295 | 0.147 | 0.088 | 0.034 | 0.100 | 0.100 | 0.011 | 0.001 to 0.100 |
| tn3k_test | 150 (150) | 0.415 | 0.314 | 0.275 | 0.071 | 0.036 | 0.072 | 0.072 | 0.007 | 0.000 to 0.072 |
| azh_train_slice | 103 (89) | 0.876 | 0.336 | 0.479 | 0.436 | 0.329 | 0.319 | 0.319 | 0.470 | 0.319 to 0.630 |

Paired differences with 95% unit-bootstrap intervals (NSD). * = interval excludes 0.

| dataset | MedSAM fixed oracle - MedSAM r1 oracle | SAM nnbox - nnU-Net | MedSAM fixed nnbox - nnU-Net | SAM nnbox - nnU-Net rescore | MedSAM fixed nnbox - nnU-Net rescore |
|---|---|---|---|---|---|
| medetec | +0.1528 [+0.1155, +0.1867] * | -0.0458 [-0.0866, -0.0043] * | -0.2698 [-0.3349, -0.2073] * | -0.0458 [-0.0866, -0.0043] * | -0.2699 [-0.3350, -0.2073] * |
| co2wounds | +0.0651 [+0.0358, +0.0963] * | +0.0032 [-0.0195, +0.0265] | -0.1182 [-0.1489, -0.0874] * | +0.0033 [-0.0194, +0.0266] | -0.1181 [-0.1487, -0.0874] * |
| fuseg | -0.0714 [-0.1182, -0.0239] * | +0.0176 [-0.0082, +0.0450] | -0.3583 [-0.4134, -0.3031] * | +0.0176 [-0.0082, +0.0450] | -0.3584 [-0.4134, -0.3031] * |
| azh | +0.1200 [+0.0788, +0.1601] * | +0.0923 [+0.0615, +0.1244] * | -0.1655 [-0.2243, -0.1044] * | +0.0922 [+0.0614, +0.1243] * | -0.1656 [-0.2243, -0.1044] * |
| kvasir_disjoint | +0.0902 [+0.0454, +0.1324] * | +0.0052 [-0.0034, +0.0151] | +0.0028 [-0.0062, +0.0144] | +0.0052 [-0.0034, +0.0151] | +0.0028 [-0.0062, +0.0144] |
| kvasir_sessile | +0.0508 [+0.0109, +0.0906] * | +0.0028 [-0.0032, +0.0124] | -0.0014 [-0.0106, +0.0096] | +0.0028 [-0.0032, +0.0124] | -0.0014 [-0.0106, +0.0096] |
| cvc_clinicdb | +0.0094 [-0.0360, +0.0544] | +0.0039 [-0.0043, +0.0136] | -0.0184 [-0.0279, -0.0097] * | +0.0039 [-0.0043, +0.0136] | -0.0184 [-0.0279, -0.0097] * |
| cvc_colondb | +0.0252 [-0.0118, +0.0606] | -0.0005 [-0.0011, +0.0000] | +0.0017 [-0.0009, +0.0048] | -0.0005 [-0.0011, +0.0000] | +0.0017 [-0.0009, +0.0048] |
| etis_larib | -0.0902 [-0.1364, -0.0462] * | +0.0006 [-0.0023, +0.0033] | -0.0012 [-0.0034, +0.0006] | +0.0006 [-0.0023, +0.0033] | -0.0012 [-0.0034, +0.0006] |
| ph2 | -0.1484 [-0.1829, -0.1157] * | -0.0118 [-0.0292, +0.0030] | -0.0663 [-0.0944, -0.0406] * | -0.0118 [-0.0292, +0.0030] | -0.0663 [-0.0944, -0.0406] * |
| tn3k_test | -0.0381 [-0.0786, +0.0004] | -0.0011 [-0.0118, +0.0090] | -0.0362 [-0.0544, -0.0199] * | -0.0011 [-0.0118, +0.0090] | -0.0362 [-0.0544, -0.0199] * |
| azh_train_slice | +0.1425 [+0.0867, +0.1948] * | +0.1174 [+0.0712, +0.1660] * | +0.0101 [-0.0598, +0.0819] | +0.1174 [+0.0712, +0.1659] * | +0.0100 [-0.0597, +0.0818] |

### 1b. Images with no nnU-Net box

Rule in `kaggle2/k2_sam.py` (predict_nn, mask_boxes, medsam_prob, sam_proc_prob): the prompts are the bounding boxes of the up to 10 largest 8-connected components of nnU-Net's P0 mask. If the mask has no component there is no box, both functions return an all-zero probability map, and the prediction is empty. It is scored with the study convention: 1 if the reference is also empty, 0 otherwise. So on these images both nnbox variants score exactly what nnU-Net scores; SAM cannot recover a wound nnU-Net missed.

The nnU-Net masks are not in the local results, so the count is inferred from the rescore file. It is exact for empty references (Dice 1 means the prediction is empty). For non-empty references, an image with Dice 0 and NSD 0 at every tolerance (1, 2, 3, 5 px and 1% of the diagonal) has either an empty mask or a mask farther from the reference than every tolerance, so the count is an upper bound. Flagged images where an nnbox variant scores above 0 are provably non-empty and are removed.

| dataset | n img | empty refs | refs empty, nnU-Net empty (exact) | refs non-empty, Dice 0 | of which all NSD 0 | of those, nnbox > 0 (so not empty) | no-box images, lower to upper bound | upper bound share |
|---|---|---|---|---|---|---|---|---|
| medetec | 152 | 0 | 0 | 0 | 0 | 0 | 0 to 0 | 0.000 |
| co2wounds | 150 | 0 | 0 | 31 | 29 | 0 | 0 to 29 | 0.193 |
| fuseg | 112 | 4 | 1 | 5 | 5 | 0 | 1 to 6 | 0.054 |
| azh | 150 | 10 | 0 | 0 | 0 | 0 | 0 to 0 | 0.000 |
| kvasir_disjoint | 120 | 0 | 0 | 91 | 87 | 0 | 0 to 87 | 0.725 |
| kvasir_sessile | 120 | 0 | 0 | 111 | 108 | 1 | 0 to 107 | 0.892 |
| cvc_clinicdb | 150 | 0 | 0 | 112 | 110 | 0 | 0 to 110 | 0.733 |
| cvc_colondb | 150 | 0 | 0 | 143 | 143 | 0 | 0 to 143 | 0.953 |
| etis_larib | 150 | 0 | 0 | 137 | 134 | 0 | 0 to 134 | 0.893 |
| ph2 | 150 | 0 | 0 | 15 | 14 | 0 | 0 to 14 | 0.093 |
| tn3k_test | 150 | 0 | 0 | 79 | 75 | 1 | 0 to 74 | 0.493 |
| azh_train_slice | 103 | 8 | 0 | 1 | 1 | 1 | 0 to 0 | 0.000 |

### 1c. Where the non-oracle variants land, and agreement with them added

Placement, DICE (group mean and range of dataset means; rank of the variant among M19 plus itself, 1 = best of 20).

| dataset | U (18) | F (14) | K (14) | M19 (19) | SAM nnbox (rank) | MedSAM fixed nnbox (rank) |
|---|---|---|---|---|---|---|
| medetec | 0.291 (0.049 to 0.442) | 0.356 (0.133 to 0.569) | 0.287 (0.007 to 0.356) | 0.709 (0.568 to 0.771) | 0.762 (3) | 0.589 (19) |
| co2wounds | 0.188 (0.077 to 0.329) | 0.277 (0.164 to 0.420) | 0.272 (0.081 to 0.361) | 0.438 (0.405 to 0.474) | 0.460 (5) | 0.360 (20) |
| fuseg | 0.061 (0.003 to 0.198) | 0.531 (0.279 to 0.616) | 0.076 (0.048 to 0.100) | 0.798 (0.738 to 0.827) | 0.800 (11) | 0.653 (20) |
| azh | 0.297 (0.108 to 0.492) | 0.420 (0.319 to 0.573) | 0.197 (0.119 to 0.300) | 0.588 (0.508 to 0.713) | 0.641 (2) | 0.518 (19) |
| kvasir_disjoint | 0.229 (0.076 to 0.381) | 0.127 (0.008 to 0.358) | 0.638 (0.426 to 0.750) | 0.063 (0.013 to 0.106) | 0.041 (17) | 0.042 (17) |
| kvasir_sessile | 0.111 (0.031 to 0.194) | 0.053 (0.002 to 0.175) | 0.352 (0.222 to 0.458) | 0.022 (0.001 to 0.045) | 0.013 (15) | 0.014 (15) |
| cvc_clinicdb | 0.179 (0.015 to 0.347) | 0.089 (0.009 to 0.270) | 0.410 (0.174 to 0.550) | 0.068 (0.048 to 0.091) | 0.055 (17) | 0.037 (20) |
| cvc_colondb | 0.125 (0.004 to 0.238) | 0.051 (0.001 to 0.175) | 0.294 (0.107 to 0.404) | 0.007 (0.001 to 0.014) | 0.003 (17) | 0.009 (6) |
| etis_larib | 0.067 (0.013 to 0.144) | 0.015 (0.001 to 0.072) | 0.211 (0.093 to 0.293) | 0.023 (0.013 to 0.030) | 0.021 (12) | 0.021 (12) |
| ph2 | 0.490 (0.016 to 0.829) | 0.081 (0.000 to 0.467) | 0.126 (0.016 to 0.419) | 0.048 (0.006 to 0.440) | 0.471 (1) | 0.283 (2) |
| tn3k_test | 0.102 (0.000 to 0.201) | 0.007 (0.000 to 0.070) | 0.072 (0.000 to 0.189) | 0.015 (0.000 to 0.140) | 0.131 (2) | 0.109 (2) |
| azh_train_slice | 0.244 (0.088 to 0.394) | 0.328 (0.207 to 0.531) | 0.161 (0.102 to 0.250) | 0.549 (0.429 to 0.710) | 0.549 (8) | 0.505 (15) |

Placement, NSD (group mean and range of dataset means; rank of the variant among M19 plus itself, 1 = best of 20).

| dataset | U (18) | F (14) | K (14) | M19 (19) | SAM nnbox (rank) | MedSAM fixed nnbox (rank) |
|---|---|---|---|---|---|---|
| medetec | 0.127 (0.017 to 0.236) | 0.149 (0.048 to 0.234) | 0.099 (0.004 to 0.172) | 0.403 (0.291 to 0.500) | 0.365 (16) | 0.141 (20) |
| co2wounds | 0.090 (0.030 to 0.157) | 0.141 (0.068 to 0.226) | 0.083 (0.033 to 0.149) | 0.252 (0.220 to 0.288) | 0.260 (7) | 0.139 (20) |
| fuseg | 0.039 (0.001 to 0.129) | 0.255 (0.145 to 0.406) | 0.022 (0.010 to 0.048) | 0.578 (0.390 to 0.666) | 0.683 (1) | 0.307 (20) |
| azh | 0.155 (0.017 to 0.416) | 0.297 (0.193 to 0.423) | 0.108 (0.083 to 0.156) | 0.494 (0.378 to 0.620) | 0.516 (7) | 0.258 (20) |
| kvasir_disjoint | 0.098 (0.010 to 0.181) | 0.044 (0.004 to 0.104) | 0.276 (0.134 to 0.426) | 0.032 (0.004 to 0.059) | 0.023 (12) | 0.020 (15) |
| kvasir_sessile | 0.046 (0.002 to 0.090) | 0.025 (0.002 to 0.075) | 0.139 (0.059 to 0.246) | 0.014 (0.001 to 0.034) | 0.011 (12) | 0.007 (17) |
| cvc_clinicdb | 0.092 (0.031 to 0.189) | 0.042 (0.006 to 0.139) | 0.163 (0.049 to 0.253) | 0.037 (0.027 to 0.052) | 0.036 (12) | 0.013 (20) |
| cvc_colondb | 0.057 (0.017 to 0.097) | 0.033 (0.001 to 0.136) | 0.114 (0.033 to 0.171) | 0.007 (0.001 to 0.015) | 0.001 (20) | 0.003 (16) |
| etis_larib | 0.042 (0.003 to 0.088) | 0.020 (0.001 to 0.092) | 0.101 (0.037 to 0.174) | 0.019 (0.006 to 0.031) | 0.007 (19) | 0.005 (20) |
| ph2 | 0.164 (0.024 to 0.412) | 0.033 (0.000 to 0.148) | 0.034 (0.003 to 0.129) | 0.011 (0.001 to 0.100) | 0.088 (2) | 0.034 (2) |
| tn3k_test | 0.045 (0.000 to 0.101) | 0.002 (0.000 to 0.018) | 0.030 (0.000 to 0.068) | 0.007 (0.000 to 0.072) | 0.071 (2) | 0.036 (2) |
| azh_train_slice | 0.132 (0.011 to 0.331) | 0.240 (0.147 to 0.343) | 0.102 (0.072 to 0.170) | 0.470 (0.319 to 0.630) | 0.436 (14) | 0.329 (19) |

Agreement, DICE (sensitivity only; the paper never pools oracle SAM, and these nnbox variants are not in any primary group).

| method set | n | wound pairs (6) | wound different-source pairs (5) | fuseg~azh | colonoscopy pairs (10) | modality_diff | modality_diff_excl_within_source | same dataset (azh~control) |
|---|---|---|---|---|---|---|---|---|
| M11 (V6 M, check) | 11 | 0.126 | 0.116 | 0.173 | 0.722 | 0.461 | 0.443 | 0.691 |
| M19 | 19 | 0.211 | 0.210 | 0.217 | 0.403 | 0.225 | 0.174 | 0.910 |
| M19 + sam_nnbox | 20 | 0.237 | 0.244 | 0.202 | 0.425 | 0.278 | 0.230 | 0.883 |
| M19 + medsam_fixed_nnbox | 20 | 0.314 | 0.312 | 0.319 | 0.385 | 0.273 | 0.229 | 0.901 |
| M19 + both nnbox (21) | 21 | 0.332 | 0.338 | 0.301 | 0.405 | 0.316 | 0.275 | 0.879 |
| M19 with nnU-Net replaced by rescore P0 | 19 | 0.211 | 0.210 | 0.217 | 0.403 | 0.225 | 0.174 | 0.910 |

Agreement, NSD (sensitivity only; the paper never pools oracle SAM, and these nnbox variants are not in any primary group).

| method set | n | wound pairs (6) | wound different-source pairs (5) | fuseg~azh | colonoscopy pairs (10) | modality_diff | modality_diff_excl_within_source | same dataset (azh~control) |
|---|---|---|---|---|---|---|---|---|
| M11 (V6 M, check) | 11 | 0.308 | 0.262 | 0.536 | 0.719 | 0.444 | 0.399 | 0.909 |
| M19 | 19 | 0.473 | 0.490 | 0.390 | 0.577 | 0.359 | 0.334 | 0.872 |
| M19 + sam_nnbox | 20 | 0.449 | 0.454 | 0.423 | 0.573 | 0.377 | 0.348 | 0.833 |
| M19 + medsam_fixed_nnbox | 20 | 0.548 | 0.563 | 0.477 | 0.600 | 0.420 | 0.396 | 0.886 |
| M19 + both nnbox (21) | 21 | 0.524 | 0.529 | 0.501 | 0.593 | 0.428 | 0.401 | 0.852 |
| M19 with nnU-Net replaced by rescore P0 | 19 | 0.473 | 0.490 | 0.390 | 0.577 | 0.359 | 0.334 | 0.872 |

## Part 2. Run checks

### 2a. nnU-Net rescore vs run 1

Code: both runs call `nnUNetv2_predict ... --save_probabilities` with the same command, read channel 1 of the saved probabilities, resize if needed, and apply `postprocess_prob(p, 0.5)`. Run 1 did not use nnU-Net's own segmentation export; the pipeline is the same in both runs (kit_run.py byte-identical in both kits: True; checks of the code text: True, True). The difference is the re-run of GPU inference itself: nnU-Net 2.8.1 sets `torch.backends.cudnn.benchmark = True`, runs the network under fp16 autocast and accumulates the sliding-window logits in a half-precision buffer. cuDNN may pick different kernels in a new session (or on a different GPU), the fp16 sums then differ in the last bits, and pixels whose smoothed probability sits at 0.5 flip. This is floating-point non-determinism, not a bug.

- Image pairs compared: 1712 (12 sets, all images). Dice differs beyond the 5e-6 rounding floor on 172, NSD on 46, either on 175.
- Absolute Dice differences: {'<=5e-6 (rounding)': 1540, '5e-6..1e-4': 65, '1e-4..1e-3': 105, '>1e-3': 2}. Non-zero |dDice| quantiles 50/90/99/100%: [0.000131, 0.000429, 0.000887, 0.003382]; sign +/-: [97, 75].
- Non-zero |dNSD| quantiles: [0.00102, 0.004018, 0.005176, 0.005401]; sign +/-: [21, 25]. Max |dDice| 0.003382, max |dNSD| 0.005401.
- Largest image differences: co2wounds dDice 0.003382, dNSD -0.001583; tn3k_test dDice -0.001161, dNSD -2e-06; tn3k_test dDice -0.000775, dNSD 0.000686; azh dDice 0.000612, dNSD 0.004401; cvc_colondb dDice 0.000591, dNSD 0.0.
- Dataset means (analysis set): largest |change| Dice 2.6e-05, NSD 8.3e-05. Means that change at the third decimal: none.
- nnU-Net's rank within M19 changes with the rescore values in: no dataset. Agreement with nnU-Net replaced by the rescore values is in the 1c agreement table (last row).
- Rescore internal check: grid Dice at 0.5 equals P0 Dice in every set: True.

| dataset | n differ (Dice/NSD) | max abs dDice | max abs dNSD | mean Dice run 1 | mean Dice rescore | mean NSD run 1 | mean NSD rescore |
|---|---|---|---|---|---|---|---|
| medetec | 27/14 of 152 | 0.000423 | 0.00417 | 0.7639 | 0.7639 | 0.4106 | 0.4106 |
| co2wounds | 29/10 of 150 | 0.003382 | 0.004901 | 0.4742 | 0.4742 | 0.2566 | 0.2565 |
| fuseg | 8/2 of 120 | 0.0004 | 0.001576 | 0.7943 | 0.7943 | 0.6658 | 0.6659 |
| azh | 23/8 of 150 | 0.000612 | 0.004401 | 0.6315 | 0.6315 | 0.4234 | 0.4235 |
| kvasir_disjoint | 9/2 of 120 | 0.000524 | 0.00021 | 0.0444 | 0.0444 | 0.0174 | 0.0174 |
| kvasir_sessile | 1/0 of 120 | 0.000486 | 5e-06 | 0.0102 | 0.0102 | 0.0083 | 0.0083 |
| cvc_clinicdb | 6/0 of 150 | 0.000312 | 5e-06 | 0.0564 | 0.0564 | 0.0319 | 0.0319 |
| cvc_colondb | 1/0 of 150 | 0.000591 | 3e-06 | 0.0033 | 0.0033 | 0.0014 | 0.0014 |
| etis_larib | 1/0 of 150 | 4.6e-05 | 4e-06 | 0.0198 | 0.0198 | 0.0061 | 0.0061 |
| ph2 | 23/1 of 150 | 0.000241 | 0.000734 | 0.4399 | 0.4399 | 0.1003 | 0.1003 |
| tn3k_test | 11/4 of 150 | 0.001161 | 0.000686 | 0.1399 | 0.1398 | 0.0721 | 0.0721 |
| azh_train_slice | 33/5 of 150 | 0.00044 | 0.005401 | 0.5476 | 0.5476 | 0.3187 | 0.3188 |

Verdict: not a bug. Every reported V6/V7 number uses the run-1 nnU-Net scores (perimage_v6 reads results/kaggle/scores/nnunet), so none is affected. The rescore family (post-processing and tolerance sensitivity) and the nnbox prompts use the re-predicted masks; that is a different but equally valid draw of the same model, and the gap is far below any reported precision.

### 2b. The three diverged trainings

Recipe (k2_train.train_model): AdamW lr 3e-4, weight_decay 1e-4; CosineAnnealingLR(T_max = 40 epochs) stepped per epoch; torch.autocast(cuda) (fp16) for the forward pass, loss on .float() logits, torch.amp.GradScaler; batch 16, 40 epochs, 392 FUSeg images, 25 steps per epoch (1000 steps). Logged protocol: [17:17:36]   protocol: {'epochs': 40, 'weights': 'imagenet'}, batch 16, seeds [1, 2], versions {'torch': '2.11.0+cu128', 'smp': '0.5.0', 'timm': '1.0.22'}.

| model | first non-finite loss in epochs | cumulative non-finite steps at epochs 10/20/30/40 | epochs whose every step was non-finite | failure |
|---|---|---|---|---|
| manet_r18_in_s2 | 11 to 20 | 0/1/4/4 | none | non-finite weights after training (36 tensors, e.g. ['decoder.blocks.0.hl_conv.0.1.running_mean', 'decoder.blocks.0.hl_conv.0.1.running_var', 'decoder.blocks.0.hl_conv.1.1.running_mean']); not saved |
| manet_r50_s0 | 11 to 20 | 0/2/134/367 | [40] | non-finite weights after training (397 tensors, e.g. ['encoder.conv1.weight', 'encoder.bn1.weight', 'encoder.bn1.bias']); not saved |
| manet_r50_s2 | 1 to 10 | 41/247/497/747 | [20, 30, 40] | non-finite weights after training (397 tensors, e.g. ['encoder.conv1.weight', 'encoder.bn1.weight', 'encoder.bn1.bias']); not saved |
| manet_r18_in_s1 | none | 0/0/0/0 | none | trained and saved |
| manet_r50_s1 | none | 0/0/0/0 | none | trained and saved |

The log reports cumulative counts every 10 epochs, so the first bad step is known only to a 10-epoch window. manet_r18 seed 2 had only 4 non-finite steps of 1000 and a finite mean loss to the end, yet 36 tensors were non-finite after training; the three the log names are decoder BatchNorm running statistics. That pattern fits AMP: GradScaler skips the optimizer step when gradients are non-finite, so the weights survive, but the train-mode forward pass has already written the non-finite batch statistics into running_mean and running_var. manet_r50 seeds 0 and 2 degraded progressively (367 and 747 non-finite steps of 1000; seed 2 non-finite on every step from epoch 20 at the latest) and ended with 397 non-finite tensors including encoder weights. fp16 overflow under AMP in MAnet is the most likely mechanism; this is an inference from the pattern, the log does not show which layer overflowed. Only MAnet diverged (2 of 3 seeds of manet_r50, 1 of 3 of manet_r18), so the problem is architecture-specific, not a broken recipe.

Diff of the training loop, run 1 `kit_run.stage_pretrained` vs phase 2 `k2_train.train_model` (normalised lines):

```
--- kit_run.stage_pretrained
+++ k2_train.train_model
@@ -3 +3 @@
-rng = np.random.default_rng(0)
+rng = np.random.default_rng(seed)
@@ -8,2 +8,3 @@
-for i in range(0, len(order), 16):
-idx = order[i:i + 16]
+ep_loss = []
+for i in range(0, len(order), BATCH):
+idx = order[i:i + BATCH]
@@ -24,0 +26,5 @@
+lv = float(loss.detach())
+if np.isfinite(lv):
+ep_loss.append(lv)
+else:
+nonfinite += 1
```
Model construction (code checks): run 1 {'torch.manual_seed(0) before build': True, 'build: getattr(smp, arch)(enc, encoder_weights=weights, in_channels=3, classes=1)': True}; phase 2 {'torch.manual_seed(seed) before build': True, 'build: getattr(smp, arch)(enc, encoder_weights=weights, in_channels=3, classes=1, **kw)': True, 'kw empty for the ten original architectures (stage_seeds jobs pass {})': True}. The only training changes are the seed (torch.manual_seed and default_rng) and bookkeeping lines that record the loss; recording calls float(loss) each step, which synchronises but does not change the computation. The recipe is the same as run 1.

Train report (41 phase-2 models + 10 run-1 models): thresholds at the grid edge (0.05 or 0.9): none. Saved models with non-finite steps: none. Sweep Dice range 0.7477 (pspnet_r18_in (run 1)) to 0.8434 (manet_r18_in (run 1)); below 0.76: pspnet_r18_in (run 1), pspnet_r18_in_s1, pspnet_r18_in_s2 (PSPNet in all three seeds, the same level as run 1, so a property of the architecture, not a failure). Thresholds range 0.15 (pspnet_r18_in (run 1)) to 0.65 (linknet_r18_in (run 1)). Final-epoch training loss is 0.0713 to 0.1968 for every phase-2 model except LinkNet (linknet_r18_in_s1 0.3064, linknet_r18_in_s2 0.4898); LinkNet still reaches normal sweep Dice, so its high loss is a calibration oddity worth a footnote, not a failure.

### 2c. Consistency file

| family | comparisons | max abs dDice | max abs dNSD2 | reproduces run 1 to 5e-6 | independent recomputation matches file |
|---|---|---|---|---|---|
| pretrained_s0 | 17120 | 5.00e-06 | 5.00e-06 | True | True |
| kvasir_nets | 17120 | 5.00e-06 | 5.00e-06 | True | True |
| kvasir_unet | 1712 | 5.00e-06 | 5.00e-06 | True | True |
| nnunet | 1712 | 3.38e-03 | 5.40e-03 | False | True |

The rescore values are rounded to 5 decimals, so 5e-6 is the rounding floor: the seed-0 pretrained models, the ten Kvasir networks and the Kvasir U-Net reproduce run 1 exactly. Only nnU-Net does not (2a).

### 2d. nnU-Net held-out FUSeg

`nnunet_heldout.json` holds keys ['dice', 'mean_dice', 'n', 'name', 'threshold', 'variant']: nnU-Net P0 Dice at threshold 0.5 on the FUSeg training images 400 to 549 without duplicates (n = 148), the same images the other models used to pick their threshold. Mean Dice 0.8328 (recomputed 0.8328), median 0.9133, 5 images with Dice 0, 11 below 0.5.

The threshold the other models' rule would pick cannot be computed: nnunet_heldout.json stores only the per-image P0 Dice at 0.5; k2_rescore.rescore_nnunet deletes the probability maps (rescore_pred_out) after a successful run, so the sweep rule cannot be applied afterwards. For comparison, the other 51 models' best sweep Dice on the same images ranges 0.7477 to 0.8434, and 6 of them exceed nnU-Net's 0.8328. Their values are optimistic (threshold chosen on these images), nnU-Net's is not.

Descriptive bound only (threshold chosen on the test set itself, so optimistic and NOT the study rule): best grid threshold of nnU-Net per wound set from the rescore grid.

| dataset | n | best grid threshold | Dice at best | Dice at 0.5 | gain |
|---|---|---|---|---|---|
| medetec | 152 | 0.2 | 0.781 | 0.764 | +0.0170 |
| co2wounds | 150 | 0.1 | 0.499 | 0.474 | +0.0247 |
| fuseg | 112 | 0.3 | 0.809 | 0.794 | +0.0152 |
| azh | 150 | 0.7 | 0.644 | 0.631 | +0.0128 |
| azh_train_slice | 103 | 0.85 | 0.579 | 0.548 | +0.0314 |

## Limitations of these checks

- 1b is inferred from scores, because the nnU-Net masks were not brought back from Kaggle: exact only for empty references; for non-empty references it is an upper bound (an empty mask and a mask that misses the reference by more than every tolerance score the same).
- The nnbox prompts come from the phase-2 re-prediction of nnU-Net, not from the run-1 masks behind the nnU-Net column; the two differ slightly on 175 of 1712 images (2a), which is negligible for the nnbox comparisons (the two nnU-Net columns give the same intervals to 4 decimals in most cells).
- The cause of the nnU-Net differences (cuDNN autotuning with fp16 accumulation, possibly a different GPU) is the most likely explanation from the code; the GPU model of each session is not logged, so it cannot be confirmed.
- The divergence mechanism (fp16 overflow in MAnet) is inferred from the logged pattern; the log does not name the overflowing layer, and the per-step losses of the failed runs were not saved.
- 2d cannot apply the study's threshold rule to nnU-Net, because the held-out probability maps were deleted after the run. The test-set grid table is descriptive and optimistic by construction.
- Adding the nnbox variants to M19 (1c) is a sensitivity only: they are nnU-Net-derived (and identical to nnU-Net on every no-box image), and MedSAM fixed nnbox is last or second to last on every wound set, which raises wound-pair rank agreement mechanically.

