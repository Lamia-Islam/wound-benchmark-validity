# Practical consequences for choosing a model (V7, modern group M19)

Script: `scripts/v7/practical_v7.py`. Full numbers: `results/v7/practical_v7.json`. Benchmark numbers quoted in the headline come from `results/v7/bench_v7.json` (`scripts/v7/bench_v7.py`).

## Definitions

- Candidates. M19: 18 ImageNet-pretrained architectures trained on FUSeg with one shared recipe, plus nnU-Net (one training). An architecture's per-image Dice is the mean over its available training seeds (3 for 16 architectures, 2 for manet_r18, 1 for manet_r50). M11: the V6 set (10 run-1 seed-0 architectures + nnU-Net). C52: every trained architecture-seed model as its own candidate (51) plus nnU-Net.
- Mean Dice over all kept images, empty references included (Dice 1 if the prediction is also empty, else 0).
- Bootstrap: 2,000 draws over resampling units (photograph for Medetec and AZH, image otherwise), paired across candidates, `default_rng([20261010, dataset index])` as in V6. M11, M19 and C52 use the same draws. 95% percentile intervals.
- Indistinguishable from the top: the 95% paired interval of (top minus candidate) has lower bound 0 or less.
- Selection loss A to B: best mean Dice on B minus the mean Dice on B of the candidate ranked first on A (12 ordered wound pairs). Random choice: mean over candidates of (best on B minus candidate on B).
- Synthetic chooser: generator B, amplitude 40 (and 11), mean Dice over 45 images per synthetic seed, averaged over seeds 2024, 7, 101; architectures first averaged over training seeds per image.
- Miss: non-empty reference and Dice < t (t = 0.1; 0.05 and 0.2 as sensitivity). For an architecture, the miss rate and Dice when detected are computed per training seed and averaged. 'Pretrained mean' averages the pretrained architectures. Intervals: same paired unit bootstrap.
- Spearman: scipy, average ranks, across the candidates, between two wound datasets; mean over the 6 pairs.

## Headline: V6 next to V7

| quantity | V6 (M11) | V7 (M19) |
|---|---|---|
| Selection loss, choose on one wound set and deploy on another (mean of 12 pairs) | 0.040 | 0.047 |
| Random-choice loss (mean of 4 wound sets) | 0.046 (11 candidates) | 0.063 (19 candidates) |
| Synthetic chooser amp40: pick, mean loss over 4 wound sets | nnunet_2d, 0.013 | nnunet_2d, 0.030 |
| Synthetic chooser amp11: pick, mean loss over 4 wound sets | nnunet_2d, 0.013 | nnunet_2d, 0.030 |
| Medetec: top; others indistinguishable from it | deeplabv3p_r18_in; 4 of 10 | manet_r18@avg; 3 of 18 |
| CO2Wounds: top; others indistinguishable from it | segformer_mitb0_in; 3 of 10 | segformer_mitb2@avg; 4 of 18 |
| FUSeg: top; others indistinguishable from it | manet_r18_in; 4 of 10 | manet_r50@avg; 3 of 18 |
| AZH: top; others indistinguishable from it | nnunet_2d; 8 of 10 | segformer_mitb2@avg; 0 of 18 |
| control azh_train_slice: top; others indistinguishable from it | segformer_mitb0_in; 2 of 10 | segformer_mitb2@avg; 1 of 18 |
| AZH miss rate t = 0.1, pretrained mean (n non-empty images 140, 118 units) | 0.301 [0.228, 0.371] (10 archs) | 0.279 [0.212, 0.343] (18 archs) |
| AZH miss rate t = 0.1, nnU-Net | 0.014 [0.000, 0.035] | 0.014 [0.000, 0.035] |
| AZH miss rate difference, pretrained mean minus nnU-Net | 0.287 [0.213, 0.355] | 0.265 [0.198, 0.328] |
| Spearman of mean Dice across modern models, mean of 6 wound pairs | 0.126 (11 models) | 0.211 (19 models) |
| Benchmark relative agreement, amp 40, diff = bench - other, U (18 methods, unchanged), method bootstrap | +0.159 [-0.104, +0.314] | +0.159 [-0.104, +0.314] |
| Benchmark relative agreement, amp 40, diff = bench - other, M (11) -> M19 (19), method bootstrap | +0.227 [-0.186, +0.417] | -0.004 [-0.406, +0.299] |
| Benchmark relative agreement, amp 40, diff = bench - other, U+M (29) -> U+M19 (37), method bootstrap | -0.081 [-0.260, +0.034] | -0.174 [-0.381, -0.017] |

V6 had no intervals for miss rates; the V6 intervals above were computed here with the same draws and reproduce the V6 point estimates exactly.

## 1 and 2. Spread within M19

| set | images (units) | top | top mean [95% CI] | indistinguishable (of 18) | top wins draws | best minus median | best minus worst (worst) |
|---|---|---|---|---|---|---|---|
| Medetec | 152 (38) | manet_r18@avg | 0.771 [0.716, 0.818] | 3 | 0.521 | 0.060 | 0.203 (pspnet_r18@avg) |
| CO2Wounds | 150 (150) | segformer_mitb2@avg | 0.474 [0.420, 0.530] | 4 | 0.368 | 0.037 | 0.069 (manet_r50@avg) |
| FUSeg | 112 (112) | manet_r50@avg | 0.827 [0.788, 0.862] | 3 | 0.821 | 0.023 | 0.090 (pspnet_r18@avg) |
| AZH | 150 (118) | segformer_mitb2@avg | 0.713 [0.658, 0.767] | 0 | 0.995 | 0.133 | 0.205 (deeplabv3p_r18@avg) |
| control azh_train_slice | 103 (89) | segformer_mitb2@avg | 0.710 [0.649, 0.768] | 1 | 0.917 | 0.164 | 0.281 (deeplabv3p_r18@avg) |

Candidates indistinguishable from the top:

- Medetec: unet_r18@avg, deeplabv3p_r18@avg, nnunet_2d.
- CO2Wounds: unet_r18@avg, fpn_r18@avg, segformer_mitb0@avg, nnunet_2d.
- FUSeg: unetpp_r18@avg, manet_r18@avg, unet_r50@avg.
- AZH: none.
- control azh_train_slice: unet_swin_t@avg.

On AZH, segformer_mitb2 (0.713) is ahead of nnU-Net (0.631); the paired interval of the gap is [0.019, 0.146]. In V6 nnU-Net was top on AZH with 8 of 10 models indistinguishable from it.

Sensitivity, M17 (no MAnet, 17 candidates): indistinguishable from the top Medetec 6 of 16 (nnunet_2d), CO2Wounds 4 of 16 (segformer_mitb2@avg), FUSeg 8 of 16 (unetpp_r18@avg), AZH 0 of 16 (segformer_mitb2@avg), control azh_train_slice 1 of 16 (segformer_mitb2@avg).

## 3. Selection loss (rows: chosen on; columns: deployed on)

| chosen on (pick) | Medetec | CO2Wounds | FUSeg | AZH |
|---|---|---|---|---|
| Medetec (manet_r18@avg) | 0.000 | 0.032 | 0.007 | 0.108 |
| CO2Wounds (segformer_mitb2@avg) | 0.039 | 0.000 | 0.021 | 0.000 |
| FUSeg (manet_r50@avg) | 0.113 | 0.069 | 0.000 | 0.118 |
| AZH (segformer_mitb2@avg) | 0.039 | 0.000 | 0.021 | 0.000 |

- Off-diagonal (12 pairs): mean 0.047, range 0.000 to 0.118. V6: 0.040.
- Random choice, expected loss: Medetec 0.061, CO2Wounds 0.036, FUSeg 0.029, AZH 0.125 (mean 0.063).
- Control chooser (azh_train_slice, pick segformer_mitb2@avg): Medetec 0.039, CO2Wounds 0.000, FUSeg 0.021, AZH 0.000.
- Synthetic chooser amp40: picks nnunet_2d (synthetic Dice 0.790, runner-up 0.647); losses Medetec 0.007, CO2Wounds 0.000, FUSeg 0.033, AZH 0.082 (mean 0.030). Spearman of synthetic with wound means: Medetec +0.10, CO2Wounds +0.47, FUSeg +0.07, AZH +0.24.
- Synthetic chooser amp11: picks nnunet_2d (synthetic Dice 0.812, runner-up 0.655); losses Medetec 0.007, CO2Wounds 0.000, FUSeg 0.033, AZH 0.082 (mean 0.030). Spearman of synthetic with wound means: Medetec +0.20, CO2Wounds +0.58, FUSeg +0.06, AZH +0.38.
- Minimax: segformer_mitb2@avg has the smallest maximum gap from the best (0.039).
- M17 sensitivity: off-diagonal mean 0.037, random 0.059, synthetic pick (amp 40) nnunet_2d with mean loss 0.025.
- The FUSeg chooser picks manet_r50, which rests on a single training (its seeds 0 and 2 diverged).

## 4. Misses

t = 0.1. Pretrained mean with 95% interval and the range over the 18 architectures; nnU-Net with interval; group means for U (18), F (14), K (14).

| set | n non-empty | pretrained-18 mean miss [95%] (range) | nnU-Net miss [95%] | difference [95%] | U | F | K |
|---|---|---|---|---|---|---|---|
| Medetec | 152 | 0.024 [0.007, 0.046] (0.000 to 0.068) | 0.000 [0.000, 0.000] | 0.024 [0.007, 0.046] | 0.175 | 0.231 | 0.248 |
| CO2Wounds | 150 | 0.271 [0.212, 0.333] (0.233 to 0.302) | 0.260 [0.193, 0.333] | 0.011 [-0.052, 0.072] | 0.478 | 0.417 | 0.380 |
| FUSeg | 108 | 0.026 [0.007, 0.050] (0.009 to 0.040) | 0.046 [0.009, 0.091] | -0.020 [-0.045, -0.001] | 0.808 | 0.165 | 0.770 |
| AZH | 140 | 0.279 [0.212, 0.343] (0.119 to 0.391) | 0.014 [0.000, 0.035] | 0.265 [0.198, 0.328] | 0.193 | 0.377 | 0.597 |
| control azh_train_slice | 95 | 0.339 [0.262, 0.416] (0.137 to 0.509) | 0.042 [0.010, 0.085] | 0.297 [0.218, 0.380] | 0.249 | 0.501 | 0.698 |

AZH by threshold (pretrained mean vs nnU-Net, M19):

| t | pretrained mean [95%] | nnU-Net [95%] | difference [95%] | seed-averaged-Dice miss (sensitivity) |
|---|---|---|---|---|
| 0.05 | 0.268 [0.203, 0.332] | 0.007 [0.000, 0.023] | 0.261 [0.195, 0.323] | 0.225 |
| 0.1 | 0.279 [0.212, 0.343] | 0.014 [0.000, 0.035] | 0.265 [0.198, 0.328] | 0.244 |
| 0.2 | 0.299 [0.229, 0.365] | 0.043 [0.014, 0.080] | 0.256 [0.187, 0.321] | 0.273 |

Best pretrained architecture on AZH by miss rate: segformer_mitb2, 0.119 at t = 0.1 (nnU-Net 0.014).

Mean Spearman over the 6 wound pairs, 19 models:

| quantity | t = 0.05 | t = 0.1 | t = 0.2 |
|---|---|---|---|
| miss rate | +0.12 | +0.11 | +0.10 |
| Dice when detected | +0.18 | +0.18 | +0.24 |

Mean Dice: +0.21 (pairs: medetec|co2wounds +0.73, medetec|fuseg -0.12, medetec|azh +0.28, co2wounds|fuseg -0.13, co2wounds|azh +0.28, fuseg|azh +0.22); non-empty images only: +0.24.

## 5. Seed-aware analysis

### 5a. Every trained model as its own candidate (C52: 51 architecture-seed models + nnU-Net)

| set | top | indistinguishable (of 51) | of these, same architecture as top | distinct architectures among top + indistinguishable | top wins draws |
|---|---|---|---|---|---|
| Medetec | manet_r18@s1 | 5 | 0 | 5 | 0.771 |
| CO2Wounds | unet_r18@s1 | 3 | 0 | 4 | 0.855 |
| FUSeg | manet_r50@s1 | 14 | 0 | 9 | 0.343 |
| AZH | segformer_mitb2@s1 | 0 | 0 | 1 | 0.976 |
| control azh_train_slice | unet_swin_t@s1 | 4 | 0 | 3 | 0.648 |

Every C52 top is a single trained model. Its other training seeds (same architecture) are never indistinguishable from it: 0 of 7 sibling models across the 5 sets. Training seed of the top model per set: Medetec 1, CO2Wounds 1, FUSeg 1, AZH 1, control azh_train_slice 1. Seed 1 is not better on average (mean Dice of the 16 architectures with all 3 seeds, seeds 0/1/2: Medetec 0.711/0.695/0.710; CO2Wounds 0.441/0.438/0.435; FUSeg 0.793/0.796/0.797; AZH 0.588/0.587/0.576; control azh_train_slice 0.561/0.550/0.535).

Selection loss over C52: off-diagonal mean 0.076 (12 pairs), random 0.085. Synthetic chooser (amp 40) picks unet_effb3@s0, mean loss 0.087; amp 11 picks unet_effb3@s0, 0.087.

### 5b. One training per architecture: the result depends on which seed was trained

Each candidate set holds one model per architecture (all trained with seed s) plus nnU-Net.

| seed set (n) | indistinguishable from top: Medetec, CO2Wounds, FUSeg, AZH, control azh_train_slice | selection loss | random | synthetic pick (amp 40), loss |
|---|---|---|---|---|
| s0 (18) | 6, 4, 6, 1, 2 | 0.058 | 0.060 | unet_effb3@s0, 0.067 |
| s1 (19) | 2, 1, 2, 0, 1 | 0.076 | 0.084 | nnunet_2d, 0.051 |
| s2 (17) | 5, 11, 10, 1, 0 | 0.053 | 0.059 | nnunet_2d, 0.024 |

Top per seed set: s0: Medetec deeplabv3p_r18@s0, CO2Wounds segformer_mitb0@s0, FUSeg manet_r18@s0, AZH segformer_mitb2@s0; s1: Medetec manet_r18@s1, CO2Wounds unet_r18@s1, FUSeg manet_r50@s1, AZH segformer_mitb2@s1; s2: Medetec deeplabv3p_r18@s2, CO2Wounds nnunet_2d, FUSeg unetpp_r18@s2, AZH segformer_mitb2@s2.

### 5c. Are two seeds of one architecture more alike than two architectures?

Share of model pairs whose paired 95% interval of the mean-Dice difference excludes 0 (51 trained models; nnU-Net excluded).

| set | same architecture, distinguishable | different architectures, distinguishable | mean (max) seed range of an architecture | median bootstrap SD of one model's mean |
|---|---|---|---|---|
| Medetec | 28 of 49 (0.571) | 612 of 1226 (0.499) | 0.052 (0.115), 17 archs | 0.036 |
| CO2Wounds | 27 of 49 (0.551) | 590 of 1226 (0.481) | 0.041 (0.079), 17 archs | 0.027 |
| FUSeg | 9 of 49 (0.184) | 548 of 1226 (0.447) | 0.014 (0.049), 17 archs | 0.022 |
| AZH | 18 of 49 (0.367) | 690 of 1226 (0.563) | 0.045 (0.129), 17 archs | 0.036 |
| control azh_train_slice | 18 of 49 (0.367) | 815 of 1226 (0.665) | 0.052 (0.145), 17 archs | 0.042 |

### 5d. How much of the selection loss is seed-to-seed variation?

Chooser sees the seed-s models on A and picks an architecture; the deployed model is that architecture trained with seed s' and evaluated on B. Loss is measured against the best model of the seed-s' set on B. nnU-Net has one model, so it is the same model in every seed set.

| scenario | with nnU-Net: mean loss (cases, dropped) | without nnU-Net |
|---|---|---|
| Seed only: A = B, s' differs from s | 0.019 (20, 4) | 0.018 (20, 4) |
| Dataset only: A differs from B, s' = s | 0.063 (36, 0) | 0.061 (36, 0) |
| Both: A differs from B, s' differs from s | 0.061 (60, 12) | 0.058 (60, 12) |
| Chooser on seed 0, seed only (s' = 1, 2) | 0.023 (7, 1) | 0.022 (7, 1) |
| Chooser on seed 0, both | 0.066 (21, 3) | 0.065 (21, 3) |
| Chooser on seed 0, dataset only | 0.058 (12, 0) | 0.058 (12, 0) |
| Random choice (4 sets x 3 seeds) | 0.068 | 0.069 |
| Synthetic chooser amp40, same seed (picks by seed: unet_effb3, nnunet_2d, nnunet_2d) | 0.047 (12, 0) | 0.070 (12, 0) |
| Synthetic chooser amp40, other seed (without nnU-Net picks: unet_effb3, unet_effb3, segformer_mitb2) | 0.054 (24, 0) | 0.060 (24, 0) |
| Synthetic chooser amp11, same seed (picks by seed: unet_effb3, nnunet_2d, nnunet_2d) | 0.047 (12, 0) | 0.070 (12, 0) |
| Synthetic chooser amp11, other seed (without nnU-Net picks: unet_effb3, unet_effb3, segformer_mitb2) | 0.054 (24, 0) | 0.060 (24, 0) |

Dropped cases: the chosen architecture has no model for seed s' (MAnet). Seed-only loss by dataset: Medetec 0.030 (5 cases), CO2Wounds 0.030 (6 cases), FUSeg 0.014 (3 cases), AZH 0.000 (6 cases).
Top architecture per seed (with nnU-Net): Medetec deeplabv3p_r18/manet_r18/deeplabv3p_r18; CO2Wounds segformer_mitb0/unet_r18/nnunet_2d; FUSeg manet_r18/manet_r50/unetpp_r18; AZH segformer_mitb2/segformer_mitb2/segformer_mitb2. The same architecture tops both seeds in 4 of 12 dataset x seed-pair cases.

### 5e. Synthetic scores move a lot between training seeds

- Generator B amplitude 40: range of an architecture's synthetic mean Dice across its training seeds, mean 0.144, max 0.332 (unet_effb3: s0 0.798, s1 0.675, s2 0.467), 17 architectures with 2 or more seeds. nnU-Net scores 0.790.
- Amplitude 11: mean 0.161, max 0.444 (unet_effb3).
- For comparison, the mean seed range on the clinical wound sets is Medetec 0.052, CO2Wounds 0.041, FUSeg 0.014, AZH 0.045.

## 6. Diverged trainings (practical finding)

Under the shared training recipe, 3 of 54 planned trainings (18 architectures x 3 seeds) diverged: manet_r18 seed 2, manet_r50 seed 0, manet_r50 seed 2. 51 trainings were saved and scored. All three are MAnet; the log shows non-finite losses building up during training and non-finite weights at the end, so the checkpoints were not saved:

- manet_r18_in_s2: cumulative non-finite loss steps by epoch 20: 1, by epoch 30: 4, by epoch 40: 4.
- manet_r50_s0: cumulative non-finite loss steps by epoch 20: 2, by epoch 30: 134, by epoch 40: 367.
- manet_r50_s2: cumulative non-finite loss steps by epoch 10: 41, by epoch 20: 247, by epoch 30: 497, by epoch 40: 747.
- The 41 saved phase-2 trainings recorded no non-finite loss steps. The 10 run-1 trainings did not record this count.
- Surviving MAnet seeds: manet_r18 [0, 1], manet_r50 [1]. MAnet is the only architecture family that failed, and it failed at both encoder sizes; a practitioner reusing one recipe across architectures should expect such failures and check for them.

## Checks

- The copied V6 logic run on M11 reproduces `results/v6/practical_v6.json` exactly (per-set tables, selection, synthetic chooser, misses at all 3 thresholds, Spearman): max abs difference per_set 0e+00, selection 0e+00, miss_0.05 0e+00, miss_0.1 0e+00, miss_0.2 0e+00, spearman_mean_dice 0e+00, spearman_mean_dice_nonempty 0e+00.
- M19 per-image scores built from per-seed members equal `perimage_v7.load7` columns: max abs difference 0e+00.
- Synthetic chooser scores recomputed with the averaging order swapped (training seeds outside, synthetic seeds inside): max abs difference 2e-16.
- Miss-rate interval routine point estimates equal the table route (M19) and the V6 values (M11) at all thresholds and sets (asserted).
- One-training-world losses (dataset only, seed s' = s) equal the mean of the per-seed candidate sets' selection losses, and the synthetic same-seed loss equals the mean of their synthetic losses (asserted, within rounding).
- The three diverged trainings found in the log are exactly the architecture-seed models missing from the score files (asserted).

## Limitations

- nnU-Net has a single training, so it carries no seed averaging and contributes no seed-to-seed variation. In the seed-only scenario it is the chosen model in 2 of 24 cases (co2wounds s2); the 'without nnU-Net' column removes this.
- manet_r50 rests on one training and manet_r18 on two; the FUSeg chooser picks manet_r50, so the largest FUSeg-to-other losses depend on a single training. M17 removes both.
- Seed averaging (M19) makes each architecture's score less noisy than any single trained model. A practitioner deploys one trained model, so section 5 is the closer description of practice.
- Three seeds per architecture give a rough view of training noise; seed ranges and seed-only losses are point estimates without intervals.
- Each wound set has 112 to 152 images, Medetec only 38 units, so intervals are wide. Pairwise distinguishability counts 1,275 overlapping comparisons and is descriptive, not a multiplicity-corrected test.
- The AZH explanation of V6 (padding and scale) was not rerun; it rested on the 10 run-1 models.
