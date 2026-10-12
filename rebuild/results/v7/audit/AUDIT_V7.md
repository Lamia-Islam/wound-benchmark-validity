# Audit of the GPU results (run 1 and run 2) before V7 analysis

Date: 2026-10-11. Script: `scripts/v7/audit_v7.py` (reads the JSON files directly; it imports `perimage_v7` only in check 8, to compare). Every number below is in `results/v7/audit/audit_v7.json` under the key named in brackets. No existing file was changed.

## Verdict in one paragraph

The run-2 data are complete and internally consistent. Every score file has the right images, the right models, finite values in [0, 1], and reference masks that match perimage32 bit for bit. The rescore files reproduce the plain score files to the 5-decimal rounding they are stored with, for every family except nnU-Net. The nnU-Net differences are small (largest 0.0034 Dice on one image, at most 0.000026 on a dataset mean), random in sign, and change no ranking; they come from re-running nnU-Net inference, not from a code difference. One real problem sits outside the GPU data: the six existing `scripts/v7/*.py` analysis scripts call functions that the current `scripts/perimage_v7.py` no longer has, so the current `results/v7` outputs cannot be reproduced from the current code.

## Summary table

| # | Check | Result |
|---|---|---|
| 1 | Inventory of score and rescore files | PASS |
| 2 | Rescore P0 equals plain scores | PASS for 8 families; NOTE for nnU-Net (re-prediction noise, no effect on means or ranks) |
| 3 | Grid Dice at own threshold equals P0 | PASS (exact) |
| 4 | Thresholds, held-out Dice, degenerate models, MAnet | PASS with NOTES (seed spread is large; MAnet 3 of 5 run-2 trainings diverged) |
| 5 | nnU-Net held-out n = 148 | PASS (2 held-out images are copies of test images and were removed by design) |
| 6 | SAM v2 groups and empty nnU-Net boxes | PASS (handling is consistent); NOTES for interpretation |
| 7 | Synthetic sets regenerated identically | PASS (945 of 945, both runs) |
| 8 | Independent recomputation vs `perimage_v7.load` | PASS (max difference 0.0) |
| 9 | Log scan | No unreported anomaly; details below |
| extra | Existing V7 scripts vs current loader | FAIL (scripts use a loader API that no longer exists) |

## 1. Inventory [check1_inventory]

Files checked: 328 score files (4 run-1 folders x 33 sets, 5 run-2 model folders x 34 sets, 2 SAM v2 folders x 13 sets) and 117 rescore files (9 families x 13 sets).

- Every file has the expected set list, a `set` field equal to its file name, the same model names in every set, and `n` rows of `len(names)` values for both Dice and NSD.
- All values are finite and inside [0, 1].
- Model names are as expected: pretrained_s2 has 9 (no manet_r18_in_s2), modern2_s0 and modern2_s2 have 7 (no manet_r50_s0, manet_r50_s2), all others complete.
- Clinical sets (12): `n`, `gt_empty` and `gt_fraction` are bit-identical to `results/perimage32` in all 11 score folders and all 9 rescore families (largest difference 0.0).
- Synthetic sets (21): `gt_empty` and `gt_fraction` identical in all 9 folders that score them.
- fuseg_r256: identical in 7 score folders and 9 rescore families, and identical (difference 0.0) to my own recomputation from `/home/claude/kaggle_kit/refs/fuseg` using the resize rule in `k2_common.fuseg_r256_dir`.
- Rescore thresholds equal the threshold files for every model; rescore model lists equal the score-file model lists.
- No WoundSeg file exists anywhere in `results/kaggle2` (expected after the failed stage).
- No result file was modified after download (all file times are 2026-10-10; the folder is identical to `/home/claude/k2_results`).

## 2. Rescore P0 versus plain score files [check2_rescore_vs_scores]

Rescore values are rounded to 5 decimals by `k2_rescore._r`, so agreement means a difference of at most 5.0e-6.

| Family | Compared with | Image-model pairs | Max abs diff Dice | Max abs diff NSD (2 px) |
|---|---|---|---|---|
| pretrained_s0 | kaggle/scores/pretrained | 17,120 | 5.0e-6 | 5.0e-6 |
| pretrained_s1 | kaggle2/scores/pretrained_s1 (incl. fuseg_r256) | 18,320 | 5.0e-6 | 5.0e-6 |
| pretrained_s2 | kaggle2/scores/pretrained_s2 | 16,488 | 5.0e-6 | 5.0e-6 |
| modern2_s0 | kaggle2/scores/modern2_s0 | 12,824 | 5.0e-6 | 5.0e-6 |
| modern2_s1 | kaggle2/scores/modern2_s1 | 14,656 | 5.0e-6 | 5.0e-6 |
| modern2_s2 | kaggle2/scores/modern2_s2 | 12,824 | 5.0e-6 | 5.0e-6 |
| kvasir_nets | kaggle/scores/kvasir | 17,120 | 5.0e-6 | 5.0e-6 |
| kvasir_unet | kaggle/scores/kvasir | 1,712 | 5.0e-6 | 5.0e-6 |
| nnunet | kaggle/scores/nnunet | 1,712 | 3.38e-3 | 5.40e-3 |

The run-2 families (pretrained_s1, s2, modern2) were not in the built-in consistency check of the run; they pass here.

### nnU-Net in detail [check2_rescore_vs_scores.nnunet_detail]

- 1,540 of 1,712 image results are identical within rounding. 107 differ by more than 1e-4 in Dice (6.3%), only 2 by more than 1e-3: co2wounds image 117 (0.74775 to 0.75113; the lesion covers 0.25% of the image, so a few pixels move Dice by 0.003) and tn3k_test image 139 (0.54140 to 0.54024).
- Among the 107: median |difference| 2.1e-4, 75th percentile 3.7e-4, 90th percentile 4.4e-4.
- Direction is random: 63 higher in run 2, 44 lower, mean signed difference 4.3e-6.
- NSD (2 px): 42 images differ by more than 1e-4, 23 by more than 1e-3, largest 5.4e-3.
- No image with an empty reference changed.
- By set (images with |dDice| > 1e-4): medetec 14, co2wounds 12, fuseg 6, azh 20, azh_train_slice 26, kvasir_disjoint 7, kvasir_sessile 1, cvc_clinicdb 3, cvc_colondb 1, etis_larib 0, ph2 9, tn3k_test 8.
- Effect on dataset means after exclusions: largest change 2.6e-5 in mean Dice (co2wounds) and 8.3e-5 in mean NSD (co2wounds).
- Rank of nnU-Net among the 11 V6 modern methods: unchanged on all 12 datasets, for both Dice and NSD. The closest other method is never nearer than 6.0e-4 (cvc_colondb), which is 150 times the change on that set. The rank among the 19 seed-averaged modern methods is also unchanged on all 12 sets [check4_models.manet_sensitivity].

Most likely cause: nondeterministic re-prediction under half precision. Evidence:
1. The code that turns nnU-Net output into a mask is the same in both runs (`kit_run.stage_nnunet.predict` and `k2_rescore._nn_prob`: channel 1 of the saved probabilities, `postprocess_prob(p, 0.5)`). So "threshold versus argmax" is ruled out; both runs threshold the same way.
2. Same model files (restored from run 1) and the same nnunetv2 version (both notebooks pin `nnunetv2==2.8.1`).
3. nnunetv2 2.8.1 inference sets `torch.backends.cudnn.benchmark = True` (`predict_from_raw_data.py` line 61) and runs under `torch.autocast` in fp16 on CUDA (line 681). Two Kaggle sessions can pick different convolution kernels, which changes the last bits of the fp16 logits, which flips the odd pixel whose smoothed probability sits at 0.5.
4. The smp networks run in fp32 without autocast (`kit_run.pt_prob`, `sq_prob`) and reproduce to rounding across the same two sessions.
5. The pattern (sparse, random in sign, largest on tiny lesions) is what pixel flips produce. A pipeline difference would shift most images in one direction.

Decision: use the run-1 nnU-Net scores (`results/kaggle/scores/nnunet`) for every P0 analysis. This matches V6 and what `perimage_v7.load` already does. Use run-2 nnU-Net values only where nothing else exists (P1, P2, NSD tolerances, threshold grid, fuseg_r256). When you compute a variant minus P0 difference for nnU-Net, take both numbers from the rescore file; mixing run-1 P0 with run-2 P1 would add the re-prediction noise to the difference.

## 3. Grid Dice versus P0 [check3_grid]

- All 9 families: grid Dice at the model's own threshold equals P0 Dice exactly (max difference 0.0 over 115,416 image-model comparisons). Both use the same function, `postprocess_prob(p, t)`.
- The grid is the 18 values 0.05 to 0.90 in steps of 0.05 in every file. Every model threshold lies on the grid and none sits at an edge (lowest 0.10 for fpn_r18_kv and unet_mbv2_kv, highest 0.65 for linknet_r18_in).
- For information: choosing the best grid threshold on the FUSeg test images would raise a FUSeg-trained smp model by at most 0.022 Dice (pspnet_r18_in), and nnU-Net (fixed 0.5, never tuned) by 0.015 (best at 0.30).

## 4. Thresholds, held-out Dice, degenerate models, MAnet [check4_models]

Threshold / held-out FUSeg Dice (148 held-out training images, threshold chosen on the same images). Seed 0 of the original ten comes from run 1 (`results/kaggle/thresholds_pretrained.json`, `train_report_pretrained.json`); everything else from `results/kaggle2/thresholds_*.json` and `train_report_k2.json` (thresholds in the report equal the files).

| Architecture | Set | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|---|
| unet_r18 | original | 0.35 / 0.827 | 0.30 / 0.831 | 0.35 / 0.835 |
| unetpp_r18 | original | 0.40 / 0.835 | 0.30 / 0.838 | 0.35 / 0.837 |
| deeplabv3p_r18 | original | 0.35 / 0.808 | 0.30 / 0.800 | 0.30 / 0.794 |
| fpn_r18 | original | 0.30 / 0.808 | 0.35 / 0.798 | 0.30 / 0.819 |
| pspnet_r18 | original | 0.15 / 0.748 | 0.20 / 0.748 | 0.30 / 0.754 |
| linknet_r18 | original | 0.65 / 0.806 | 0.45 / 0.809 | 0.55 / 0.818 |
| manet_r18 | original | 0.40 / 0.843 | 0.25 / 0.812 | diverged |
| segformer_mitb0 | original | 0.35 / 0.778 | 0.40 / 0.781 | 0.35 / 0.783 |
| unet_mbv2 | original | 0.40 / 0.816 | 0.35 / 0.819 | 0.40 / 0.817 |
| unet_effb0 | original | 0.35 / 0.806 | 0.35 / 0.812 | 0.35 / 0.818 |
| unet_r50 | new | 0.45 / 0.811 | 0.50 / 0.815 | 0.35 / 0.818 |
| deeplabv3p_r50 | new | 0.30 / 0.800 | 0.30 / 0.804 | 0.25 / 0.799 |
| upernet_convnext_t | new | 0.50 / 0.809 | 0.40 / 0.814 | 0.40 / 0.826 |
| unet_swin_t | new | 0.40 / 0.838 | 0.25 / 0.827 | 0.50 / 0.830 |
| segformer_mitb2 | new | 0.40 / 0.820 | 0.30 / 0.803 | 0.40 / 0.808 |
| unet_effb3 | new | 0.20 / 0.805 | 0.40 / 0.807 | 0.25 / 0.807 |
| fpn_regnety016 | new | 0.35 / 0.789 | 0.35 / 0.811 | 0.20 / 0.811 |
| manet_r50 | new | diverged | 0.40 / 0.826 | diverged |

Findings:
- No model has held-out Dice below 0.7 (lowest 0.748, pspnet_r18 seeds 0 and 1, matching run 1). The only model more than 0.03 below its siblings is manet_r18_in_s1 (0.812 against 0.843 for seed 0); other architectures show seed spreads up to 0.022 (fpn_regnety016: 0.789 to 0.811), so this is ordinary.
- All 41 run-2 report entries carry the intended protocol: correct seed, 40 epochs, 392 training and 148 sweep images, ImageNet weights, torch 2.11.0+cu128, smp 0.5.0, timm 1.0.22, and 0 non-finite steps for every saved model.
- No degenerate output on wound sets. The smallest per-image Dice SD of any model on any wound set is 0.171. The 29 "near constant" cases (per-image SD below 0.01) are all on off-domain sets (tn3k_test, cvc_colondb, kvasir_sessile, ph2) where the mean Dice is at most 0.002: wound models predict nothing on polyps and thyroid nodules, as expected.
- Training loss is high for linknet_r18_in_s1 (0.306 at epoch 40), linknet_r18_in_s2 (0.490) and unet_r50_s1 (0.197), against 0.07 to 0.13 for the rest. Their held-out Dice is normal (0.809, 0.818, 0.815), so this reads as less confident probabilities, not a failure.
- Seed-to-seed spread of wound-set means is large for some architectures: fpn_regnety016 0.145 (azh_train_slice: 0.689 vs 0.571, 0.544), unet_swin_t 0.129 (azh: 0.568 vs 0.653, 0.697), unet_r50 0.115, unet_effb3 0.111, pspnet_r18 0.106. Pooled within-architecture SD of dataset means across seeds: 0.010 (fuseg) to 0.034 (azh_train_slice). Real training noise, not an error.

MAnet divergence (from `log.txt`; 25 optimizer steps per epoch, 1,000 per training):

| Model | Non-finite steps by epoch 10 / 20 / 30 / 40 | Logged mean loss at epoch 40 | Non-finite tensors at the end |
|---|---|---|---|
| manet_r18_in_s2 | 0 / 1 / 4 / 4 | 0.140 (finite) | 36, examples are decoder BatchNorm running_mean and running_var |
| manet_r50_s0 | 0 / 2 / 134 / 367 | none (every step of epoch 40 non-finite) | 397, including encoder.conv1.weight |
| manet_r50_s2 | 41 / 247 / 497 / 747 | none from the epoch-20 log on | 397, including encoder.conv1.weight |

- manet_r18_in_s2 first overflowed between epochs 11 and 20, only 4 times. Its loss stayed normal but the BatchNorm running statistics were corrupted. Likely mechanism: under fp16 autocast, the GradScaler skips the optimizer step for an overflowed batch, but BatchNorm updates its running statistics during the forward pass, so the inf/NaN is written there anyway. Training loss uses batch statistics, so it looks fine; evaluation would not.
- manet_r50_s0 (from epochs 11 to 20 on) and manet_r50_s2 (already within epochs 1 to 10) diverged completely.
- MAnet failed in 3 of its 5 run-2 trainings. The 39 run-2 trainings of other architectures had no non-finite step at all.
- Surviving MAnet models look normal: manet_r18_in (run 1) held-out 0.843; manet_r18_in_s1 0 non-finite steps, final loss 0.112, held-out 0.812; manet_r50_s1 0 non-finite steps, final loss 0.103, held-out 0.826. The two manet_r18 seeds differ by at most 0.047 on any wound set, inside the range of other architectures.
- Sensitivity [manet_sensitivity]: removing both MAnet columns moves the 19-method dataset mean by -0.0030 (fuseg) to +0.0035 (ph2). It does matter for ranks, though: manet_r18 (2 seeds) ranks 1st of 18 architectures on medetec and 2nd on fuseg; manet_r50 (1 seed) ranks 17th on medetec, 18th on co2wounds, and 1st on fuseg and tn3k_test.

## 5. nnU-Net held-out file [check5_nnunet_heldout]

- n = 148 is correct. `kit_run.fuseg_split` takes held-out indices 400 to 549 and removes `manifest.json` `fuseg_train.eval_duplicates` = [93, 452, 506]. Two of these (452 = FUSeg train 0569.png, 506 = 0636.png) fall in the held-out range. I checked pixel by pixel that they are identical to evaluation images refs/fuseg 0109 and 0070 (already excluded in `exclusions_v5.json` as 0545.png and 0356.png). Index 93 (0121.png, identical to eval 0056) is in the training range, which is why training has 392 images. `_heldout_files` also checks the files against `fuseg_split` at run time.
- Mean held-out Dice 0.8328 (148 finite values; recomputed 0.83280). 11 images are below 0.5; minimum 0.
- Run-1 nnU-Net on the FUSeg test set: 0.7943 after exclusions (112 images), 0.7959 on all 120. Run-2 re-prediction: 0.7943.
- For comparison, smp held-out Dice ranges from 0.748 to 0.843 (median 0.811), but those numbers are optimistic because each threshold was chosen on the same 148 images. nnU-Net's 0.5 was fixed.

## 6. SAM v2 groups [check6_sam_v2]

Mean Dice after exclusions (all images for fuseg_r256):

| Set | n | MedSAM fixed, GT box | Run-1 MedSAM, GT box | Run-1 SAM, GT box | Fixed minus run-1 MedSAM | SAM, nnU-Net box | MedSAM fixed, nnU-Net box | nnU-Net run 2 | nnU-Net run 1 |
|---|---|---|---|---|---|---|---|---|---|
| medetec | 152 | 0.728 | 0.376 | 0.885 | +0.352 | 0.762 | 0.589 | 0.764 | 0.764 |
| co2wounds | 150 | 0.659 | 0.552 | 0.792 | +0.107 | 0.460 | 0.360 | 0.474 | 0.474 |
| fuseg | 112 | 0.728 | 0.752 | 0.896 | -0.024 | 0.800 | 0.653 | 0.794 | 0.794 |
| azh | 150 | 0.664 | 0.585 | 0.908 | +0.080 | 0.641 | 0.518 | 0.631 | 0.631 |
| azh_train_slice | 103 | 0.660 | 0.559 | 0.887 | +0.102 | 0.549 | 0.505 | 0.548 | 0.548 |
| kvasir_disjoint | 120 | 0.839 | 0.792 | 0.932 | +0.047 | 0.041 | 0.042 | 0.044 | 0.044 |
| kvasir_sessile | 120 | 0.752 | 0.719 | 0.885 | +0.033 | 0.013 | 0.014 | 0.010 | 0.010 |
| cvc_clinicdb | 150 | 0.724 | 0.736 | 0.904 | -0.012 | 0.055 | 0.037 | 0.056 | 0.056 |
| cvc_colondb | 150 | 0.686 | 0.675 | 0.887 | +0.011 | 0.003 | 0.009 | 0.003 | 0.003 |
| etis_larib | 150 | 0.750 | 0.798 | 0.896 | -0.048 | 0.021 | 0.021 | 0.020 | 0.020 |
| ph2 | 150 | 0.561 | 0.813 | 0.887 | -0.253 | 0.471 | 0.283 | 0.440 | 0.440 |
| tn3k_test | 150 | 0.698 | 0.747 | 0.764 | -0.049 | 0.131 | 0.109 | 0.140 | 0.140 |
| fuseg_r256 | 120 | 0.712 | n/a | n/a | n/a | 0.754 | 0.622 | 0.726 | n/a |

Empty nnU-Net masks (counted on all images, before exclusions):
- How the code handles them (`k2_sam.predict_nn`): `mask_boxes` calls `kit_run.gt_boxes`, which returns no box for an empty mask. `sam_proc_prob` and `medsam_prob` then return an all-zero map, so the prediction is empty. The image is scored (Dice 0 against a lesion, 1 against an empty reference), not skipped.
- Images with a lesion where nnU-Net most likely predicted nothing (nnU-Net Dice 0 and every NSD 0, and both nnU-Net-box methods Dice 0): at most 711 of the 732 lesion images where nnU-Net has Dice 0. By set: kvasir_disjoint 87, kvasir_sessile 108, cvc_clinicdb 110, cvc_colondb 143, etis_larib 134, tn3k_test 74, co2wounds 29, ph2 14, fuseg 6, fuseg_r256 6, medetec 0, azh 0, azh_train_slice 0. This is an upper bound because a mask that is not empty but lies far from the lesion gives the same pattern.
- On reference-empty images nnU-Net predicted nothing 4 times (fuseg 1, fuseg_r256 3), and both nnU-Net-box methods scored 1 on all 4. There are 0 inconsistencies.
- The oracle group gives no box and a score of 1 to all 26 images with an empty reference (fuseg 4, azh 10, azh_train_slice 8, fuseg_r256 4), the same rule as run 1.

## 7. Synthetic sets [check7_synth]

`results/kaggle2/synth_check.json`: 21 sets x 45 images, 945 of 945 identical to the fingerprints of the locally scored images (log line 164). Run 1 also had 945 of 945. The reference masks of all 21 sets are identical across all 9 folders (check 1).

## 8. Independent recomputation [check8_recompute]

For medetec (152 images, no exclusions) and azh (150 images, no exclusions), I computed the mean Dice of the 19 seed-averaged modern methods directly from the score files. That means a per-image mean over the available seeds, then the mean over images, plus run-1 nnU-Net. The largest difference from `perimage_v7.load` is 0.0 on both sets, and the `M19` group order matches. Seeds used: manet_r18 2, manet_r50 1, all others 3. The exclusion indices in `exclusions_v5.json` match their stored file names (0 mismatches for fuseg and azh_train_slice).

Means (medetec / azh): unet_r18 0.7536 / 0.5800, unetpp_r18 0.6847 / 0.5853, deeplabv3p_r18 0.7554 / 0.5083, fpn_r18 0.7356 / 0.5750, pspnet_r18 0.5681 / 0.5437, linknet_r18 0.7165 / 0.5674, manet_r18 0.7707 / 0.6055, segformer_mitb0 0.7106 / 0.6122, unet_mbv2 0.6917 / 0.5323, unet_effb0 0.6994 / 0.5585, unet_r50 0.6872 / 0.5758, deeplabv3p_r50 0.7105 / 0.5456, upernet_convnext_t 0.7277 / 0.6112, unet_swin_t 0.6984 / 0.6395, segformer_mitb2 0.7315 / 0.7131, unet_effb3 0.6927 / 0.5614, fpn_regnety016 0.7202 / 0.6303, manet_r50 0.6573 / 0.5949, nnunet_2d 0.7639 / 0.6315.

## 9. Log scan [check9_log]

Stage order in `results/kaggle2/log.txt` (863 lines): restore done; woundseg FAILED (line 11, 322 of 2686 size mismatches); seeds FAILED (line 235, only because of manet_r18_in_s2); rescore done (first pass, lines 250 to 426); sam2 done; modern2 FAILED (line 734, only manet_r50_s0 and s2); rescore done (second pass, lines 749 to 862); pack started at line 863.

Other items:
- First rescore pass ran before modern2 was trained, so it logged "no model could be loaded, skipped" for modern2_s0 to s2 (lines 298 to 324). The second pass scored them (lines 760 to 801) and skipped families already done. Expected given the stage order; the files are complete.
- The SAM nnU-Net-box stage used the nnU-Net masks from the first rescore pass. nnU-Net was not re-predicted in the second pass (line 804).
- nnU-Net predicted 1,980 images = 1,712 clinical + 120 fuseg_r256 + 148 held-out (line 354).
- The consistency check printed the nnU-Net "LARGE" flag in both passes with identical numbers (lines 424, 860), explained in section 2.
- No WARNING line, no file moved to stale/, no "recomputing" line, so no score file was invalidated or rewritten mid-run.
- The 13 repeated "run-1 output candidates" lines per family come from `consistency_check` calling `find_prev` once per set. Harmless.
- `DONE_pack` is dated 21:58:58, from the pack before the second rescore. The final pack's completion line and marker are missing because the log was zipped while that pack was still running. The bundle does contain the second-pass modern2 rescore files and `DONE_rescore` (22:24:44), so it is the final pack's output.
- The run-1 log has no FAIL, WARN, error or NaN line.

## Extra check: existing V7 scripts versus the current loader (FAIL)

`scripts/perimage_v7.py` was rewritten on 2026-10-11 at 17:35. The six analysis scripts in `scripts/v7/` (bench, practical, sam_checks, seeds, stats, rescore_local; written 10:31 to 10:40) and `BRIEF_V7.md` use an older API: `load7`, `load_seed`, `GROUPS7`, `NAMES7`, `ARCH`, `SEEDS`, `seeds_of`, columns `<arch>@avg`, groups M17 and M18p. None of these exist in the current file, which uses `load`, `GROUPS`, columns `<base>_in@seeds` / `<base>@seeds`, and has no M17 (no-MAnet) group. So the current `results/v7/*.json` and `*_V7.md` were produced by a loader that is gone, and the scripts would fail if run now. Before any V7 number goes into the paper, either restore the old functions or port the scripts to the new loader, and regenerate every V7 output.

## Recommendations

(a) nnU-Net scores: use run 1 (`results/kaggle/scores/nnunet`) for all P0 Dice and NSD analyses. Use run-2 rescore values only for P1, P2, the other NSD tolerances, the threshold grid and fuseg_r256, and compute nnU-Net variant-minus-P0 differences within the rescore file. Mention in the methods that the re-prediction agrees with run 1 within 0.0034 Dice per image and 0.000026 per dataset mean.

(b) MAnet: keep the available seeds (manet_r18: seeds 0 and 1; manet_r50: seed 1) in the main analysis, and report a sensitivity analysis without both MAnet columns. Reasons: every surviving MAnet model passes every check (finite, 0 non-finite steps, held-out 0.812 to 0.843, wound means inside normal seed spread); the failures are fp16 training overflows that have nothing to do with the evaluation data; and dropping MAnet after the fact would remove the top medetec architecture and change ranks. Say in the methods that MAnet diverged in 3 of 5 run-2 trainings (0 of 39 for other architectures). Mark manet_r50 as a single-seed column: its seed noise is about 1.7 times that of a 3-seed average (within-architecture SD 0.010 to 0.034 on wound sets). In per-seed comparisons, use only architectures present in both seeds.

## Caveats and limits of this audit

- The empty-mask count in section 6 is an upper bound. The nnU-Net masks themselves (`nnunet_masks/*.npz`) were not packed.
- The nnU-Net cause is inferred from code, versions and the pattern of differences. Proving it would need a second GPU prediction of the same images in one session.
- MedSAM with official preprocessing (oracle boxes) is 0.253 below run-1 MedSAM on ph2, and lower on etis_larib, tn3k_test, fuseg and cvc_clinicdb. `SAM_CHECKS_V7.md` reports the same and found no preprocessing mismatch by reading the code; I found none either. This cannot be settled without running the official MedSAM script on a few PH2 images. Report it as observed behaviour, not as an error.
- Seed 0 of the original ten architectures was trained in run 1 and seeds 1 and 2 in run 2. smp 0.5.0 and nnunetv2 2.8.1 were pinned in both notebooks, but run 1 did not record the torch and timm versions. For the original ten, seed effects therefore include any session difference. The eight new architectures were all trained in run 2.
- I checked consistency and plausibility of the files, not the correctness of the masks against the images.
