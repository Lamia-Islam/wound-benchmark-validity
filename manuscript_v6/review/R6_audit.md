# R6: independent number audit of the revised manuscript (exclusions_v6)

Scope: `manuscript_v6/manuscript_v6_blinded.docx` at commit 4a8b18e (its text is identical to the build I audited; that commit changed only layout), on 16 October 2026: the text, Tables 1 to 5, the captions, the footer and the footnote parts. I cross-checked it against `facts_v6.json`, the results JSON in `rebuild/results/v7/`, the raw per-image files, `Online_Resource_1.docx` and `Online_Resource_2.xlsx`. I edited no project file. My scratch code is in `/tmp/claude-0/-home-user/600f97c5-f540-5493-b6d5-259c76b6f050/scratchpad/r6/`: `recompute.py` (main recomputation), `extra.py`, `nncheck.py`, `halves.py`, `selrow.py`, `ties.py`, `excl_check.py`, `or2check.py` and `extract.py`. All Python ran with `python3 -I`, and none of it imports a project loader.

## Summary verdict

- **No wrong number.** Every printed number equals its JSON source after rounding. Each fact key means what its sentence says (method set, metric, pair category, dataset and interval type). Rounding and signs follow the house rules: 2 decimals for correlations and Dice, 3 for selection losses, 1 for percentages, 2 significant digits for p, and U+2212 for minus signs.
- **The pipeline is current.** I regenerated `facts_v6.json` from the results JSON into my scratch directory: 2,347 facts, identical to the committed file. I rebuilt the docx from the current `build_v6.js` and its text is identical to the audited docx. A keyed rebuild, in which each number carries its fact key, ties all 799 numbers in the text, tables and captions to 556 distinct fact keys.
- **Independent recomputation from the raw per-image files:** 381 comparisons. 371 numeric values are bit-identical to the facts (difference 0 or below 1e-12), and 10 categorical values (model names and the threshold argmax) also match. None needed rounding to agree, and none disagreed. A further 8 values that depend on random splits agree within resampling noise.
- **The exclusion claims in Methods are all correct.** I recounted them from `exclusions_v6.json` and the two overlap files: FUSeg 8 (7 + 1); AZH 61 matches, 4 removed, 57 kept, 146 left; control 70 crops (51 + 19) plus 2 copies, 78 left, 1 crop kept. The 12 AZH matches to FUSeg validation frames are frames 0757 to 0880, outside the reference set (the reference ids end at 0582). The Methods are right on this point, and HANDOFF.md section 5 P0 is wrong.
- **House rules.** The abstract has 241 words (limit 250). There is no em dash, and no ASCII hyphen-minus is used as a minus sign; there are 68 U+2212 signs. "relabelling" is spelled the British way 3 times and "relabeling" never appears. American spellings occur only inside reference titles. The blinded reference 60 reads "Reference removed for double-blind review".
- **Problems found: 0 critical, 1 major, 8 minor.** The major one is a qualitative "largest" claim that is false when the NSD tolerance changes are counted. The minor ones are wording, an understated bound, a mismatched method set, Online Resource cross-references and display, and provenance.

## Problem table

| # | Severity | Exact quote | What is wrong | Correct value and source | Suggested fix |
|---|---|---|---|---|---|
| 1 | major | "Apart from the common threshold described above, no change has an interval that excludes zero; the largest was the oracle threshold (−0.09, −0.28 to 0.09), so changes of that size cannot be ruled out." | The same paragraph also covers NSD and its tolerances. Under NSD, moving the tolerance from 2 to 5 pixels changes M19 wound-pair agreement by −0.17 (−0.42 to 0.10), almost twice the oracle change. The "no interval excludes zero" part is true. | `robust_v7.json: tolerance_delta_vs_nsd2/M19|nsd5-nsd2|wound pairs` est −0.170, ci [−0.418, 0.095]. Next largest: nsd3 −0.087; post-processing at most +0.051; FUSeg at 256 px −0.040. | "...no change has an interval that excludes zero; the largest were the oracle threshold under Dice (−0.09, −0.28 to 0.09) and the 5-pixel NSD tolerance (−0.17, −0.42 to 0.10), so changes of that size cannot be ruled out." Add a fact key such as `rob.M19.nsd5.d`. |
| 2 | minor | Methods: "there the Pearson value differs from the weighted average by at most 0.005 for a single pair" | The maximum is 0.0054 (U+F, Kvasir-sessile with TN3K, Dice; 0.0052 under NSD). Rounding an upper bound down makes the "at most" claim false. | `review_v7.json: R3_m1_stratification_identity/dice|U+F/max_abs_single_pair_diff...` = 0.005416. Recomputed: 0.005416. | "by at most 0.006", or "by about 0.005". In `facts_v6.py`, round bounds up (the way `rv.ident.maxmean` is printed as a bound). |
| 3 | minor | "Excluding empty reference masks leaves the modality differences unchanged (U+M19 1.10, M19 0.23)" | M19 goes from 0.22 (Table 3 and the text) to 0.23 (0.2247 to 0.2300), and U+M19 from 1.101 to 1.097. Next to the printed 0.22, "unchanged" reads as wrong. | `stats_v7.json: dice_noempty/M19/modality_diff` 0.2300; `point/dice|M19/modality_diff` 0.2247. | "...leaves the modality differences nearly unchanged (U+M19 1.10, M19 0.23 against 0.22) and wound-pair agreement similar (U+M19 0.81, M19 0.24 against 0.21)". |
| 4 | minor | "Agreement between the datasets shows why. Among the 16 seed-averaged architectures FUSeg agreed with Medetec at −0.21, with CO2Wounds-V2 at −0.01 and, from the same clinic, with AZH at 0.17, while Medetec and CO2Wounds-V2 agreed at 0.68" | The selection uses the 19 M19 candidates. The 16 architectures quoted leave out the model chosen on FUSeg (MA-Net ResNet-50, a single-seed model) and nnU-Net. The values are correct for C16. Also, among the same 16, AZH agrees with Medetec at only 0.16 and with CO2Wounds-V2 at only 0.22, yet choosing on AZH loses only 0.020. Pairwise agreement therefore does not by itself "show why". | M19 values (own recomputation; OR2 Pair agreement sheet): FUSeg with Medetec −0.12, with CO2Wounds-V2 −0.13, with AZH 0.21; Medetec with CO2Wounds-V2 0.73. Row means of the loss matrix: Medetec 0.050, CO2Wounds-V2 0.020, FUSeg 0.101, AZH 0.020. | Quote the M19 values (new fact keys from the OR2 or stats pairs) and write "is consistent with this" in place of "shows why". |
| 5 | minor | Methods: "If cross-seed agreement were as low as cross-dataset agreement, the modern models would be treated as not distinguishable. Details are in Online Resource 1 (Table S4)." | Table S4 in OR1 holds the seed results but does not describe the rule. OR1 contains no "0.7" threshold and no mention of a pre-specified or reading rule. | OR1 text search: "pre-specified", "rule" (apart from the additive and product rules), "0.7" not found in Table S4. | "The quantities the rule refers to are in Online Resource 1 (Table S4)", or add the rule wording to the Table S4a or S4d caption. |
| 6 | minor (OR1 display) | Results: "the rise has an interval above zero for every common threshold from 0.45 to 0.90"; OR1 Table S6d, row 0.45: "0.163 [0.000, 0.320]" | The claim is true (lower bound 0.00028), but OR1 prints the lower bound as 0.000. A reader who checks OR1 sees an interval that touches zero. | `robust_v7.json: threshold/delta_vs_default/M19|0.45|wound pairs/ci` = [0.000280, 0.3203]. | In OR1 Table S6d, print 4 decimals for interval ends below 0.001, or add a note: "lower bound at 0.45 is 0.0003". |
| 7 | minor (traceability) | Several main-text numbers from `review_v7.json`: Holm "largest adjusted p 0.0092"; "difference 0.90, 0.16 to 1.50" and "difference 0.49, −0.03 to 1.05"; "1.20 FUSeg pixels", "about 1.9 times", "about 0.83 times"; "0.79, against 0.68"; "0.45 ... 0.36" (halves); "loses 0.001 to 0.024"; "0.93" (synthetic seeds); "−0.016 (−0.033 to 0.003)"; Table 5 "0.032 to 0.073 (unit)" and "0.055 to 0.081 (unit)"; "0.42 (−0.07 to 0.83)"; "5.7 percentage points, 3.4 to 8.0 ... −0.8 to 13.0"; "without FUSeg it is 0.35"; "0.84 to 0.99" | All values are correct; I recomputed many of them (see below). But neither Online Resource reports them, so a reader cannot trace them. R1 found every non-integer manuscript fact in OR1; that no longer holds. | `results/v7/review_v7.json` and `REVIEW_V7.md` | Optional: an OR1 table "Additional analyses requested in review" listing these values with their definitions. |
| 8 | minor (provenance) | Methods: "The pixel classifiers were fitted on 247,975 pixels sampled from images 0 to 249 ... tuned on images 250 to 399" | The values are right, but the fact keys `d.px.img` and `d.px.thrimg` are read by regex from `kaggle/kit_run.py: kvasir_split()`, the Kvasir split. The FUSeg pixel classifiers are defined in `woundbench/learned.py` (`N_TRAIN_IMAGES_PIXEL = 250`) and `scripts/fit_learned.py` (images 0 to 250 for fitting, 250 to 400 for the threshold sweep). The ranges coincide. | `scripts/fit_learned.py` lines 61 and 80; `woundbench/learned.py` line 26 | Point the two fact sources at `fit_learned.py` and `learned.py`. |
| 9 | minor | Abstract: "The model ranked first on the test set from the training clinic lost 0.101 Dice on other wound datasets, against 0.075 for a random choice." | 0.101 is the mean over the three other datasets (0.113, 0.069, 0.119). The same abstract says "on average" for the miss rate, and the Results say "on average" here. | `practical_v7.json: selection/M19/matrix_rows.../fuseg` | "...lost 0.101 Dice on average on the other wound datasets..." The abstract stays at 242 words. |

Not counted as problems:
- "rises to 0.40 (...; change 0.20, 0.03 to 0.35)" while the main protocol is printed as 0.21. The printed values suggest 0.19, but the raw values are 0.404 − 0.206 = 0.198, which correctly rounds to 0.20.
- Values that round to zero from below print as "0.00", not "−0.00" (for example `sd.avgminus` −0.002 and `b.M19.mean.diff` −0.0006). This is consistent throughout.
- Reference page ranges use ASCII hyphens between digits, such as "pp. 37401-37412". These are not minus signs, and the JIIM example uses the same form.
- Outside the manuscript: HANDOFF.md (section 5, P0, line 70) says "The 12 validation frames are our FUSeg reference set". That is not true (see Task 4), and the manuscript states it correctly.

## Counts of what I checked

- **Numbers in the docx:** 799 number tokens from placeholders, mapping to 556 distinct fact keys. Each was traced to its JSON or code source through the keyed rebuild. The only digits not supplied by fact keys are formula symbols ((r − 0.5)/n_g, 1/2, /12) and labels such as ResNet-18 and seed-0.
- **Meaning:** every key's `src` was checked against its sentence for method set, metric, pair category, dataset and interval type. All interval keys ending in `.m` or `.h`, and the unit and architecture intervals, carry the label the sentence or Table 5 row gives them. Value errors: 0. Meaning errors: 0. Wording issues: rows 1 to 5 and 9.
- **Tables:** Table 1 (84 cells), Table 2 (16 range ends and 4 n), Table 3 (70 cells), Table 4 (56 cells including intervals) and Table 5 (24 rows) are all correct.
- **Qualitative words checked against the data:** 46. Examples:
  - "exceeds the higher of the two" (0.47, interval above zero); "includes zero" (U+F excess, F and K benchmark, the difference between gaps).
  - "every per-reference interval above zero" (U+M19 texture: lower bounds 0.095, 0.124, 0.069, 0.077); "only 1 of the 16" (U on CO2Wounds-V2 only); "about 0.8".
  - "lead ... on every wound dataset and fall below them on every other dataset" (M19 against U on all 11); "most non-wound datasets" (U and K each below 0.1 on 1 of 7); "six of the seven" (F); "at most 0.07 on each of the seven" (0.068).
  - "with the seed on three of the four datasets" (AZH is the only exception); "every texture amplitude" (nnU-Net at 11, 28 and 40); "most settings"; "the worst strategy" (FUSeg row 0.101 against 0.050, 0.020 and 0.020).
  - "about as well as a random pick"; "several times the median gap" (seed SD over gap = 5.4 to 8.7); "below a hundredth" (largest median gap 0.006).
  - "neither branch of the rule applies" (0.44 is below 0.7 and above 0.09); "nine of the original ten and seven of the eight new ones"; "ten models; nine at seed 2"; "eight models; seven at seeds 0 and 2".
  - "each wound dataset less" (all four U+M19 per-reference differences are negative); "the best, SegFormer (MiT-B2)" (lowest AZH miss rate among the 18); "highest mean Dice" on AZH.
  - "In 39 ... the reverse occurs in 0"; "ties occur only on TN3K and PH2" (checked for Dice and NSD and all four groups); "for the wound pairs the identity is exact" (difference 0); "equals the size-weighted within-group average in every condition" (largest difference 2.5e-5 for U+M19); "each time with an interval above zero" (12 of 12); "in any condition" for U+F (0 of 12 hold); "every pooled set stays above zero" (12 of 12 intervals).
  - "no change of rank" for the nnU-Net re-prediction: I rechecked it against all 19 M19 means under the new exclusions, for Dice and NSD, and there is no rank change; the closest neighbour gap is 2.2e-5 against the 2.6e-5 shift.
  - "interval above zero for every common threshold from 0.45 to 0.90" (all 10 lower bounds are positive; 0.45 is 0.0003); "unchanged" (row 3); "shows why" (row 4); "the largest" (row 1).
- **Design numbers (Task 5):** 41 design and count values checked by reading the code they come from:
  - `woundbench/postproc.py`: element 5, elliptical; sigma 2; minimum component 0.0005; closing then opening for untrained methods; Gaussian smoothing then threshold for probability maps.
  - `woundbench/deep.py`: SIZE 128, EPOCHS 30, LR 1e-3, Adam, horizontal flips, Dice plus BCE, 117,681 parameters.
  - `woundbench/methods.py`: thresholds 0.9 and 0.5. `fit_learned.py`: 0 to 249 and 250 to 399. `results/refit_14.log`: 247,975 pixels.
  - `kaggle/kit_run.py`: training 0 to 392 without duplicates; thresholds 400 to 549; nnU-Net 2D, fold "all", 100 epochs, trainer only overrides the epochs, threshold 0.5.
  - `kaggle2/kit_run.py`: PT_SIZE 256, letterbox. `kaggle2/k2_train.py`: AdamW 3e-4, weight decay 1e-4, cosine schedule, batch 16, 40 epochs, horizontal and vertical flips plus rot90, autocast with GradScaler.
  - Train reports: 41 run-2 models; 51 individual models with held-out Dice 0.748 to 0.843; n_train 392, n_sweep 148; MA-Net 2 of 5 saved.
  - Rescore grid 0.05 to 0.90 in steps of 0.05 (18 values). `perimage32.py`: max side 256, threshold 127. `imagestats.py`: window 9 at 512 px. Black pixel means every channel at most 8. NSD at 2 px; tolerances nsd1, nsd3, nsd5 and nsd_diag1.
  - Bootstrap sizes: 2,000 method, 1,000 hierarchical (2,000 for the benchmark), 2,000 architecture, 2,000 unit, 500 splits.
  - Pair counts: 3, 13, 39, 16, 6, 28, 10 and 11; 52; 13,860 labellings (= 11!/(4! 5!)); minimum p 2/13,860.
- **Online Resources (Task 6):**
  - 362 non-integer main-text facts searched in the OR1 text. 358 match at 3 decimals or the same p; see row 7 for those that are reported only in the main text, where I also did a context check. No shared value disagrees.
  - OR1 Tables S9a and S9e and the OR2 README match the Methods counts, including the S9e training-split count of 52 = 51 used + 1 unused.
  - OR1 S7f black share 0.856 and 0.263 matches the main text's 85.6% and 26.3%. OR1 S7a and S7b and S7e match Table 5. S6e has 92.3% and 96.2%.
  - OR2: all 1,392 Mean Dice cells and 1,392 Mean NSD cells equal my recomputed means (largest difference 8e-16). The OR2 wound-pair means for U, F, K, M19, U+F, U+K and U+M19 equal mine to 1e-16. The OR2 Image sets sheet gives 146/118 for AZH and 78/71 for the control.

## Task 4: exclusions, counted independently

From `exclusions_v6.json`, `azh_overlap_v7.json`, `azh_control_overlap_v7.json`, `fuseg_train_index.json` and `perimage32/fuseg.json`, with my own classification of every matched photograph by how the study uses it:

- **FUSeg.** 8 excluded: 7 byte-identical to training images and 1 to validation image 0303. The indices match the ids.
- **AZH test, 150 checked, 61 SIFT matches:**
  - 4 match training frames used for training (indices 100, 100, 341, 319, all at most 392). They are excluded at indices 68, 69, 73 and 139, which match the ids.
  - 45 match training frames never used (index 550 or above, outside the kit's 550-file list).
  - 12 match validation frames 0757 to 0880. These are not in the reference, whose 120 ids run from 0001 to 0582.
  - None match a threshold-tuning frame (400 to 549) or the pixel-threshold range 250 to 399.
  - So 4 removed, 57 kept, 146 left: correct.
- **Control, 150 checked, 71 SIFT matches:**
  - 51 used for training (49 + 2 that tie with an unused frame) and 19 in the reference: 70 removed.
  - 1 tied to training file 0121 (the duplicate dropped from training, index 93) and to validation 0284 (excluded from the reference). It is kept, as Methods says.
  - 2 control images are pixel-identical to kept AZH test images (d8931..._0 and c8cf..._2) and are removed.
  - 2 further control images are identical to AZH test images that were themselves excluded; they were already removed as crops.
  - Total 72 removed, 78 kept: correct.
- **Methods wording:** "61 are crops ... 4 of these were cut from training images ... the other 57 come from FUSeg photographs that no model saw and that are not among the FUSeg reference images" and "70 ... (51 from its training split and 19 from the validation images used as reference), and 2 more are pixel-identical to AZH test images" are both exactly right.
- **Units:** I defined units by reading `perimage_v6.unit_of`: the id without its last `_k` suffix, for Medetec, AZH and the control. That gives Medetec 38 groups of exactly 4, AZH 118 and control 71.

## Independent recomputations and agreement

All of these come from my own code, reading `perimage32`, `kaggle/scores`, `kaggle2/scores`, `pool_decomposition_cells.jsonl`, `kaggle2/rescore` and `black_share_v7.json`. Exclusions are applied by index, seeds are averaged per image over the seeds that exist, and correlations use scipy spearmanr and pearsonr. "Identical" means a difference below 1e-12 from the fact's raw value.

| Block | Values | Agreement |
|---|---|---|
| Table 1: images, units, empty masks, median lesion fraction (12 sets) | 48 | identical |
| Table 3: all 7 sets (control, same source, different source, across, wound pairs, modality difference, NSD modality difference); colonoscopy pairs; composition categories (wound with other, non-wound across modality, colonoscopy with other); NSD wound pairs; empty-mask wound pairs and modality difference | 105 | identical |
| Exact relabelling p (13,860 labellings enumerated): 55 pairs for 7 sets; 52 pairs, NSD and empty-masks-excluded for U and K; M, seed 1, seed 2 and N8s differences and p | 37 | identical (including 0.00029, 0.0046, 0.00058, 0.0061, 0.00014, 0.00087, 0.38, 0.022, 0.33, 0.00043) |
| Table 2 ranges and the floor counts | 20 | identical |
| Stratified (Pearson on within-group percentiles): U+M19 wound pairs and modality difference; U+F and U+K modality difference; size-weighted average with n_g − 1/n_g weights (0.276224); largest single-pair difference | 8 | identical |
| Table 4: 7 sets x 4 references, mean, wound with wound and difference; texture effect for 7 sets; M19 per-reference range; K and U per-reference differences | 59 | identical |
| Single-seed benchmark ranges (P10, N8, M19 and P10 "other"), leave-one-out range, synthetic-seed agreement 0.93 | 12 | identical |
| Seeds: C16 r-bar 0.440, per-dataset range, control 0.832, cross 0.090, gap 0.350, single 0.172, average 0.170, difference; P9 and N7 gaps and r-bar; gap difference 0.417; NSD gap; six C16 pair agreements; 36 cross pairs | 25 | identical |
| Spread: M19 range 0.069 to 0.205, median gap, pooled seed SD, ICC for one seed and for three seeds | 10 | identical |
| Selection: off-diagonal 0.0476, maximum 0.119, minimum 0, random 0.0633; FUSeg row 0.1007 against random 0.0747; seed 0, 1 and 2 chooser against random (0.059, 0.077, 0.053 against 0.060, 0.085, 0.059) and candidate counts 18, 19, 17; synthetic loss 0.030 and seed losses 0.067, 0.051, 0.024; EfficientNet-B3 synthetic Dice 0.80, 0.67, 0.47; SegFormer B2 against nnU-Net AZH Dice 0.713 and 0.631 | 21 numeric + 9 names | identical |
| Misses: AZH P18 28.66%, P10, N8, nnU-Net 1.47%, SegFormer B2 12.3%, 136 non-empty images, largest seed spread 14.0 points, difference 5.7 points; FUSeg 2.6% and 4.6%; 51 models; Dice when detected (architecture mean 0.7886, nnU-Net 0.6827) | 13 | identical |
| Padding: black share 85.59% and 26.33% (all kept images), Spearman 0.732 and −0.462, 39 and 0, tercile shares 0.78/0.23, 0.38/0.06, 0.19/0.10 | 10 | identical for the means, correlations and counts; terciles equal at 2 decimals with quantile terciles (n 45, 45, 46, as in OR1 S7f) |
| Thresholds (rescore files): common 0.5 gives 0.404, change 0.198; maximum 0.625 at 0.90; oracle 0.115, change −0.090; oracle share at 0.05: Medetec 92.3%, CO2Wounds-V2 96.2% (52 models); calibration rho 0.272 and 0.545 | 10 | identical (threshold 0.90 matches as a value) |
| Robustness for M19 (rescore P0, P1, P2 x Dice and 5 NSD tolerances) | 18 | reproduce the endpoints of 0.12 to 0.26 and 0.31 to 0.53 under the "one change at a time" reading (P0 to P2 at 2 px, and the tolerances at P0) |
| nnU-Net re-prediction: largest per-image difference 0.0034, largest dataset-mean difference 2.6e-5, no rank change among M19 | 3 | identical |
| OR2 means (Dice and NSD) and wound-pair means | 2,791 | within 8e-16 |
| Random-split quantities (my own random splits): M19 split-half 0.867, 0.874, 0.816, 0.921 against the printed 0.82 to 0.92; U, F and K range 0.833 to 0.994 against the printed 0.84 to 0.99; shared and independent halves 0.453 and 0.359 against the printed 0.45 and 0.36; half-split selection 0.001 to 0.022 against the printed 0.001 to 0.024 | 8 | agree within resampling noise |

Totals, not counting OR2: 381 deterministic comparisons. 371 numeric values are bit-identical, 10 categorical values match, 0 needed rounding to agree and 0 disagree. A further 8 random-split values agree within noise.

I did not recompute the bootstrap intervals themselves (method, hierarchical, architecture and unit), because they depend on the scripts' random-number streams. I traced each one to its JSON path and checked its interval type against the sentence and the Table 5 labels.

## Resolution (applied after this audit)

1. Robustness paragraph: now names both largest changes, the oracle threshold under Dice and the 5-pixel tolerance under NSD (facts `tol5.d`, `tol5.d.m`; facts_v6.py asserts that the 5-pixel tolerance is the largest NSD change).
2. Identity bound: the single-pair bound is now rounded up (prints 0.006).
3. Empty masks: "nearly unchanged", with the main values given alongside.
4. FUSeg selection: now uses the M19 pair agreements (the candidate set of the selection analysis) and reads "is consistent with this"; the citation is Fig. S1.
5. Pre-specified rule: the sentence now cites Table S4 for the seed analyses only.
6. Table S6d: intervals print 4 decimals where a bound is not zero but rounds to zero at 3.
7. Traceability: Online Resource 2 has a new sheet, "Further analyses", that lists every review_v7.json value cited in the main text, with intervals; the Data availability statement mentions it.
8. Provenance: `d.px.img` and `d.px.thrimg` are now read from `woundbench/learned.py` and `scripts/fit_learned.py`.
9. Abstract: "on average" added (243 words).

After the fixes the full pipeline was rerun (`rebuild/scripts/v7/run_all_v7.sh`): blind replication 41/41, V6 reproduced under the old exclusions, no em dashes, no ASCII minus before a digit, no "relabeling".
