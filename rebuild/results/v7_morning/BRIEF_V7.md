# V7 analysis brief (phase-2 Kaggle results)

Study: does a synthetic wound-segmentation benchmark rank methods the way clinical data do, and does agreement between clinical datasets depend on which methods are pooled? Agreement = Spearman correlation between per-method mean scores (Dice or NSD) of two datasets, over a method set. V6 (committed, results/v6/) is the current analysis; V7 adds the phase-2 GPU results. Do NOT modify anything under scripts/v6, results/v6, scripts/perimage_v6.py, or any earlier results.

## Data
- `scripts/perimage_v6.py`: V6 loader. 11 references (REFS), control `azh_train_slice`, exclusions, resampling units (photograph for Medetec/AZH). Groups U 18 untrained, F 14 from scratch on FUSeg, K 14 Kvasir-trained, M 11 modern (10 seed-0 pretrained + nnU-Net), S 2 SAM/MedSAM with reference boxes (never pooled).
- `scripts/perimage_v7.py`: V7 loader (read it). Adds 18 seed-averaged architecture columns `<arch>@avg`; groups M19 (primary modern group: 18 architectures + nnU-Net), M17 (no MAnet), M18p (no nnU-Net), NEW8. `load_seed(ref, metric, seed)` gives per-seed matrices.
- Raw phase-2 files: `results/kaggle2/` (log.txt, scores/<group>/<set>.json in kit layout, rescore/<family>/<set>.json, nnunet_heldout.json, thresholds_*.json, train_report_k2.json, rescore/consistency.json). Run-1 Kaggle files: `results/kaggle/`.
- Synthetic sets (21): genB_amp{11,28,40}_s{7,101,2024}, nrA/nrB0/nrC0/nrC34 _s{7,101,2024}. Kaggle score files hold them; local 32 methods' synthetic scores are in the existing V6 bench pipeline (read scripts/v6/bench_v6.py to see where).
- Rescore file layout: `var[image][model][variant P0,P1,P2][metric dice,nsd1,nsd2,nsd3,nsd5,nsd_diag1]`, `grid_dice[image][model][18 thresholds]`, `grid_thresholds`, `thresholds`, `names`, `gt_empty`, `gt_fraction`. P0 = the study pipeline (postprocess_prob at the model's own threshold), P1 = threshold then the untrained methods' postprocess (5 px close/open + small components), P2 = threshold then small-component removal only. Families: pretrained_s0..2, modern2_s0..2, kvasir_nets, kvasir_unet, nnunet. Sets: 11 REFS + control + fuseg_r256 (FUSeg resampled to long side 256). Rescore files have NO exclusions applied: apply perimage_v6 EXCL yourself (same image order as perimage32).

## What failed in phase 2 (report, do not hide)
- WoundSeg (5th wound dataset) was not added: the size check stopped it (322 of 2686 masks at a different size from their image). No WoundSeg scores exist.
- Three trainings diverged (non-finite weights) and were not saved: manet_r18 seed 2, manet_r50 seeds 0 and 2.

## Decisions (fixed in advance; do not change them, add sensitivity analyses instead)
- Primary modern group = M19, architecture score = mean over its available seeds. Sensitivities: M11 (V6), M17, per-seed groups.
- Same agreement definitions, pair sets, exact relabeling tests, bootstrap designs and seeds as V6 (scripts/v6/*.py). Reuse V6 functions by import where possible instead of rewriting them; if you must copy logic, say so.
- Pre-specified reading for seeds: the claim "modern models disagree across wound datasets because the datasets differ, not because of training noise" is supported if same-dataset cross-seed agreement is high (> 0.7) while cross-dataset wound agreement stays low (about 0.1 to 0.3). If cross-seed agreement is as low as cross-dataset agreement, say so plainly: then modern models are statistically indistinguishable and the framing must change.

## Rules
- Accuracy over speed. State every definition. Print n (methods, images, pairs) next to every number.
- Every number you report must come from a script you saved under scripts/v7/ and an output under results/v7/ (json + a short markdown). No numbers typed by hand.
- Check each new number against an independent route where one exists (e.g. recompute one value by a second method, check seed-0 values equal V6).
- Plain language in markdown. No em dashes.
- Write only inside your assigned files. Do not commit to git.
