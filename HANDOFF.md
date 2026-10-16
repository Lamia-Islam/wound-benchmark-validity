# Handoff: synthetic-benchmark validity paper (JIIM submission)

Read this file first, then `manuscript_v6/BRIEF_MS_V6.md`. Everything below was true on 16 October 2026.

## 1. What the project is
A methods paper for the Journal of Imaging Informatics in Medicine (JIIM, Springer). Question: does a synthetic wound-segmentation benchmark rank methods the way clinical data do? Headline findings:
- **Composition effect (Simpson's paradox).** Pooling untrained classical methods (U, 18) with modern models (M19 = 18 ImageNet-pretrained architectures, each seed-averaged over 3 training seeds, plus nnU-Net) makes 4 wound datasets agree at 0.82, although they agree at 0.37 within U and 0.21 within M19. Stratified agreement equals the size-weighted within-group average exactly; the identity is derived in Methods.
- **Benchmark validity follows the pool.** Generator B agrees with the wound data at 0.65 over U+M19 but 0.21 over M19.
- **Modern-model rankings are weak and noisy.** Two seeds of the same architectures agree only 0.45 on the same wound dataset. The threshold protocol changes the level of agreement.
- **Practical results.** Selection loss is 0.047 vs 0.063 for a random pick. On AZH the pretrained architectures miss 27.9% of wounds vs 1.4% for nnU-Net.

## 2. Owner and house rules (strict)
- The owner is Lamia Islam. She wants direct, plain language and demands zero numerical mistakes.
- Text and figures:
  - No em dashes anywhere. Negative numbers use the minus sign U+2212.
  - British spelling ("relabelling").
  - Caveats go only in the Limitations section.
- Documents:
  - Black, white and grey only. Greys must be dark enough to read (#444 or darker for text).
  - Links muted blue #28598A, underlined.
- Content:
  - No placeholder or "fill in" text.
  - No author names or "draft" notes in the blinded manuscript.
- **Every number** in the manuscript and supplements comes by script from results JSON, never typed by hand. The pipeline:
  1. `rebuild/scripts/v7/facts_v6.py` writes `manuscript_v6/facts_v6.json`.
  2. `manuscript_v6/build_v6.js` substitutes `{{key}}` placeholders and throws on a missing key.
- Commits end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01KZ96kvnMzWaL6YRN9b9JVB
  ```

## 3. Map of the repository
- `rebuild/results/perimage32/*.json`: per-image Dice/NSD for the 32 local methods (U 18, F 14) on 12 clinical sets.
- `rebuild/results/kaggle/scores/{pretrained,nnunet,kvasir,sam}/`: GPU run 1 per-image scores. `rebuild/results/kaggle2/`: GPU run 2 (extra seeds, 8 new architectures, rescoring variants, SAM v2, logs).
- `rebuild/results/exclusions_v5.json`: images excluded before any analysis. Currently 8 FUSeg and 47 control (azh_train_slice).
- `rebuild/scripts/perimage_v7.py`: the shared loader (groups, seed averaging, exclusions, resampling units). Scripts that use it: `rebuild/scripts/v7/*.py`. Results go to `rebuild/results/v7/` (STATS_V7, BENCH_V7, PRACTICAL_V7, ROBUST_V7, ROBUST_POOLED_V7, BLIND_V7, audit/AUDIT_V7, FIGURE_VALUES_V6, md plus json). The md files define every quantity.
- V7 reproduces V6 bit for bit. A blind replication from raw files (`scripts/v7/blind_v7.py`) passed 41 of 41 headline numbers. An independent number audit (review R1) recomputed 289 of 292 manuscript values bit-identically and found no wrong number.
- `manuscript_v6/`:
  - Build: `build_v6.js` (with `lib.js` and `refs.json`) builds `manuscript_v6_blinded.docx`. Run `cd manuscript_v6 && node build_v6.js`; docx 9.x is installed globally in the old container, otherwise `npm i docx@9`.
  - Figures: `figures/` (made by `rebuild/scripts/v7/fig_v6.py`).
  - Supplements: `Online_Resource_1.docx` and `Online_Resource_2.xlsx`, made by `rebuild/scripts/v7/online_resources_v6.py`.
  - Reviews: `review/R1_numbers.md`, `R2_domain.md`, `R3_stats.md`, `R4_compliance.md`, `R5_references.md`.
- Not in git: the raw datasets (`rebuild/data/`, gitignored) and the GPU kit images (`/home/claude/kaggle_kit` in the old container). The raw wound data are public: clone https://github.com/uwm-bigdata/wound-segmentation (folder `data/`) into `rebuild/data/wound-segmentation/`. Text, table and figure work needs only the committed results JSON.

## 4. Status
- **Done:**
  - all analyses;
  - manuscript V6 draft (abstract 250 words, main text about 7,000 words, 69 references);
  - Figures 1 to 3;
  - Online Resources 1 and 2;
  - five independent reviews.
- **Not done:** everything in section 5.

## 5. Remaining work, in priority order

### P0. AZH test images overlap FUSeg photographs (new finding, must be handled before submission)
`rebuild/scripts/v7/azh_overlap_v7.py` writes `rebuild/results/v7/azh_overlap_v7.json`. It uses the Methods rule: SIFT, ratio 0.75, RANSAC similarity, at least 25 inliers, against all 810 FUSeg train and 200 validation frames.

**Validation.** The script finds all 47 recorded control exclusions again. It also finds **24 further control images** that are crops of FUSeg frames and are not in `exclusions_v5.json` (`validation_control.new_hits_not_recorded`).

**AZH test (150 images).** 61 match a FUSeg frame: 49 FUSeg-train frames and 12 FUSeg-validation frames. The 12 validation frames are our FUSeg reference set.

**Which train frames were actually used.** Map frames with `kaggle_kit/manifest.json` → `fuseg_train.files`, where training index i is the i-th file in that list. Modern models and F used indices 0 to 392, thresholds 400 to 549.
- Only **4** AZH test images are crops of frames used in training.
- 0 are crops of threshold frames.
- 45 are crops of FUSeg train frames we never used.

The index-to-file list is in `rebuild/results/v7/fuseg_train_index.json` (from the kit manifest).

**Required:**
1. **Decide and document one exclusion rule, applied consistently.** The rule used so far: remove evaluation images that are copies or crops of FUSeg photographs. Recommended: create `exclusions_v6.json` that adds (a) the 24 extra control crops and (b) the AZH test images that are crops of FUSeg photographs used anywhere: the 4 training frames plus the 12 crops of FUSeg reference images (16). Keep AZH crops of unused FUSeg frames, since they are not leakage.
2. **Point `perimage_v7.py` at the new exclusions and rerun everything.** Order:
   1. stats_v7
   2. bench_v7
   3. practical_v7
   4. robust_v7
   5. robust_pooled_v7
   6. blind_v7, with its reported values updated
   7. fig_v6
   8. online_resources_v6
   9. facts_v6
   10. build_v6

   The scripts are deterministic. Some scripts assert that V6 values are reproduced, so they will fail under new exclusions; turn those checks into "V6 under the old exclusions" comparisons rather than deleting them. Update the units count for AZH and the control.
3. **Alternatively, if rerunning everything is not possible,** keep the current exclusions and report a sensitivity analysis (the AZH-dependent headline numbers without the 16 or 61 images, and the control without the 24). Either way, the Methods must state the check and its result. Also restore the sentence that AZH test images were checked against FUSeg photographs.
4. The SIFT check took 428 s on 2 CPUs.

### P1. Text revisions from the reviews (read the five review files; they give exact quotes and replacement wording)
- **Must fix:**
  - **Overclaims.** Remove the "as good a proxy and no better" equivalence and "ceiling" claims (R3 M6, R2 #5); the difference interval is −0.41 to 0.30. Remove "an oracle threshold lowers agreement" (R3 M7): the change is −0.089 (−0.279 to 0.083), and the oracle is capped by the grid. "The gap comes from the nine original architectures" becomes "was larger among the nine original (0.49) than among the seven new (0.05)", with hierarchical intervals from `stats_v7.json` `seeds/dice|P9|N7/summary/gap.../hierarchical` (R3 M3, R1 #3). Also add that sharing test images inflates same-dataset cross-seed agreement and that the hierarchical gap interval is not conservative (R3 M4).
  - **"Confirmatory".** Say "primary" (R3 M1, R2 #9). The composition effect is judged by bootstrap intervals, not the relabelling test. The 0.7 seed rule was written after training and scoring but before the seed analysis; describe it exactly that way and report its second branch (R3 M2).
  - **Selection.** Report that with single-seed models the other-dataset chooser is about as bad as random: 0.058, 0.076, 0.053 against 0.060, 0.084, 0.059, from `practical_v7.json` (R3 M5). Add the deployment-relevant result that choosing on the in-domain FUSeg set loses 0.100 elsewhere, worse than random 0.063, from the FUSeg row of the loss matrix (R2 #1). Add the SegFormer-B2 vs nnU-Net trade-off on AZH: mean Dice 0.713 vs 0.631, but misses 11.9% vs 1.4% (R2 #7).
  - **AZH misses.** Present them as a comparison of two pipelines: letterbox 256 px vs nnU-Net native spacing (R2 #3). See PRACTICAL_V6.md section 5: the magnification is about 1.9 times for 10 models. Any number for the 18 architectures must be computed by script first.
  - **Floor rule.** Apply it evenly: F is near the floor on 6 of 7 non-wound sets, and U on FUSeg is at 0.06 (R2 #6).
  - **R1 wording items 2 to 20.** Fix the abstract "agreed less (0.10)" (that value is at different seeds). Fix "span 0.07 to 0.20" (it is the width of the range). The texture effect is "only in pooled sets" but not in U+K (0.05, −0.04 to 0.14). Fix the Table 5 interval types, the single-seed group membership, and the "Tables S2 and S3" citation (should be S2 only).
  - **Split-half reliability.** Over 65 pooled methods it is inflated by pooling; report within-group values (R3 m2).
- **Compliance (R4):**
  - **Blinding.** In the blinded copy, reference 60 (the author's own preprint, `islam2026`) must read "Reference removed for double-blind review"; keep it in an unblinded build.
  - Add page numbers (Footer with PageNumber.CURRENT).
  - Write the stratification formula with real subscripts.
  - Trim the abstract to about 235 words.
  - **AI-use sentence in Methods,** which JIIM requires. Suggested: "Large language model assistants were used to write analysis code and to draft and edit the text; all code, numbers and text were checked by the authors." Flag it to the owner.
  - **Data availability.** Mention the synthetic generators and code ("available from the corresponding author on reasonable request" unless the owner gives a repository URL; flag it).
- **References (R5):**
  - Reword the sentence citing lucho2024 and benfatah2026, which compare deep models only; zhang2022 is a wound survey.
  - Drop zhang2026 from the "agreement depends on which models" claims.
  - Move urooj2026 next to the metric/protocol citations.
  - Switch nikolov2018 to nikolov2021.
  - Metadata fixes listed in R5: teney2023, prakash2019, tan2019, kievit2013 and tushar2026a author lists.
  - Optional: JIIM style (Index Medicus abbreviations, all authors).
- **Title.** R2 suggests "Validating a synthetic segmentation benchmark by rank correlation: how the methods compared, the clinical reference and training noise change the answer in wound imaging". Choose between this and the current title; the current title generalises from one generator.

### P2. Figures and Online Resources
- **Figures.** Rerun `nice -n 10 python3 rebuild/scripts/v7/fig_v6.py` from `rebuild/`. The Fig. 3a label clipping fix is already in the script but has not been rendered. Then view `manuscript_v6/figures/Fig3_modern_noise.png`. The figures already meet the spec: Liberation Sans (Arial-metric; no Arial on the machine), 8 pt, Fig. 2 at 129 mm, 1200 dpi TIFF, EPS/PDF, `figures/submission/Fig1-3.*`.
- **Online Resource fixes, not yet done:**
  - Table S9a and the OR2 README must say "8 FUSeg validation images are byte-identical to FUSeg training images (7) or to another validation image (1)".
  - Use "validation", never "test", for FUSeg.
  - Remove internal labels and file paths (M (V6), P10_s0, C16, P9, N7, M11s, ALL57, K11, rebuild/..., *.json) or define them once in a notation paragraph.
  - British "relabelling".
  - Black-pixel share basis: all kept images (AZH 85.2%, FUSeg 26.3%).
  - OR2 negative-number format showing U+2212.
  - Journal name in the headers.
  - Fig. S1 numbers at least 6 pt, RGB TIFF, Liberation Sans.
  - Export `Online_Resource_1.pdf`.
  - Add Table S9e with the AZH overlap check (P0).

### P3. Final verification and delivery
1. Rebuild facts and docx. Check the build output: no missing or unused-key surprises, abstract at most 250 words, no em dashes.
2. Rerun an independent number audit like R1: every number in the docx traced to JSON, and its meaning checked.
3. Rerun the blind replication.
4. Render the PDF and look at every page.
5. Commit.
6. Deliver to the owner:
   - `manuscript_v6_blinded.docx`, plus an unblinded build with reference 60 restored;
   - `Online_Resource_1.docx` and `.pdf`;
   - `Online_Resource_2.xlsx`;
   - `figures/submission/*`;
   - a one-page list of what she must supply: title page (names, affiliations, ORCID, corresponding author), funding, competing interests, author contributions, acknowledgements, the AI-use wording, the cover letter (which mentions reference 60 is her own preprint), and the code/data availability decision.

## 6. Decisions already made (do not reopen without new evidence)
- No third GPU run.
- WoundSeg is excluded: 322 of 2,686 masks are size-mismatched.
- The three diverged MA-Net trainings are not imputed.
- SAM and MedSAM are oracle-prompted and never pooled.
- The main protocol uses each model's own tuned threshold; nnU-Net is fixed at 0.5.
- Do not change the framing in BRIEF_MS_V6.md except as the reviews require. The composition effect is strong for U+M19, weaker and not distinguishable from zero for U+F, and absent in the K mirror. The draft already says this; keep it.

## 7. Practical notes
- The old container had 2 CPUs; run heavy scripts with `nice -n 10`. The full V7 pipeline runs in under a minute except the bootstraps (about 15 s each) and the SIFT check (7 min).
- LibreOffice (`soffice --headless --convert-to pdf`) and `pdftoppm` were used for visual checks.
- `rebuild/scripts/v6/stats_v6.py` runs at import and rewrites results/v6; do not import it.
