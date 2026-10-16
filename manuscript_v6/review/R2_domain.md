# Referee report, Reviewer 2 (domain: medical image segmentation, benchmarking, wound imaging)

Manuscript: "Method pool, clinical reference and training seed shape the measured validity of synthetic segmentation benchmarks: a study in wound imaging" (version 6, blinded)

Journal: Journal of Imaging Informatics in Medicine

Material read: the manuscript text and rendered PDF (Figs. 1 to 3 checked visually), Online Resource 1 (Tables S1 to S9, Fig. S1), and, for the internal section only, the authors' results files (rebuild/results/v7, plus PRACTICAL_V6.md and exclusions_v5.json for history) and the earlier response to review (manuscript_v4).

---

## Part A. Report to the editor and authors

### 1. Summary

The authors ask how much the usual evidence for a synthetic segmentation benchmark, a Spearman correlation of per-method mean Dice between the synthetic set and a clinical set, depends on which methods are ranked, which clinical dataset is the reference, and how much training noise the models carry. They score 65 methods (18 classical, 14 trained from scratch on wound photographs, the same 14 trained on colonoscopy, and 18 ImageNet-pretrained architectures at three seeds plus one nnU-Net) on 11 public datasets in four modalities and on parametric synthetic wound images. Pooling classical with modern methods raises agreement between the four wound datasets from 0.37 and 0.21 within the groups to 0.82, and the same synthetic benchmark scores 0.65 over that pool but 0.21 among the modern models alone, which is also how weakly the wound datasets agree with each other. Among modern models, rankings are close and noisy (two seeds of the same architectures agree at 0.45 on the same data), choosing a model on one wound dataset helps only modestly on another, and on one dataset the pretrained models miss far more wounds than nnU-Net.

### 2. Recommendation

**Major revision.**

The work is careful, unusually transparent about its own failures (the pre-specified seed criterion that was not met, the diverged trainings, the excluded fifth dataset), and its practical message is useful. However, the most general claims (title, abstract, conclusions) rest on one parametric synthetic generator of the authors' own design, four wound datasets from three sources, and a comparison between "pretrained architectures" and nnU-Net that is confounded by the authors' own preprocessing pipeline. The paper is also written for statisticians rather than for JIIM's imaging informatics readership.

Honest probability of eventual acceptance at JIIM after one revision: **about 35%** if submitted as it stands (a second reviewer could well see the composition effect as a textbook between-group artefact and recommend rejection on significance). I would put it at **about 50 to 55%** if the authors make the text changes listed in the internal section before submission, and somewhat higher if they also run the two cheap new checks marked "high" below (AZH overlap check and the AZH rescaling test).

### 3. Major concerns, ranked

Each concern ends with **Already answered?**, saying whether the existing analyses (rebuild/results/v7 and Online Resource 1) already answer it, and if not, what new work is needed and how important it is.

#### M1. Significance for JIIM readers is under-argued, and the strongest practical result is buried

The central composition result is, as the authors themselves say, Simpson's paradox: if one group of methods beats another on every wound dataset, any pooled rank correlation between wound datasets must be high. The identity in "Stratified agreement and why it equals the within-group average" is correct (I checked the derivation: with percentiles (r − 0.5)/n_g the within-group variance is (1 − 1/n_g²)/12 and the weights n_g − 1/n_g follow), but it is a short algebraic consequence of equal group means, not a substantive finding. A reader from JIIM will ask two things the manuscript does not answer:

- **Is pooled validation actually practised in medical imaging?** The two citations given for the usual evidence ([12, 13]) are person re-identification and face recognition. The wound papers cited ([25, 26]) compare classical and deep methods but are not synthetic-benchmark validations. Without two or three medical imaging papers that validated a synthetic or proxy test set over a mixed pool, the composition result risks reading as a straw man.
- **What should a hospital, vendor or regulator do differently?** The paper's most deployment-relevant numbers are in the supplement, not the main text. Table S7a shows that choosing the model that is best on the in-distribution test set (FUSeg, the same clinic as the training data) loses 0.100 Dice on average on the other wound datasets, which is worse than a random pick (0.063). That is exactly the situation of a hospital accepting a vendor's internal test result. Similarly, Table S4h shows that among the 16 seed-averaged architectures the in-distribution set FUSeg agrees with the two external-source sets at −0.21 (Medetec) and −0.01 (CO2Wounds-V2), and with AZH, from the same clinic, at only 0.18, while the two external sets Medetec and CO2Wounds-V2 agree at 0.68. Internal versus external validity is a much more familiar and important question for JIIM readers than the modality structure of untrained methods.

**Already answered?** Mostly yes for the second point: the selection matrix (Table S7a) and the per-pair values (Table S4h, Fig. S1, Online Resource 2) exist. Add them to the main text and lead with them. The first point needs a short targeted literature search (no computation), and it is important: without it the significance argument is weak.

#### M2. The title and conclusions generalise from one parametric generator of the authors' own design

All benchmark-validity numbers in the main text come from one generator family (B) at one texture amplitude, 45 images per synthetic seed, released with the authors' earlier system [60]. Generators A and C appear only in Table S5f, and the comparisons between families "changed with the pool and the references". Learned generators (diffusion, image editing as in RadEdit [4]) and published in-silico pipelines are what current readers mean by synthetic test data. The general claim in the title ("synthetic segmentation benchmarks") is therefore stronger than the evidence. In fairness, the composition result is a mathematical property and will transfer to any benchmark on which groups differ in level, but the claims about how well a synthetic benchmark orders modern models ("validated no better than one clinical dataset predicts another") are specific to generator B.

**Already answered?** Partly. The pool dependence is general by construction and the text can say so. The modern-model claims need either (a) new work: one learned or externally published synthetic wound set scored with the existing M19 models (moderate effort, high importance if the title stays general), or (b) narrowing the title, abstract and conclusions to "a parametric synthetic benchmark". Option (b) is acceptable for JIIM.

#### M3. The "pretrained architectures versus nnU-Net" failure mode is a pipeline comparison, not an architecture comparison

The abstract headline "on one dataset the pretrained architectures missed 27.9% of wounds against 1.4% for nnU-Net" compares two training and inference pipelines that differ in many ways at once:

- input handling: the 18 pretrained models letterbox every image to 256 pixels; nnU-Net works at native resolution with its own per-image intensity normalisation and patch-based inference;
- augmentation: nnU-Net's default trainer applies scaling, rotation, elastic and intensity augmentation; the pretrained models get flips and 90-degree rotations only;
- training length and threshold: 40 epochs with a FUSeg-tuned threshold versus 100 nnU-Net epochs with a fixed 0.5 threshold.

AZH images are 224-pixel wound crops pasted on a black canvas. The authors' own earlier analysis (PRACTICAL_V6, not in the manuscript) used SIFT matches to show that one AZH pixel covers 1.20 FUSeg pixels, so the letterboxed models see AZH tissue at about 1.9 times their training magnification while nnU-Net sees it at about 0.83 times; in the model's input frame the median AZH lesion is 1,298 pixels against 503 for FUSeg. The same earlier analysis found that when the pretrained models do detect an AZH wound they score better than nnU-Net (0.79 against 0.69 Dice). That points to a scale mismatch introduced by the preprocessing, which the manuscript reports only as a correlation with "black padding". A wound imaging reader will immediately suspect this, and it matters beyond the miss rates: if AZH rankings among modern models are mostly rankings of scale robustness, part of the low wound-pair agreement among modern models is created by the evaluation pipeline rather than by the datasets.

**Already answered?** Partly. The scale measurement exists and should be reported, and the comparison should be described as pipeline against pipeline throughout (text change). A direct test is new but cheap and inference-only: rescale or crop the AZH canvas to FUSeg magnification before letterboxing and rescore misses for the existing 51 models. Importance: high, because the number is in the abstract.

#### M4. Data provenance and independence of the wound datasets are not fully documented

Wound photography is represented by four datasets from three sources, two of them crops of one clinic's photographs, so every wound conclusion rests on five independent-source pairs. Three points need attention:

- **AZH test against FUSeg training.** The authors found by SIFT matching that 47 of 150 control images (AZH training slice) are crops of FUSeg photographs, and removed them. The manuscript does not say whether the 150 AZH test images were checked against FUSeg training photographs in the same way. FUSeg and the AZH dataset of Wang et al. [33] come from the same centre and, as far as I know, overlapping patients. If AZH test crops come from FUSeg training photographs, AZH is not an external test set for FUSeg-trained models.
- **Medetec masks.** Medetec [30] is a public photograph archive without official masks. The masks used appear to come from the foot ulcer subset in the repository of the group that built FUSeg and AZH. If so, three of the four wound datasets share an annotation team and protocol, and CO2Wounds-V2 is the only independent annotation. Also state what the "four variants" of each Medetec photograph are (flips, rotations, crops).
- **Annotation protocols.** Differences in what counts as wound (wound bed only, slough, periwound) are a likely cause of cross-dataset disagreement and are discussed only through one citation [69]. A short table of who annotated each wound dataset and with what rule would help readers more than the modality analysis does.

**Already answered?** No for the AZH test check: new, cheap work (the SIFT pipeline already exists), importance high for data integrity. The other two are text additions drawn from the dataset papers.

#### M5. Some conclusions go beyond the data

- **Equivalence claimed from an interval that spans ±0.4.** "A benchmark that agrees with them at 0.21 is as good a proxy as another clinical dataset and no better" (Discussion) and the abstract's "at 0.21 among the modern models, the level at which the wound datasets agreed with each other" rest on a difference of 0.00 with interval −0.41 to 0.30. The data cannot distinguish "equally good" from "much worse" or "somewhat better".
- **"Ceiling" is the wrong concept.** The conclusion "a synthetic benchmark can then be validated no better than one clinical dataset predicts another" is not a bound. A benchmark can agree with a reference more than other clinical datasets do, and the authors' own Table S5b shows this (untrained methods with CO2Wounds-V2: benchmark 0.80 against 0.28 for the other wound datasets, difference 0.52, interval 0.13 to 0.92; U+K with AZH: 0.70 against −0.03). The real upper limit is the reliability of each ranking (split-half and seed reliability), which the paper measures in Tables S4g to S4i.
- **"Inflates agreement within one modality."** The colonoscopy mirror view (Table S6h) shows no inflation for any pooled set (U+M19 gap −0.04, interval −0.17 to 0.16; U+K11 0.07, −0.004 to 0.17). Inflation occurs within the modality where the groups differ strongly in level, which is the wound domain here.
- **The oracle threshold does not "remove calibration differences".** Table S6e shows that on Medetec and CO2Wounds-V2 92% and 96% of the models have their oracle threshold at the lowest grid value (0.05), as do 96 to 100% on the non-wound sets. The oracle is censored by the grid, so calibration differences remain.

**Already answered?** Yes. All four are fixed by rewording; the numbers needed are in Tables S5b, S6e and S6h.

#### M6. The floor rule is applied to the modern group but not to others

The authors rightly refuse to interpret the modern group's modality difference because its mean Dice is at most 0.07 on every non-wound dataset. But the same rule is not applied elsewhere:

- the from-scratch group is below 0.1 on six of the seven non-wound datasets, yet it is counted as showing modality structure (0.47, p = 0.0045), and its colonoscopy agreement of 0.88 is listed next to the untrained and Kvasir-trained values as evidence that colonoscopy datasets agree;
- the untrained group's mean Dice on FUSeg is 0.06 and on ETIS-Larib 0.07 (floor table, STATS_V7 section 9), so the untrained wound-pair agreement of 0.37, which is the baseline of the main composition comparison, includes rankings of floor-level scores on FUSeg; the Kvasir-trained group is at 0.08 on FUSeg.

Either floor-level rankings can be consistent and meaningful (the from-scratch colonoscopy value of 0.88 suggests they can, for example through systematic false-positive behaviour), in which case the modern group's exclusion should rest on its seed dependence alone, or they cannot, in which case the from-scratch evidence and the FUSeg-containing untrained pairs need the same caveat.

**Already answered?** Yes. The floor table exists. State one rule and apply it to every group; report untrained wound-pair agreement without FUSeg as a sensitivity check if the leave-one-dataset-out output allows it (Table S3a gives the modality difference only, so a wound-pair version would be a small new computation, low importance).

#### M7. Rank correlation is a weak summary when models are this close

Among the 19 modern models the method means on a wound dataset span 0.07 to 0.20 Dice and neighbouring models differ by 0.001 to 0.007. A Spearman correlation over such a set is dominated by differences nobody would act on. JIIM readers need an agreement measure tied to decisions. The authors already have most of what is needed:

- selection loss (Table S7a), the share of picks within 0.01 or 0.02 Dice of the best (Table S7b);
- per-dataset "indistinguishable from the top" sets from paired image-level intervals (PRACTICAL_V7: on FUSeg three other models are indistinguishable from the top model; on AZH none).

A further point: by mean Dice SegFormer (MiT-B2) is best on AZH (0.713 against 0.631 for nnU-Net), yet it misses 11.9% of wounds against 1.4%. Which model is "best" for a clinic depends on whether a missed wound or a rougher boundary matters more. That tension is the most clinically interesting result in the paper and is not discussed.

**Already answered?** Mostly yes; make selection loss and the miss rate the primary practical outcomes for modern models and keep Spearman as secondary. A new analysis restricting agreement to model pairs whose difference is distinguishable on both datasets would be nice but is optional (medium-low importance). A clinically meaningful wound metric such as relative area error would strengthen the JIIM case (new work, medium importance, and only possible if predicted masks or areas were stored).

#### M8. Design weaknesses that limit the modern-model conclusions

- **One nnU-Net training, 100 epochs, fixed threshold 0.5.** nnU-Net is the synthetic benchmark's pick in most settings and the comparator for the miss analysis, yet it is the one modern method without seed replicates and the only one whose threshold is not tuned. Two more nnU-Net seeds would cost little (new work, medium importance).
- **Only 392 of FUSeg's roughly 810 training images were used**, with no reason given. Reviewers will ask whether the modern models are under-trained and whether that compresses their differences.
- **Seed and session are confounded for the original architectures**, and the gap result (0.35) comes entirely from those nine (0.49), while the seven new architectures, trained in one session, show none (0.05). The text reports this honestly; it should also say plainly that the one result supporting "datasets really differ" rests on the architectures where seed and session cannot be separated.
- **Seed averaging is underpowered as a test.** "Averaging three seeds did not raise agreement across datasets" (0.18 and 0.18, difference interval −0.15 to 0.18) cannot distinguish no effect from the modest rise that Spearman-Brown predicts. Do not use it as evidence either way.
- **Generality beyond wounds.** The paper argues that wound photography is special ("Why wound photography differs"), but the pretrained group was trained only on wounds, so there is no comparison of how pretrained models trained on colonoscopy rank across the five colonoscopy datasets. Training the pretrained group on Kvasir-SEG would test whether the low within-group agreement is wound-specific (new work, medium importance; needed only if the authors keep the "why wound photography differs" argument).

**Already answered?** Partly; the confounding and power points need text only. nnU-Net seeds and a colonoscopy-trained pretrained group are new work.

#### M9. "Confirmatory" status is asserted but not documented

The Methods call the composition effect and the modality structure "the confirmatory analyses of the study design" and say both "are tested with the exact relabeling test". The relabeling test tests the modality difference only; the composition effect is assessed with bootstrap intervals. More importantly, after six manuscript versions a reader will want evidence that these analyses were fixed in advance: a dated analysis plan, a repository commit or a registration. The same applies to the seed "reading rule fixed before the new trainings were scored".

**Already answered?** Text only: either cite a dated record (commit hash or timestamped plan in Online Resource 1) or describe these as "primary analyses" rather than confirmatory.

#### M10. The main message is hard to find, and the title does not match the paper's strongest content

The paper has three contributions, a modality analysis, a threshold analysis, a seed analysis, a selection analysis and a failure-mode analysis, with about 8,200 words including tables, 14 named method sets in the supplement (U, F, K, M, M11s, M19, N8s, P9, N7, C16, P10s, U+F, U+K, U+M19 and more), and roughly 15 numbers in the 250-word abstract. JIIM readers are not statisticians. The strongest and most general results concern agreement between clinical datasets and the noise in ranking modern models; the synthetic benchmark part is the weakest (one generator, intervals including zero). The title puts the weakest part first and is 21 words long.

**Already answered?** Text only. Suggested titles (pick one):
- "Validating a synthetic segmentation benchmark by rank correlation: how the methods compared, the clinical reference and training noise change the answer in wound imaging"
- "Why rank agreement between segmentation test sets depends on the methods compared: lessons for synthetic benchmarks from wound imaging"

### 4. Minor concerns

Each item ends with what to change; all are text or layout unless marked.

1. **Abstract.** Cut to the four or five numbers that carry the message (0.82 against 0.37 and 0.21; 0.65 against 0.21; 0.45 seed agreement; selection loss including the in-domain pick; misses). Say in words what a Spearman correlation of 0.21 means ("barely better than unrelated orderings").
2. **Abstract wording.** "18 ImageNet-pretrained architectures, each trained with three random seeds" is not accurate for the two MA-Net architectures (2 and 1 usable seeds). Write "trained with up to three random seeds".
3. **Terminology.** "Modern" is ambiguous: a ResNet-18 U-Net is modern in M19 but not in F. Use "pretrained" for the 18 and name nnU-Net separately. "Untrained" reads oddly for classical algorithms; "classical (no learned parameters)" is clearer. "Validity", "agreement", "measured validity" and "benchmark agreement" are used interchangeably; define once and stick to one.
4. **Fig. 1.** Category labels "domain with domain", "domain with other", "other, same modality" are opaque. Label each panel in its own terms ("wound with wound", "wound with non-wound", and so on). Consider showing only panel a in the main text.
5. **Fig. 3a.** The six open circles for "each of 6 pairs" are unlabelled; label them or drop them. The per-pair spread (one pair near 0.5, the rest near 0) is itself informative and deserves a sentence.
6. **Table 3** has ten numeric columns and mixes agreement levels with tests; split it or move the NSD and p columns to the supplement. The orphan note "Within modality: different sources." under the table should be a proper table note.
7. **Table 5** mixes seed reliability, selection losses and miss rates. Split into a selection-and-misses table (the practical one) and leave seed reliability to the text or supplement.
8. **No image figure.** An imaging journal paper with no images of the data is unusual. Add one figure with two examples from each wound dataset (showing AZH's black canvas and FUSeg's small lesions) and examples of generator B at amplitudes 11 and 40. All exist already.
9. **Ordering.** The modality section ("Modality structure is clear within the untrained and Kvasir-trained groups") is of least interest to JIIM readers and depends on the floor issue (M6). Move it after the modern-model results or mostly to the supplement.
10. **Two kinds of intervals.** Method-bootstrap and hierarchical intervals are both quoted in the text. Pick one for the main text (hierarchical where available) and put the other in the supplement.
11. **Split-half reliability over all 65 methods (0.97 to 0.99)** is inflated by the very between-group effect the paper documents. Report only within-group reliabilities.
12. **"The new architectures miss less than the original ten (24.9% against 30.3%)"**: the intervals overlap (18.4 to 31.1 against 23.3 to 37.0). Say "somewhat less" or give the paired difference.
13. **NSD sentence in "Why wound photography differs"**: the rise from 0.37 to 0.62 under NSD is shown for untrained methods only; under NSD the from-scratch group rises (0.16 to 0.31) but the Kvasir-trained group falls (0.36 to 0.28). Report all three or soften "consistent with".
14. **Recommendation 7** ("report stratified agreement") contradicts the paper's own conclusion that stratified agreement adds nothing beyond the within-group values. Recommend reporting the within-group agreements directly.
15. **Introduction**: "two datasets cannot agree on an order that training noise does not preserve" is too strong; noise lowers the expected agreement but does not forbid agreement.
16. **References.** Several key citations are 2026 arXiv preprints (refs 3, 11, 13, 17, 26). Prefer peer-reviewed versions where they exist (for NSD [61], cite the journal version by Nikolov et al., J Med Internet Res 2021). Make sure the self-citation [60] stays in the third person, as now, to keep the blinding.
17. **Medetec citation**: cite the source of the masks, not only the image archive [30].
18. **Methods detail**: say why images 393 to 399 of the FUSeg training set are unused, and that the held-out Dice used for threshold tuning is optimistic (Table S1 says so; the main text does not).
19. **Length.** About a quarter of the Results could move to the supplement without loss, mainly the per-seed modality values, the generator-family remarks and the leave-one-out ranges.

### 5. Summary table: existing data or new work

| Concern | Existing data answers it? | Action | Importance |
|---|---|---|---|
| M1 significance and buried practical result | Yes (Tables S7a, S4h, Fig. S1) plus literature search | Lead with in-domain versus external selection; cite medical examples of pooled validation | High |
| M2 one parametric generator | Partly | Narrow claims (text) or score one learned or published synthetic set (new) | High |
| M3 AZH misses are a pipeline effect | Partly (V6 scale analysis) | Report scale finding, reword as pipeline comparison; rescale-AZH inference test (new, cheap) | High |
| M4 dataset provenance and AZH overlap | No for AZH test overlap | Run existing SIFT check on AZH test against FUSeg training (new, cheap); document Medetec masks and protocols (text) | High |
| M5 conclusions beyond data | Yes | Reword four statements | High |
| M6 floor rule inconsistent | Yes | One rule for all groups | Medium |
| M7 Spearman over close models | Mostly yes | Make selection loss and misses primary; discuss Dice against misses | Medium |
| M8 design (nnU-Net once, 392 images, seed-session confound, averaging power, colonoscopy-trained pretrained group) | Partly | Text for confound and power; nnU-Net seeds and colonoscopy training are new | Medium |
| M9 confirmatory status | Text | Cite dated plan or relabel as primary | Medium |
| M10 message and title | Text | Retitle, simplify, cut | High |

---

## Part B. Internal section for the authors (not for submission)

### B1. The five highest-value text changes before submission

All suggested wording is plain English and contains no em dashes.

**Change 1. Retitle, and rebuild the abstract around what a clinic or vendor should do.**

Why: this addresses M1, M2 and M10 at once, and costs no computation. The in-domain selection result from Table S7a is the single most useful number for JIIM readers and is currently not in the main text.

Suggested title: "Validating a synthetic segmentation benchmark by rank correlation: how the methods compared, the clinical reference and training noise change the answer in wound imaging"

Suggested replacement for the last four sentences of the abstract:

"Pooling classical and modern methods made the four wound datasets appear to agree at 0.82, although they agreed at only 0.37 among the classical and 0.21 among the modern methods. The same synthetic benchmark therefore looked valid over the pooled methods (0.65) but agreed with the wound datasets at only 0.21 among the modern models, and the data could not tell whether it was better or worse than another clinical dataset. Among modern models, rankings were close and noisy: two training runs of the same architectures ranked them at only 0.45 alike on the same images. Choosing the best model on the test set from the training clinic lost 0.100 Dice on average on the other wound datasets, more than a random choice (0.063). A validity figure for a synthetic benchmark should therefore be reported within the group of methods it is meant to compare, against several clinical datasets, and next to how well those datasets agree with each other."

(Before using it, check that this version stays within 250 words with the opening sentences you keep.)

**Change 2. Reframe the AZH failure mode as a pipeline effect and report the scale finding you already have.**

Why: addresses M3. The current abstract sentence will draw the first objection from any wound imaging reviewer, and the explanation is already in PRACTICAL_V6.

Replace the abstract clause "and on one dataset the pretrained architectures missed 27.9% of wounds against 1.4% for nnU-Net" with:

"and on one dataset the pretrained models, trained and run on images resized to 256 pixels, missed 27.9% of wounds against 1.4% for nnU-Net, which works at native resolution."

Add to the Results paragraph that begins "Mean Dice hides a failure mode", after the padding sentences:

"Feature matching between AZH crops and the FUSeg photographs they were cut from shows that one AZH pixel covers 1.20 FUSeg pixels. After resizing to 256 pixels, the pretrained models therefore see AZH tissue at about 1.9 times the magnification they were trained on, while nnU-Net, which works on native pixels, sees it at about 0.83 times. When the pretrained models do detect an AZH wound, they outline it better than nnU-Net does. The misses are therefore more likely to come from the difference in input scale between the two pipelines than from the architectures themselves."

Before adding the last two sentences, recompute or confirm the 1.20, 1.9, 0.83 and the "detected Dice" values (0.79 against 0.69 in V6, 10 models) for the 18-architecture V7 set, because the house rule is that every number comes from a results file. In Limitations, add: "The pretrained models and nnU-Net differ in input resizing, augmentation, training length and threshold, so their comparison is between two pipelines, not between architectures."

**Change 3. Remove the equivalence and "ceiling" claims.**

Why: addresses M5. These are the statements a statistically minded reviewer will call wrong.

Replace in the Discussion "Wound datasets agree with each other at 0.21 among modern models, so a benchmark that agrees with them at 0.21 is as good a proxy as another clinical dataset and no better." with:

"Among modern models the wound datasets agree with each other at only 0.21, and the benchmark agrees with them at the same level. The interval of the difference runs from −0.41 to 0.30, so our data cannot tell whether the benchmark is a better or a worse proxy than another clinical dataset."

Replace in the Conclusions "A synthetic benchmark can then be validated no better than one clinical dataset predicts another." with:

"In that situation a high validity figure is hard to obtain and hard to interpret, because the clinical datasets used as references do not agree with each other."

Replace in "Relation to prior work" "to show that it inflates agreement within one modality as well as creating disagreement across modalities" with:

"to show that it inflates agreement between datasets of the modality on which the groups differ most in level, here wound photography, as well as creating disagreement across modalities"

**Change 4. Fix three method statements and apply one floor rule.**

Why: addresses M5, M6 and M9. These are small edits that remove factual errors.

(a) In "Confirmatory and exploratory analyses", replace "and are tested with the exact relabeling test" with:

"The modality difference is tested with the exact relabeling test, and the composition effect with bootstrap intervals for the difference between pooled and within-group agreement."

If you cannot point to a dated record of the analysis plan, also replace "confirmatory" with "primary" throughout.

(b) In the threshold paragraph, replace "With a per-dataset oracle threshold for every model, which removes calibration differences, agreement falls to 0.12 (−0.13 to 0.40)." with:

"With a per-dataset oracle threshold for every model, chosen on the evaluation images, agreement falls to 0.12 (−0.13 to 0.40). On Medetec and CO2Wounds-V2 most models reach the lowest threshold on the grid (0.05), so this oracle reduces but does not remove differences in calibration."

(c) Apply one floor rule. In "Modality structure is clear within the untrained and Kvasir-trained groups", replace "The from-scratch group shows the same structure (0.47, p = 0.0045), although its group mean Dice is below 0.1 on six of the seven non-wound datasets." with:

"The from-scratch group shows a similar difference (0.47, p = 0.0045), but its mean Dice is below 0.1 on six of the seven non-wound datasets, so, as for the modern group, its rankings there order near-zero scores and we do not count it as evidence."

and change "Colonoscopy datasets from three sources agree at 0.86, 0.88 and 0.83 within the three groups" to report the untrained and Kvasir-trained values only (0.86 and 0.83). Add to Limitations: "The untrained methods score near zero Dice on FUSeg (group mean 0.06), so their wound-pair agreement includes rankings of near-zero scores on that dataset."

**Change 5. Put the deployment results in the main text: in-domain selection, internal against external pairs, and Dice against misses.**

Why: addresses M1 and M7 and gives JIIM readers the practical content they expect. All numbers exist (Tables S7a, S4h, S7d).

Add to "Consequences for choosing a modern model", after the sentence on the 0.047 loss:

"Choosing on FUSeg, the test set from the clinic that supplied the training images, was the worst strategy: the model ranked first there lost 0.100 Dice on average on the other three wound datasets, including AZH from the same clinic, more than a random choice (0.063). The reason is visible in the agreement between datasets. Among the 16 architectures with three seeds, FUSeg agreed with Medetec at −0.21 and with CO2Wounds-V2 at −0.01, and even with AZH, from the same clinic, at only 0.18, while the two external datasets Medetec and CO2Wounds-V2 agreed with each other at 0.68. A good result on the internal test set therefore said little about performance elsewhere."

Add after the AZH miss paragraph:

"Mean Dice and misses can point to different models. On AZH, SegFormer (MiT-B2) has the highest mean Dice (0.71 against 0.63 for nnU-Net) but misses 11.9% of wounds, against 1.4% for nnU-Net. Which model is better depends on whether a missed wound or a less exact outline costs more in the clinical task."

Add a recommendation: "Do not select a model on a test set from the training clinic alone; check the choice on at least one external dataset."

### B2. Sentences I consider factually or logically wrong

| Location | Sentence (abridged) | Problem |
|---|---|---|
| Conclusions, and echoed in the abstract | "A synthetic benchmark can then be validated no better than one clinical dataset predicts another." | Not a bound. A benchmark can agree with a reference more than other clinical datasets do; Table S5b has cases (untrained methods, CO2Wounds-V2: 0.80 against 0.28). The real limit is ranking reliability. |
| Discussion, "What a validity figure measures" | "a benchmark that agrees with them at 0.21 is as good a proxy as another clinical dataset and no better" | Equivalence claimed from a difference whose interval is −0.41 to 0.30. |
| Methods, "Confirmatory and exploratory analyses" | the composition effect and modality structure "are tested with the exact relabeling test" | The relabeling test tests only the modality difference; the composition effect uses bootstrap intervals. |
| Results, threshold paragraph | the per-dataset oracle threshold "removes calibration differences" | On Medetec and CO2Wounds-V2, 92% and 96% of models hit the grid minimum (Table S6e); the oracle is censored. |
| Discussion, "Relation to prior work" | pooling "inflates agreement within one modality as well as creating disagreement across modalities" | No inflation in the colonoscopy mirror (Table S6h); true only for the wound domain. |
| Results, modality section | from-scratch group "shows the same structure" while modern group "cannot be interpreted" because of floor | Same floor condition (below 0.1 on six of seven non-wound sets), different treatment. |
| Results, first section | "Each dataset ranks the methods reproducibly: split-half reliability over all 65 methods is 0.97 to 0.99" | Inflated by the between-group composition effect the paper documents; only within-group reliabilities are meaningful. |
| Abstract | "18 ImageNet-pretrained architectures, each trained with three random seeds" | Two MA-Net architectures have 2 and 1 usable seeds. |
| Abstract and Results | pretrained architectures "missed 27.9% of wounds against 1.4% for nnU-Net" presented as an architecture contrast | Confounded with input scaling, augmentation, training length and threshold (M3). |
| Introduction | "two datasets cannot agree on an order that training noise does not preserve" | Too strong; noise attenuates expected agreement but does not make agreement impossible. |
| Results, seed paragraph | "Averaging three seeds did not raise agreement across datasets" used as evidence | Underpowered (difference interval −0.15 to 0.18); the predicted rise is smaller than the interval. Not wrong as a statement of fact, but should not be leaned on. |
| Results, misses | "The new architectures miss less than the original ten (24.9% against 30.3%)" | Intervals overlap; no paired test given. |
| Discussion, "Why wound photography differs" | NSD rise "is consistent with framing conventions contributing to the disagreement" | Selective: shown for untrained methods; the Kvasir-trained group falls under NSD (0.36 to 0.28). |

### B3. Points to check before submission (quick)

- Confirm whether the 150 AZH test images were ever matched against FUSeg training photographs. exclusions_v5.json lists only FUSeg and the AZH training slice. If they were not checked, run the existing SIFT check; if crops match, either exclude them or report the overlap and its effect.
- Confirm who annotated the Medetec masks used (the data come from the FUSeg group's repository) and what the four variants per photograph are, and state both in Methods.
- If you keep the scale explanation (Change 2), make sure its numbers are regenerated for the 18-architecture set and written by script into facts_v6.json, as the brief requires.
