# Scoop check: have claims C1 to C4 already been published?

Date of check: 8 October 2026. Reviewer stance: skeptical. About 45 distinct web searches, with abstracts or full text pages fetched for every work listed below.

How to read this file:
- "Verified" means I opened the page at the URL given and the description comes from that page.
- "Venue from listing" means the venue came from a search result or index page (proceedings index, ML Anthology), not from the paper itself.
- "Unverified" means I could not confirm it from a page I read. Treat those lines with care.
- For most works I read the abstract page only. Where I read the full HTML text, I say so. A full text could contain something closer to our claims than the abstract shows.

Short answer: no single work pre-empts any of C1 to C4 as stated. But C1, C2 and C3 each have close relatives that make their general idea known. The specific combination is still new as far as I can find: medical segmentation, a mixed classical plus modern method pool, clinical references from two modalities, parametric synthetic benchmarks, and a split-half control. C2 is the most exposed. Its core principle, that agreement between benchmarks depends on which models are compared, is already shown in LLM evaluation (Perlitz et al.) and in ID/OOD work (Teney et al.; Zhang, Dominguez-Olmedo and Hardt). C4 looks not pre-empted.

---

## C1. Rank agreement between clinical references of the same task is far from perfect (about 0.94 split-half, about 0.5 same modality, about 0 across modalities)

### Closest prior works

**1. Piérard, Deliège, Halin, Van Droogenbroeck. "A Methodology to Evaluate Strategies Predicting Rankings on Unseen Domains." arXiv:2505.15595, 2025. DOI 10.1109/ICMEW68306.2025.11152102 (IEEE ICME workshops 2025; the venue name is my inference from the DOI prefix).** URL: https://arxiv.org/abs/2505.15595 (verified, full HTML read)
- What they did: ranked 40 unsupervised background subtraction methods on 53 video domains (CDnet). They measured how well rankings on known domains predict the ranking on an unseen domain, using Kendall's tau.
- Finding: the average agreement between domain pairs is low (tau about 0.25, which they describe as about a 62.5% chance of ordering two methods correctly). Some domain pairs have negative tau. They also note that the CDnet ranking changes when methods are added to or removed from the pool.
- Relevance: this is the closest general precedent for "rankings do not carry over between datasets of the same task". It is not medical, has no split-half control, and does not separate same-modality from cross-modality pairs.

**2. Bassi et al. (53 authors, last author Zongwei Zhou). "Touchstone Benchmark: Are We on the Right Way for Evaluating AI Algorithms for Medical Segmentation?" NeurIPS 2024 Datasets and Benchmarks track (venue from proceedings index). arXiv:2411.03670.** URL: https://arxiv.org/abs/2411.03670 (verified; full HTML of v1 read)
- What they did: tested 19 or more abdominal CT segmentation algorithms on external test sets from several hospitals.
- Finding: different winners on different test sets (MedNeXt, STU-Net and MedFormer on JHH; STU-Net and ResEncL on TotalSegmentator). Winners are statistically tied. Per-organ Dice varies by 15% or more across test sets. The text I read gives no cross-dataset rank correlation.
- Relevance: qualitative evidence in medical segmentation that leaderboards depend on the test set. It is not a rank correlation analysis.

**3. Antonelli et al. "The Medical Segmentation Decathlon." Nature Communications 13:4128, 2022. DOI 10.1038/s41467-022-30695-9.** URL: https://www.nature.com/articles/s41467-022-30695-9 (verified)
- What they did: bootstrap ranking stability within each task (median Kendall's tau 0.92 to 0.99). They also compared development-phase and mystery-phase ranks: no top-8 team moved more than two places.
- Relevance: this is a counterpoint. It reports that rankings were fairly stable across tasks for top teams. Note the difference: those are rankings of whole self-configuring pipelines trained on each task, which is not the same as one method pool scored on several clinical references. Reviewers may raise it, so cite it and explain the difference.

**4. Isensee et al. "nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation." MICCAI 2024 (venue from the papers.miccai.org listing). arXiv:2404.09556.** URL: https://arxiv.org/abs/2404.09556 (verified; full HTML of v2 read)
- What they did: scored how well each dataset separates methods, using the ratio of between-method spread to within-method (fold) spread. They comment on cross-dataset rankings only in words (for example, MedNeXt is best everywhere except KiTS).
- Relevance: a well-known paper on which datasets are fit for benchmarking. It does not report cross-dataset rank agreement.

**5. Ali et al. "Assessing generalisability of deep learning-based polyp detection and segmentation methods through a computer vision challenge." Scientific Reports 14:2032, 2024. DOI 10.1038/s41598-024-52063-x.** URL: https://eprints.whiterose.ac.uk/208473/1/s41598-024-52063-x.pdf (verified)
- What they did: EndoCV2021. They compared team performance on seen-centre and unseen-centre colonoscopy data, and folded the deviation into the ranking. Rankings shift between score and deviation. No rank correlation is reported.
- Relevance: colonoscopy-specific evidence that rankings depend on the data source.

**6. Wound-specific cross-dataset studies.**
- Lucho, Naemi, Castañeda, Treuillet. "Can Deep Learning Wound Segmentation Algorithms Developed for a Dataset Be Effective for Another Dataset? A Specific Focus on Diabetic Foot Ulcers." IEEE Access 12:173824-173835, 2024. DOI 10.1109/ACCESS.2024.3502467. URL: https://salford-repository.worktribe.com/output/3434139/ (verified abstract). Seven models tested across FUSeg and DFUC, with "substantial variability". The abstract does not say whether they compared rankings.
- Benfatah. "Do Diabetic Foot Ulcer Segmentation Models Generalize? A Cross-Dataset Benchmark of CNN and Transformer Architectures." arXiv:2607.02555, June 2026. URL: https://arxiv.org/abs/2607.02555 (verified abstract). Three models trained on FUSeg plus AZH and tested on DFUC2022 and Medetec. The authors say the ranking is consistent across the two external sets. This is a possible counterpoint to our same-modality value of about 0.5. Their pool is tiny (3 trained deep models), so it says little about a 32-method pool. Cite it and say why the results differ.

**7. Godau, Srivastava, Ulrich, Adler, K. Maier-Hein, L. Maier-Hein. "Beyond Knowledge Silos: Task Fingerprinting for Democratization of Medical Imaging AI." arXiv:2412.08763 (v2, January 2026).** URL: https://arxiv.org/abs/2412.08763 (verified; full HTML read)
- What they did: transfer between 71 medical tasks. They report very low agreement of outcome rankings across transfer scenarios (weighted tau about 0.0 to 0.08). They did not compare method rankings between datasets of the same task.
- Relevance: indirect. It is from the Maier-Hein group and shows that rankings tied to medical tasks are unstable.

**8. Background works the authors already know (Maier-Hein et al. 2018).** This paper's bootstrap resampling of the test set is the same kind of check as our split-half control. It does not cover cross-dataset agreement, so it pre-empts nothing beyond what the authors assume.

**9. General ML precedent for a same-distribution control.** Recht, Roelofs, Schmidt, Shankar. "Do ImageNet Classifiers Generalize to ImageNet?" ICML 2019, PMLR 97 (venue from proceedings index). arXiv:1902.10811. URL: https://arxiv.org/abs/1902.10811 (verified abstract). The abstract says gains on the original test set carry over to the new one. That rank order is preserved is well known from the paper's figures, but I did not see it stated on the abstract page.

### Verdict for C1: partly pre-empted

The general message, that method rankings do not transfer well between datasets of the same task, is already shown outside medicine (Piérard 2025, tau about 0.25 across video domains). It is also shown qualitatively in medical segmentation (Touchstone 2024; EndoCV2021). I found no medical segmentation work that measures a graded hierarchy (split-half, then independent same-modality, then cross-modality) with a split-half control over one fixed method pool. That quantified hierarchy is the new part.

Suggested phrasing: "Prior work has shown that segmentation leaderboards can change between test sets [Touchstone; Ali 2024] and that rankings transfer poorly across domains in other vision tasks [Piérard 2025]. We quantify this for 2D lesion segmentation over a fixed pool of 32 methods. Rank agreement falls from about 0.94 between disjoint halves of one dataset, to about 0.5 between independent datasets of the same modality, to no detectable agreement across modalities." Do not call the "far from perfect" observation new on its own. Mention MSD (stable top-team ranks) and Benfatah 2026 (consistent ranking for 3 models), and explain the difference.

---

## C2. Synthetic benchmark validity depends on the method pool (98% vs 49% of the clinical ceiling). Mixed pools of trained and untrained methods produce composition (Simpson's paradox) effects that can flip the sign.

### Closest prior works

**1. Perlitz, Gera, Arviv, Yehudai, Bandel, Shnarch, Shmueli-Scheuer, Choshen. "Do These LLM Benchmarks Agree? Fixing Benchmark Evaluation with BenchBench." arXiv:2407.13696 (2024). A version titled "Benchmark Agreement Testing Done Right: A Guide for LLM Benchmark Evaluation" was a poster at a NeurIPS 2025 workshop ("Evaluating the Evolving LLM Lifecycle").** URLs: https://arxiv.org/html/2407.13696v2 (verified, full HTML read) and https://neurips.cc/virtual/2025/122451 (verified)
- What they did: studied "benchmark agreement testing", which validates a benchmark by its rank correlation with established ones. The full text shows that the model subset changes the agreement score a lot. Top 5, top 10 and top 15 models give "substantially different correlation scores". Kendall tau between LMSys Arena and MT-Bench ranges from about 0.65 to 0.99 depending on how many models are used. Agreement is lower among models that sit close together in rank. Bottom-tier, middle-tier and top-tier models give very different tau values.
- Relevance: this pre-empts the general principle that a benchmark's measured validity (its rank agreement with a reference) depends on the pool of models compared. It is LLMs, not segmentation. It has no synthetic data, no trained vs untrained mix, and no sign flip.

**2. Teney, Lin, Oh, Abbasnejad. "ID and OOD Performance Are Sometimes Inversely Correlated on Real-world Datasets." NeurIPS 2023 (main track). DOI 10.52202/075280-3139. arXiv:2209.00613.** URL: https://proceedings.neurips.cc/paper_files/paper/2023/hash/e304d374c85e385eb217ed4a025b6b63-Abstract-Conference.html (verified)
- What they did: showed that the correlation between in-distribution and out-of-distribution accuracy can be negative on real datasets. They argue that earlier positive findings came from "a biased selection of models".
- Relevance: a direct precedent for "the sign of a cross-domain correlation depends on which models are in the pool". Cite it next to Miller et al. 2021.

**3. Zhang, Dominguez-Olmedo, Hardt. "Train-before-Test Harmonizes Language Model Rankings." ICLR 2026. arXiv:2507.05195.** URLs: https://mlanthology.org/iclr/2026/zhang2026iclr-trainbeforetest/ and https://arxiv.org/abs/2507.05195 (verified)
- What they did: 61 models on 24 benchmarks. With direct evaluation, the average Kendall tau between benchmark rankings is 0.51. When every model is first fine-tuned on each benchmark's training data, it rises to 0.74, and agreement improves for 95% of benchmark pairs.
- Relevance: the closest precedent for our mechanism. Disagreement between rankings is largely driven by which models had task-matched training, and equalizing training exposure restores agreement. Our "dominated by which methods had domain-matched training" is the same idea in a different field.

**4. Dominguez-Olmedo, Dorner, Hardt. "Training on the Test Task Confounds Evaluation and Emergence." ICLR 2025 (oral, venue from iclr.cc listing). arXiv:2407.07890.** URL: https://arxiv.org/abs/2407.07890 (verified)
- What they did: showed that apparent superiority of newer model families is largely explained by how much they trained on the test task. The gap disappears after everyone gets the same task data.
- Relevance: an earlier paper by the same group, with the same confound logic as #3.

**5. Synthetic versus real rank agreement, using real-real agreement as the ceiling.**
- Kang. "Is Synthetic Dataset Reliable for Benchmarking Generalizable Person Re-Identification?" IJCB 2022 (venue from the MBZUAI library page). arXiv:2209.05047. URL: https://arxiv.org/abs/2209.05047 (verified). Proposes "Pairwise Ranking Analysis". It compares synthetic-to-real Kendall correlations with real-to-real ones (KS test) over 10 algorithms. The conclusion is that ClonedPerson behaves like real data.
- Borsukiewicz, Lunghi, Ouédraogo, Klein, Bissyandé. "Benchmarking Face Recognition without Real Faces." arXiv:2607.14932, July 2026. URL: https://arxiv.org/html/2607.14932v1 (verified, full HTML read). Spearman rho between synthetic and real benchmarks over 24 models, with real-real pairs as the reference ceiling. The best synthetic sets reach rho above 0.85, and some synthetic sets give negative rho. The paper does not stratify agreement by model subset.
- Shoshan, Bhonker, Kviatkovsky, Fintz, Medioni. "Synthetic Data for Model Selection." ICML 2023 (venue from ML Anthology listing). arXiv:2105.00717. URL: https://ar5iv.labs.arxiv.org/html/2105.00717 (verified). Spearman between synthetic and real error rankings ranges from 0.98 down to -0.19 depending on the generator. Gives a total-variation condition for when rankings are preserved.
- Relevance: these pre-empt the method of measuring a synthetic benchmark's validity as rank agreement with real data, normalized by real-real agreement. Our "98% of the clinical-to-clinical ceiling" uses the same construction. None of them shows validity changing with the method pool.

**6. Maheshwari et al. 2024 (already known).** Its "bias factor" is the effect where a smaller LLM that generated the synthetic data does better on it. That is a narrow case of a domain-matched training advantage, so it pre-empts slightly more than "synthetic data as an NLP benchmark". Cite it for the mechanism as well.

**7. Background.** Taori et al., "Measuring Robustness to Natural Distribution Shifts in Image Classification", NeurIPS 2020 (venue from neurips.cc listing), arXiv:2007.00644 (verified abstract): robustness to synthetic shifts transfers "little to no" to natural shifts. Salaudeen, Chiou, Weng, Koyejo, "Are Domain Generalization Benchmarks with Accuracy on the Line Misspecified?", TMLR 2025, arXiv:2504.00186 (verified abstract): a strong ID-OOD correlation can be an artifact of benchmark design.

### Verdict for C2: partly pre-empted

The principle that rank-agreement validity depends on the model pool is published for LLM benchmarks (Perlitz). The fact that model selection can hide or flip the sign of a cross-domain correlation is published for ID/OOD accuracy (Teney 2023; Miller 2021). The fact that differing task-matched training drives ranking disagreement is published for LLMs (Zhang, Dominguez-Olmedo and Hardt 2026). The synthetic-vs-real, ceiling-normalized agreement measure is also published (Kang 2022; Borsukiewicz 2026). What I could not find: any of this shown for synthetic segmentation benchmarks, with a pool of classical algorithms, pixel classifiers and deep models trained from scratch, with the effect large enough to halve normalized validity (98% vs 49%), and with an explicit composition (Simpson's) sign flip driven by domain-matched training.

Suggested phrasing: "Benchmark agreement is known to depend on the models compared [Perlitz], and differences in task-matched training are known to confound model rankings [Zhang 2026; Dominguez-Olmedo 2025; Teney 2023]. We show the same effect for synthetic segmentation benchmarks. Normalized validity falls from 98% to 49% of the clinical ceiling when modern architectures join a mostly classical pool. In mixed pools, cross-domain correlations are dominated by which methods had domain-matched training, which can reverse their sign." Do not present the ceiling-normalized agreement measure itself as new. Cite Kang 2022 and Borsukiewicz 2026 for it.

---

## C3. A synthetic benchmark's validity cannot be reduced to one measured image statistic (background texture)

### Closest prior works

**1. Borsukiewicz et al. 2026 (above), arXiv:2607.14932.** (verified, full HTML read) States that "FID is necessary but not sufficient for good benchmarking performance". Synthetic sets with worse FID gave better rank agreement with real benchmarks, and "distributional similarity alone does not determine benchmark reliability".
- Relevance: the closest match. It shows, for ranking validity specifically, that one image-level statistic does not determine validity. The difference is that FID is not matched experimentally. It is a correlational observation across existing datasets.

**2. Bartkowiak, Kotrys, Michels, Pirk, Palubicki. "SADGE: Structure and Appearance Domain Gap Estimation of Synthetic and Real Data." arXiv:2605.22467, May 2026.** URL: https://arxiv.org/abs/2605.22467 (verified abstract)
- States that "neither appearance nor geometry alone can reliably predict downstream performance". Their combined measure reaches Spearman 0.77 with downstream performance. This concerns training utility, not benchmark ranking validity.

**3. Xing, Nan, Felder, Walsh, Yang. "The Beauty or the Beast: Which Aspect of Synthetic Medical Images Deserves Our Focus?" arXiv:2305.09789, 2023 (a MICCAI 2023 listing appeared in search results; venue unverified).** URL: https://arxiv.org/abs/2305.09789 (verified abstract)
- Finds that high-fidelity synthetic medical images are not necessarily more useful, and that low-fidelity images sometimes did better downstream. This concerns training utility.

**4. Deo et al. 2025 (already known), arXiv:2505.07175.** (verified abstract) Image quality metrics correlate poorly with downstream segmentation suitability. Generative MRI, not parametric benchmarks, and no method ranking. It pre-empts nothing beyond what the authors assume.

**5. Shoshan et al. ICML 2023 (above).** Gives a sufficient condition for rank preservation based on total variation distance, which is a distribution-level quantity rather than a single image statistic.

**6. Zhang, Konz, Kramer, Mazurowski. "Quantifying the Limits of Segmentation Foundation Models: Modeling Challenges in Segmenting Tree-Like and Low-Contrast Objects." arXiv:2412.04243 (v3, November 2025).** URL: https://arxiv.org/abs/2412.04243 (verified abstract)
- Uses controlled synthetic images in which "textural separability" (object vs background texture) is varied, and checks that the trends also hold on real data. This is a precedent for parametric synthetic segmentation data with controlled texture and contrast. They study what makes SAM-type models fail, not whether the benchmark is valid for ranking.

### Verdict for C3: partly pre-empted

The broad statement, that one image-level statistic such as FID or appearance similarity does not determine whether synthetic data is a valid benchmark, already appears in 2026 preprints (Borsukiewicz for ranking validity; SADGE for utility) and earlier for utility (Xing 2023; Deo 2025). The controlled design is not found: generators matched on a measured background-texture statistic that still differ widely in rank-agreement validity against clinical data.

Suggested phrasing: "Consistent with reports that FID is necessary but not sufficient for synthetic benchmark reliability [Borsukiewicz 2026], and that fidelity metrics poorly predict utility [Deo 2025; Xing 2023], we show by direct matching that synthetic segmentation benchmarks equal in measured background texture differ widely in rank agreement with clinical data." Frame it as a controlled confirmation in a new setting, not as the first observation.

---

## C4. Within a modality, the clinical reference's lesion fraction monotonically decides whether it rewards high or low lesion-to-surround contrast in synthetic benchmarks, under both Dice and NSD

### Closest prior works

**1. Raina, Molchanova, Graziani, Malinin, Muller, Bach Cuadra, Gales. "Tackling Bias in the Dice Similarity Coefficient: Introducing nDSC for White Matter Lesion Segmentation." arXiv:2302.05432, 2023 (an ISBI 2023 listing appeared in search results; venue unverified).** URL: https://arxiv.org/abs/2302.05432 (verified abstract)
- Shows that Dice depends on lesion load (foreground proportion), measured with rank correlation, and proposes nDSC.
- Relevance: a known mechanism by which foreground fraction affects Dice-based comparisons. Our use of NSD as a second metric addresses this, so cite it.

**2. Reinke et al. "Understanding metric-related pitfalls in image analysis validation." Nature Methods 21:182-194, 2024. DOI 10.1038/s41592-023-02150-0.** URL: https://www.research.ed.ac.uk/en/publications/understanding-metric-related-pitfalls-in-image-analysis-validatio-2/ (verified)
- A taxonomy of metric pitfalls. From my own knowledge, not checked on the page, it includes structure size and class imbalance pitfalls for Dice. It is the companion paper to Metrics Reloaded, which the authors already know.

**3. Liu and Jin. "Sensitivity of image-derived wound-healing proxies to sparse visit schedules and image framing: a retrospective multi-dataset audit." Research Square preprint, August 2026 (under review at BMC Medical Imaging).** URL: https://www.researchsquare.com/article/rs-10420381 (verified abstract)
- Shows that zooming or framing wound photos changes area-based healing signals much more than a CNN-based signal.
- Relevance: a framing convention changing conclusions in wound images. Different target: healing proxies, not benchmark validity or method rankings.

**4. Kovács and Fazekas, retinal vessel segmentation FoV paper. arXiv:2111.03853.** URL: https://arxiv.org/abs/2111.03853 (page content verified; title, authors and venue unverified. The page had no front matter. From memory it is "A new baseline for retinal vessel segmentation...", Medical Image Analysis, 2022).
- Shows that scoring inside the field-of-view mask versus over all pixels inflates accuracy and specificity. Rankings across about 100 DRIVE papers mix the two conventions, so they are not comparable.
- Relevance: an evaluation-region (framing) convention changing rankings. A mechanism related to C4, not the same claim.

**5. Liu et al. 2024 MedIA (already known).** Dice size bias in losses. It does not pre-empt more than the authors assume.

### Verdict for C4: not pre-empted

The mechanism is known: Dice depends on foreground proportion, and framing or evaluation-region conventions can change scores and rankings. I found no work showing that a clinical reference's lesion fraction monotonically decides which synthetic contrast regime it rewards, still holding under a boundary metric (NSD).

Suggested phrasing: "Dice is known to depend on foreground proportion [Raina 2023; Reinke 2024], and evaluation-region conventions can reorder rankings [Kovács and Fazekas]. We find that within a modality, the lesion fraction of the clinical reference, a framing or cropping convention, monotonically decides whether that reference rewards high or low lesion-to-surround contrast in synthetic benchmarks. This holds under both Dice and the boundary-based NSD, so it is not purely a Dice region-size artifact." Because the result rests on a few within-modality references, keep "monotonic" tied to the number of references actually tested.

---

## Overall summary

| Claim | Verdict | Most threatening prior work |
|---|---|---|
| C1 rank agreement hierarchy | Partly pre-empted | Piérard et al. 2025 (tau about 0.25 across domains); Touchstone 2024 (winners change across test sets) |
| C2 pool dependence and composition effect | Partly pre-empted (most exposed) | Perlitz et al. (agreement depends on model subset); Zhang, Dominguez-Olmedo, Hardt ICLR 2026 (task-matched training drives disagreement); Teney et al. NeurIPS 2023 (biased model selection hides inverse correlation) |
| C3 not reducible to one statistic | Partly pre-empted | Borsukiewicz et al. 2026 ("FID necessary but not sufficient" for synthetic benchmark ranking validity) |
| C4 lesion fraction decides contrast preference | Not pre-empted | Raina et al. 2023 (Dice and lesion load), for mechanism only |

What remains new, taken together: a medical segmentation study that (a) calibrates synthetic benchmark validity against a measured clinical-to-clinical ceiling with a split-half control, (b) shows that the validity number itself changes with the method pool, (c) traces mixed-pool correlations to domain-matched training, including sign flips, and (d) links a framing convention (lesion fraction) to which synthetic regime a clinical reference rewards. The paper should present (b) and (c) as carrying known evaluation-science results into synthetic medical benchmarks, with stronger and more mechanistic evidence, not as discovering them.

### Works the authors must cite (beyond those already known)
1. Perlitz et al., BenchBench / Benchmark Agreement Testing, arXiv:2407.13696 (NeurIPS 2025 workshop version). Required for C2.
2. Zhang, Dominguez-Olmedo, Hardt, Train-before-Test, ICLR 2026, arXiv:2507.05195. Required for C2's mechanism.
3. Teney et al., NeurIPS 2023, arXiv:2209.00613. Required for C2's sign-flip claim.
4. Kang, IJCB 2022, arXiv:2209.05047, and Borsukiewicz et al. 2026, arXiv:2607.14932. Required for the synthetic-vs-real, ceiling-normalized agreement measure (C2) and for C3.
5. Shoshan et al., ICML 2023, arXiv:2105.00717. Synthetic data ranking validity and a rank-preservation condition (C2, C3).
6. Piérard et al. 2025, arXiv:2505.15595. Cross-domain ranking transfer (C1).
7. Touchstone (Bassi et al., NeurIPS 2024 D&B) and Ali et al. 2024 Sci Rep. Medical evidence that leaderboards change across test sets (C1).
8. Antonelli et al. 2022 (MSD) and Benfatah 2026 (arXiv:2607.02555). Counterpoints for C1 that reviewers may raise.
9. Raina et al. 2023 (nDSC) and Reinke et al. 2024 Nature Methods. Metric mechanism for C4.
Recommended but optional: Dominguez-Olmedo et al. ICLR 2025; Taori et al. NeurIPS 2020; Salaudeen et al. TMLR 2025; Zhang, Konz et al. arXiv:2412.04243; Isensee et al. MICCAI 2024; Lucho et al. IEEE Access 2024; Godau et al. arXiv:2412.08763.

### Side note
The searches also returned a Research Square preprint by Lamia Islam: "Surgical Vision ROS2 ...", DOI 10.21203/rs.3.rs-10362132/v1, July 2026. It uses a 45-image synthetic wound benchmark. It is not a scoop, but if that benchmark shares a generator with this paper, self-cite it for consistency.

## Limits of this check
- Most works were checked at the abstract level. Full texts were read for Perlitz, Godau, Isensee, Touchstone v1, Piérard and Borsukiewicz. A full text elsewhere could hold a closer analysis, for example a cross-dataset rank correlation table in Lucho et al. 2024.
- Paywalled or reCAPTCHA-blocked pages (for example PMC) could not be read. PETSTEP (synthetic PET lesions for evaluating segmentation, Berthon et al.) and the first MICCAI PET segmentation challenge (Hatt et al. 2018, which used phantom and clinical images) came up but could not be opened. Both could be relevant to "synthetic or phantom data used to rank segmentation methods". Check them by hand.
- Web search does not index everything. Conference proceedings from 2026 that have no arXiv version (for example MICCAI 2026 papers) may be under-covered.
