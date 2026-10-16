# R5: Reference verification, manuscript V6 (JIIM submission)

Checked 12 October 2026. Sources: arXiv abstract pages, OpenAlex records (these mirror Crossref metadata), Springer, PMLR, NeurIPS, MLSys and ICLR proceedings pages, and the JIIM submission guidelines. Direct Crossref API and doi.org access were blocked or rate-limited from this environment, so DOIs were checked through OpenAlex and publisher pages. Well-known items (classic architectures, datasets, Nature-family papers) were checked against known publisher metadata.

The list has 69 references, numbered in order of first citation. The numbering in the text runs 1 to 69 with no gaps. Every key in build_v6.js resolves to an entry in refs.json.

New in V6 compared with V5 refs.json: bouthillier2021 (27), varoquaux2022 (28), guo2017 (29), liu2021 (53), xiao2018 (54), liu2022 (55), radosavovic2020 (56). All seven are verified.

Status key: **verified**; **metadata error** (a fix is needed); **cannot verify** (part of the entry could not be confirmed); **does not exist**.
Claim key: **supported**; **partial** (supports part of the sentence, or only weakly); **not supported**; **better source** (supported, but a stronger or published source exists).

## Table

| # | Key | Status | Correction | Claim support |
|---|-----|--------|------------|---------------|
| 1 | badano2018 | verified | none | supported: VICTRE in-silico imaging trial |
| 2 | sizikova2023 | verified | DOI 10.52202/075280-1626 and pp. 37401-37412 confirmed. JIIM wants all 7 authors: add Lago M, Sahiner B, Delfino J, Badano AG | supported |
| 3 | tushar2026a | verified | arXiv v1 7 May 2026, v2 12 Jun 2026. Only 4 authors, so list them all: "Tushar FI, Momy UH, Lo JY, Rubin GD" | supported in both places: ranks on synthetic data correlate with real data (rho 0.93) |
| 4 | perezgarcia2024 | verified | LNCS 15070, pp. 358-376 confirmed on Springer. Springer's own citation gives the year as 2025 (first online 28 Nov 2024). 2024 is acceptable but should be applied the same way across all LNCS entries | supported: edited images used to stress-test models |
| 5 | maierhein2018 | verified | none | supported: the exact source for test cases, metric, aggregation and annotators |
| 6 | reinke2024 | verified | none | supported |
| 7 | dehghani2021 | verified | arXiv only; no archival version found | supported |
| 8 | pierard2025 | verified | DOI and pp. 1-6 confirmed | supported |
| 9 | bassi2024 | verified (pages cannot verify) | NeurIPS 37 D&B confirmed. No page range or DOI in the entry; add them from the NeurIPS 2024 proceedings if available | supported: tests on 11 external hospitals |
| 10 | ali2024 | verified | none | supported: multi-centre generalisation |
| 11 | urooj2026 | verified | arXiv v1 9 Jul 2026; authors confirmed | **partial**: the paper audits metric choice, split protocols and missing significance tests. It is not about which hospital the test data come from. Move it to the "test cases, metric, aggregation" group [5, 6], or drop it here |
| 12 | kang2022 | verified | DOI 10.1109/IJCB54206.2022.10007952, single author, pp. 1-8 confirmed (V5 used the arXiv URL 2209.05047; the V6 DOI is correct) | supported: Kendall rank correlation of synthetic vs real, compared with real vs real |
| 13 | borsukiewicz2026 | verified | arXiv v1 16 Jul 2026; 5 authors confirmed | supported: agreement measured against real-vs-real disagreement |
| 14 | hu2023 | verified | arXiv only | partial (minor): Hu validates synthetic data by downstream test gains, not by rank correlation, so "on the same basis" fits refs 15 and 16 better than 14 |
| 15 | vanbreugel2023 | verified | DOI and pp. 1889-1904 confirmed | supported |
| 16 | shoshan2023 | verified | PMLR 202:31633-31656 confirmed | supported |
| 17 | tushar2026b | verified | arXiv v1 28 Jun 2026, single author | supported: presence and size orderings transfer, lobe does not |
| 18 | perlitz2024 | verified | arXiv v2 Sep 2024. A retitled version ("Benchmark Agreement Testing Done Right") was a NeurIPS 2025 workshop poster; not archival, so arXiv is fine | supported: "BAT conclusions depend on the models considered"; agreement is lower among similarly strong models |
| 19 | zhang2026 | verified | ICLR 2026 confirmed (oral) | **not supported (overstated)** in both sentences. The paper says direct-evaluation rankings disagree across benchmarks and that benchmark-specific fine-tuning fixes this. It does not show that agreement depends on the model pool, and it does not establish a composition or Simpson effect. Drop it from both sentences, or reword (e.g. "and on how models were trained before testing [19]") |
| 20 | teney2023 | **metadata error** (incomplete) | Add pp. 71703-71722 and DOI 10.52202/075280-3139 (NeurIPS 36), in place of the arXiv URL | supported: inverse correlation is missed when "a biased set of models" is evaluated |
| 21 | miller2021 | verified | PMLR 139:7721-7735 correct. Prefer the PMLR URL over arXiv for consistency with 22 and 29 | supported (as a group with 20 and 22) |
| 22 | fang2022 | verified | none | supported: training data sets the effective-robustness line |
| 23 | simpson1951 | verified | Volume, issue, pages and DOI correct. In 1951 the journal was "Journal of the Royal Statistical Society: Series B (Methodological)"; the "Statistical Methodology" name is the current one | supported |
| 24 | kievit2013 | verified | Only 4 authors; list "Kievit RA, Frankenhuis WE, Waldorp LJ, Borsboom D" (V5 had the full list) | supported |
| 25 | lucho2024 | verified | IEEE Access 12:173824-173835 and DOI confirmed | **not supported**: compares only deep models (U-Net variants, DeepLabV3+, SAM) across FUSeg and DFUC. No classical colour or threshold methods |
| 26 | benfatah2026 | verified | arXiv v1 27 Jun 2026, single author | **not supported**: compares U-Net, DeepLabV3+ and SegFormer-B2 only. No classical methods |
| 27 | bouthillier2021 | verified | MLSys 3:747-769 confirmed | supported |
| 28 | varoquaux2022 | verified | none | partial for the first sentence (the paper is mainly about evaluation noise and test-set size, not seed variance); supported for the third sentence ("training seeds and small test sets") |
| 29 | guo2017 | verified | none | supported |
| 30 | medetec2014 | verified | Page exists; metadata date 2014-02-06. Matches the JIIM online-document format | supported |
| 31 | sanchez2024 | verified | arXiv only. The text's numbers are confirmed: 764 images, 96 patients, a hospital in Colombia, smartphone cameras. Format: other arXiv entries use "arXiv preprint" plus the abs URL; this one uses "arXiv preprint arXiv:2408.10827" plus the 10.48550 DOI. Pick one style | supported |
| 32 | wang2024 | verified | none | supported |
| 33 | wang2020 | verified | none | supported |
| 34 | jha2020 | verified | none | supported |
| 35 | bernal2015 | verified | none | supported (CVC-ClinicDB) |
| 36 | bernal2012 | verified | none | supported (CVC-ColonDB) |
| 37 | silva2014 | verified | none | supported (ETIS-Larib) |
| 38 | mendonca2013 | verified | none | supported (PH2) |
| 39 | gong2021 | verified | none | **better source**: the TN3K dataset (3,493 images, test split) is described and released with the journal version that the TRFE-Net repository asks users to cite alongside the ISBI paper. That is Gong et al., "Thyroid region prior guided attention for ultrasound segmentation of thyroid nodules", Computers in Biology and Medicine. Add it next to [39]. Volume, article number and DOI were not confirmed here (believed 155:106389, 2023, 10.1016/j.compbiomed.2022.106389) and must be checked before use |
| 40 | iakubovskii2019 | verified | Repository name is styled "Segmentation Models PyTorch" (capital T) | supported |
| 41 | ronneberger2015 | verified | none | supported |
| 42 | zhou2018 | verified | none | supported |
| 43 | chen2018 | verified | none | supported |
| 44 | lin2017 | verified | none | supported |
| 45 | zhao2017 | verified | none | supported |
| 46 | chaurasia2017 | verified | none | supported |
| 47 | fan2020 | verified | none | supported |
| 48 | he2016 | verified | none | supported |
| 49 | sandler2018 | verified | none | supported |
| 50 | tan2019 | **metadata error** (minor) | "Tan M, Le Q" should be "Tan M, Le QV". Prefer the PMLR URL (proceedings.mlr.press/v97/tan19a.html) over arXiv | supported |
| 51 | xie2021 | verified | none | supported |
| 52 | deng2009 | verified | none | supported |
| 53 | liu2021 | verified | IEEE pagination 9992-10002 correct | supported |
| 54 | xiao2018 | verified | none | supported |
| 55 | liu2022 | verified | none | supported |
| 56 | radosavovic2020 | verified | none | supported |
| 57 | isensee2021 | verified | none | supported: the nnU-Net repository asks v2 users to cite this paper |
| 58 | kirillov2023 | verified | IEEE pagination 3992-4003 correct (the CVF open-access version is 4015-4026; V5 used that) | supported |
| 59 | ma2024 | verified | none | supported |
| 60 | islam2026 | verified | Research Square preprint, posted 16 Jul 2026, single author, DOI resolves in OpenAlex | supported. See Blinding below |
| 61 | nikolov2018 | verified | **better source**: the peer-reviewed version is already in refs.json as nikolov2021: "Nikolov S, Blackwell S, Zverovitch A et al.: Clinically applicable segmentation of head and neck anatomy for radiotherapy: Deep learning algorithm development and validation study. J Med Internet Res 23(7):e26151, 2021. https://doi.org/10.2196/26151". Swap the key in build_v6.js | supported (origin of surface Dice / NSD) |
| 62 | wiesenfarth2021 | verified | none | supported |
| 63 | christodoulou2024 | verified | LNCS 15010, pp. 124-132, DOI confirmed (22 authors) | supported |
| 64 | tobin2017 | verified | none | supported |
| 65 | tremblay2018 | **metadata error** (minor) | pp. 969-977 is the CVF open-access pagination, but the DOI is the IEEE record, which starts at p. 1082. Crossref/OpenAlex shows "1082-10828", an upstream typo; the 9-page paper implies 1082-1090. Confirm on IEEE Xplore, then use the IEEE range with the IEEE DOI | supported |
| 66 | prakash2019 | **metadata error** (incomplete) | Add pages: "pp. 7249-7255" (confirmed in OpenAlex) | supported |
| 67 | raina2023 | verified (DOI cannot verify) | Published at ISBI 2023 (in the IEEE SPS resource centre). The entry gives pp. 1-5 with an arXiv URL; add the IEEE DOI from Xplore. The per-paper DOI could not be confirmed here | supported for "Dice depends on lesion size" (the paper shows dependence on lesion load). Adding [6] (Reinke 2024, which covers Dice's sensitivity to structure size) would strengthen it |
| 68 | torralba2011 | verified | none | supported (capture or framing bias) |
| 69 | kendrick2022 | verified | arXiv v2 3 Oct 2022; no published version found. 9 authors | supported, plausibly: clinical vs refined delineation conventions |

## Problem rows only

| # | Key | Problem | Fix |
|---|-----|---------|-----|
| 25 | lucho2024 | Claim not supported: deep models only | Reword the sentence or cite a wound study that actually benchmarks classical colour or threshold methods against networks (see below) |
| 26 | benfatah2026 | Claim not supported: deep models only | Same as 25 |
| 19 | zhang2026 | Overstated in both sentences | Drop it, or reword to "rankings also depend on how models are trained before testing" |
| 11 | urooj2026 | Weak fit for "the hospital the test data come from" | Move it to [5, 6] (metric and protocol choice) |
| 61 | nikolov2018 | Preprint cited when a peer-reviewed version exists | Use nikolov2021 (JMIR 23(7):e26151, 2021) |
| 39 | gong2021 | TN3K is described in the journal version | Add the Computers in Biology and Medicine paper after confirming its metadata |
| 20 | teney2023 | Missing pages and DOI | NeurIPS 36, pp. 71703-71722, https://doi.org/10.52202/075280-3139 |
| 66 | prakash2019 | Missing pages | pp. 7249-7255 |
| 65 | tremblay2018 | CVF pages paired with the IEEE DOI | Use the IEEE pagination starting at p. 1082 (confirm the end page on Xplore) |
| 50 | tan2019 | Initials | "Le QV" |
| 67 | raina2023 | No DOI; arXiv URL for an ISBI paper | Add the IEEE DOI from Xplore |
| 9 | bassi2024 | No pages or DOI | Add from the NeurIPS 2024 proceedings if available |
| 3, 24 | tushar2026a, kievit2013 | "et al." hides a single fourth author | List all four authors |
| 14 | hu2023 | Minor: "on the same basis" fits it loosely | Optional rewording |
| 28 | varoquaux2022 | Minor: seed claim rests mainly on 27 | Optional: cite only [27] in the first sentence |

On 25 and 26: zhang2022 (a wound-imaging survey in refs.json covering both traditional and deep methods) can back a reworded sentence such as "wound segmentation work spans classical colour and threshold methods and trained networks". It does not back "studies compare" either. During this check I did not find a wound study that benchmarks both families head-to-head, so the authors should supply one or reword the sentence.

## Format against the JIIM guidelines

Checked against link.springer.com/journal/10278/submission-guidelines.

1. **All authors.** The guideline pattern is "Name(s) and initials of all author(s)". The list cuts to three authors plus "et al." in 27 entries: 1, 2, 3, 4, 5, 6, 7, 9, 10, 18, 21, 22, 24, 27, 33, 39, 52, 53, 58, 59, 61, 62, 63, 65, 66, 67, 69. Entries 3 and 24 have only four authors. Either list every author, or confirm that the journal accepts "et al." (the printed example shows three authors, all listed).
2. **Journal names and layout.** The guideline asks for Index Medicus abbreviations and the pattern "J Digit Imaging 12:203-204, 1999". The list keeps full names and writes "Nature Methods 18(2), 203-211, 2021". build_v6.js says the full names are deliberate, but this departs from the guideline. JIIM layout would be "Nat Methods 18:203-211, 2021". It applies to all journal entries: 1, 5, 6, 10, 23, 24, 25, 28, 32, 33, 35, 36, 37, 47, 57, 59, 62.
3. **Unpublished works.** The guideline says "Only works referred to in the text and already accepted for publication can be included". Twelve entries are preprints: 3, 7, 11, 13, 14, 17, 18, 26, 31, 60, 61, 69. Entry 40 is a GitHub repository. Number 61 can be swapped for its published version. For the rest, ask the editorial office, or move them to footnotes or text mentions if the rule is enforced. Many of these (3, 11, 13, 17, 26) carry recent and central claims.
4. **DOI and URL handling is inconsistent.** Some conference papers that have DOIs or proceedings pages are linked to arXiv: 20 (has a NeurIPS DOI), 21, 50 (PMLR pages exist), 67 (IEEE), 9, 51. Entries 22, 29 and 27 use proceedings URLs. Entry 31 uses a different arXiv style from the other preprints.
5. **Venue naming is inconsistent.** CVPR appears as "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)" (44, 45, 48), as "Proceedings of the IEEE/CVF Conference on ..." (49), and as "2020 IEEE/CVF Conference on ..." (55, 56, 52). ICCV and ECCV vary in the same way. Pick one form.
6. **Capitalisation.** Titles are mostly in sentence case. "FUSeg: The Foot Ulcer Segmentation Challenge" (32) and "Segmentation Models Pytorch" (40) are in title case. The capital after a colon is applied consistently.
7. **Pagination source is mixed.** Most IEEE CVPR/ICCV entries use IEEE Xplore pages (44, 45, 53, 58), but 65 uses CVF pages.

## Blinding

Entry 60 (Islam L, 2026, single author, Research Square) is cited in the third person. The text describes it as the source of the reproduced benchmark (Generator A). In a double-blind JIIM submission, the name in the reference list combined with "reproduces the benchmark released with" may identify the author. Consider "Anonymous, 2026" in the blinded copy, following the journal's instructions for self-citation.

## Counts

- References checked: 69. All exist (does not exist: 0).
- Core metadata correct (authors, title, venue, year, volume, pages, DOI): 63.
- Metadata errors or gaps: 4 (20 missing pages and DOI; 50 initials; 65 pagination; 66 missing pages). Format-only author truncation in 3 and 24 is not counted here.
- Cannot verify one element: 2 (9 pages and DOI; 67 IEEE DOI).
- Claim support: supported 60; partial 4 (11, 14, 28, 67 minor); not supported or overstated 3 (19, 25, 26); better source available 2 (39, 61).
- The 8 references dated 2025 or 2026 (3, 8, 11, 13, 17, 19, 26, 60) all exist and match their metadata. Entry 4 also has the LNCS-year note. The claim problems among them are 19 and 26, and 11 in part.

## Limitations

- Crossref and doi.org could not be queried directly; DOIs were confirmed through OpenAlex, which mirrors Crossref, and publisher pages.
- Claim verdicts rest on abstracts plus targeted passages (Perlitz, CO2Wounds-V2), not full readings.
- Classic references (41 to 59, 34 to 38) were checked against known publisher metadata rather than fetched one by one.
