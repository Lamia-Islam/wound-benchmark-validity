# R4: JIIM compliance, language and layout review (manuscript V6, blinded)

Reviewed 12 October 2026. Files read: `manuscript_v6_blinded.docx` (XML unpacked), `review/manuscript_v6_text.txt`, `review/manuscript_v6_blinded.pdf` (21 pages, rendered at 80 and 110 dpi), `Online_Resource_1.docx` (converted to PDF in scratch, 53 landscape pages, rendered), `Online_Resource_2.xlsx` (all 5 sheets), `figures/*` (PNG, TIF, PDF), `build_v6.js` header, `BRIEF_MS_V6.md`. No existing file was edited.

## 1. What was verified on the JIIM site

Source: https://link.springer.com/journal/10278/submission-guidelines (fetched twice on 12 October 2026). The fetch tool returns a summary with short quotes, not the full page. The linked page https://link.springer.com/journal/10278/ethics-and-disclosures could not be read (the proxy refused it with HTTP 429, rate limited), so ethics and data-policy points below rest on the main guidelines page only.

Points confirmed on the page:
- Hypothesis-driven research: Abstract, Introduction, Materials and Methods, Results, Discussion, Conclusions, Acknowledgements.
- Abstract 150 to 250 words, no undefined abbreviations or unspecified references. 4 to 6 keywords.
- Double-blind: a full version and a blinded version; no names or affiliations in the blinded text or title page. "Self-identifying citations and references in the article text should either be avoided or left blank."
- Title page: names, affiliations, e-mail, ORCID; acknowledgements on the title page; "The following statements must be included on the separate Title Page under the heading 'Statements and Declarations'": Funding, Competing Interests (ICMJE form), Author Contributions (encouraged). Ethics approval and consent for human data. The page also says ethics and consent statements go in a "Declarations" section before the reference list, so the two instructions are not reconciled. Keeping ethics, consent and data availability in the blinded manuscript and also on the title page covers both.
- LLM use must be documented in Methods; use for copy editing only does not need declaring.
- References numbered in order of first citation; journal names "as abbreviated in Index Medicus"; names and initials of all authors (no et al. rule is given). Example: "Zaidel M, Hopper K, Iyriboz T: Interactive web-based radiology teaching file. J Digit Imaging 12:203-204, 1999". Web source: "Roxio Inc. Napster. Available at http://www.napster.com. Accessed 26 January 2003."
- Figures: EPS for vector, TIFF for halftones; line art at least 1200 dpi, halftone 300 dpi, combination 600 dpi; widths 39, 84, 129 or 174 mm, height at most 234 mm; Helvetica or Arial, 8 to 12 pt (2 to 3 mm); lines at least 0.1 mm; files named Fig1.eps etc.; captions in the manuscript, starting with bold "Fig." and bold number, no punctuation after the number or at the end; lower-case panel letters; must work in black and white.
- Tables: Word table function, Arabic numbers, cited in order, caption for each; footnotes as superscript lower-case letters (asterisks for significance) below the table.
- Supplementary material: cited as "Online Resource"; files named consecutively (ESM_1.pdf, ESM_2.xlsx); each file should carry article title, journal name, author names and corresponding author details; text files preferably PDF; spreadsheets .xlsx or .csv; ESM figures numbered separately.
- Page numbering with Word's automatic page numbers. Plain font such as 10-point Times Roman. At most three heading levels. Abbreviations defined at first mention. No word limit stated. Line numbering not mentioned.

## 2. Guideline checklist

| # | Item | Result | Evidence |
|---|---|---|---|
| 1 | Article type and section order | PASS | Abstract, Introduction, Materials and Methods, Results, Discussion (with Recommendations, Limitations), Conclusions, Statements and Declarations, References. Acknowledgements correctly left to the title page. Two heading levels only. |
| 2 | Abstract 150 to 250 words, unstructured allowed | PASS, at the limit | 250 words by whitespace count. Any added word breaks the limit; Editorial Manager may count differently. |
| 3 | No undefined abbreviations in abstract | PASS | Only names (Dice, Spearman, ImageNet, nnU-Net). |
| 4 | Keywords 4 to 6 | PASS | 6 keywords. |
| 5 | Blinded manuscript: no names or affiliations in text | FAIL | Reference 60 "Islam L: Surgical Vision ROS2 ... Research Square preprint, 2026", cited at Methods, Synthetic benchmarks: "Generator A ... reproduces the benchmark released with an earlier system description [60]". Single-author self-citation that names the author. |
| 6 | File metadata (docx core, app, custom props, comments, tracked changes) | PASS | Manuscript core: title only, lastModifiedBy "Un-named", no creator; custom.xml empty; comments.xml has no comments; no w:ins/w:del; PDF has no Author field. OR1 creator "python-docx", OR2 creator "openpyxl". Figure PNG/PDF metadata only name Matplotlib. |
| 7 | Online Resources blinded | PASS | No names in OR1 or OR2 text, headers, footers or metadata. |
| 8 | Statements and Declarations | PARTIAL | Ethics, consent and data availability present. Funding, competing interests, author contributions absent by design (title page); author must supply. No LLM-use statement (see Major M3). |
| 9 | Reference numbering and in-text format | PASS | Numbered in order of first citation, brackets, ranges with en dash. |
| 10 | Reference style vs JIIM example | FAIL | Full journal names (e.g. "Nature Communications 9, 5217, 2018") instead of Index Medicus ("Nat Commun 9:5217, 2018"); volume and pages separated by comma, not colon; "et al." after three authors although JIIM asks for all authors. The build header records the full-name choice as deliberate; it still departs from the stated rule. |
| 11 | Figure format and resolution | FAIL (minor) | TIFF 600 dpi RGB LZW. Figures 1 and 2 are line art (no halftone), for which JIIM asks 1200 dpi bitmap or EPS vector. Vector PDFs exist but JIIM names EPS. |
| 12 | Figure width | FAIL for Fig. 2 | Fig. 1 174 x 66 mm, Fig. 3 174 x 96 mm: pass. Fig. 2 140 x 74 mm: not one of 39, 84, 129, 174 mm. |
| 13 | Figure lettering | FAIL (minor) | All figures use DejaVu Sans (pdffonts), not Helvetica or Arial. Sizes in Fig. 1 to 3 are about 8 to 10 pt at print size: pass. Fig. S1 cell values about 5 pt. |
| 14 | Figure file names | FAIL (minor) | `Fig1_pooling.tif` etc.; JIIM pattern is `Fig1.tif`. |
| 15 | Captions start with bold "Fig." and number, no final punctuation | PASS | Checked in XML: "Fig. 1" bold, caption text plain, no end stop. Same for Tables 1 to 5. |
| 16 | Black and white legibility | PASS | All figures greyscale (0 coloured pixels), markers differ by shape and fill. |
| 17 | Tables: Word tables, captions, cited in order | PASS | Tables 1 to 5 are real w:tbl tables, cited in order (Table 1 Methods, 2 Methods, 3 and Fig. 1 Results, 4 and Fig. 2, 5 and Fig. 3). |
| 18 | Table footnotes | FAIL (minor) | Under Table 3: "Within modality: different sources. The smallest attainable p is 0.00014." not in superscript-letter form. |
| 19 | Figures cited in order | PASS | Fig. 1a first at Results "Pooling creates agreement", then Fig. 1b, 1c, Fig. 2, Fig. 3a, 3b. |
| 20 | Online Resource naming and citation | PASS in text, FAIL for files | Text says "Online Resource 1/2", cited in order (OR1 in Methods, OR2 in Results). Files should be named ESM_1 and ESM_2; OR1 should be supplied as PDF; neither file carries the journal name. |
| 21 | Online Resource tables cited match OR1 | PARTIAL | OR1 holds Tables S1 to S9 (with sub-tables a to j) and Figure S1, matching the brief. One citation does not support its sentence (see Major M6). |
| 22 | Page numbers | FAIL | Manuscript sectPr has no footer reference; no header/footer parts exist; the PDF has no page numbers. |
| 23 | Line numbers | Not required | JIIM does not mention them. Manuscript has none (no w:lnNumType). Optional but reviewers often want them. |
| 24 | Font | PASS | Times New Roman, 11 pt body, 9.5 pt captions, 8 to 9.5 pt tables. |
| 25 | Word limit | PASS | None stated. Main text about 8,200 words including tables. |
| 26 | Abbreviations defined at first use | FAIL (minor) | SIFT, RANSAC, GPU, CT, SAM, MedSAM, FUSeg not expanded (Minor m2). |

## 3. House-rule checklist (BRIEF_MS_V6.md)

| Rule | Result | Evidence |
|---|---|---|
| No em dash (U+2014) | PASS | 0 in manuscript XML (all parts), 0 in OR1 XML, 0 in OR2 strings; none seen in the four rendered figures. |
| En dash not for negatives | PASS | 6 en dashes, all in citation ranges ([1–4], [9–11], [14–16], [20–22] twice, [64–66]). |
| Minus sign U+2212 consistently | PASS in docx, PARTIAL in OR2 | 52 in manuscript, 1,872 in OR1, none of ASCII "-" before a digit in either. Figures use U+2212 (matplotlib default). OR2 stores numbers as numbers with format "0.000", so Excel shows an ASCII hyphen-minus for negatives. |
| British spelling | FAIL (minor) | "relabeling" 3 times in manuscript (Methods "Agreement between image sets" twice: "relabeling (52 pairs)", "exact relabeling test"; Table 3 caption "exact relabeling p") and 7 times in OR1 ("relabeling", "relabelings"). All else British (colour, normalised, organised, binarised, programme, centre, randomisation). "AZH Wound Center" is a proper name. |
| Black, white, grey only; readable greys; links #28598A underlined | PASS | Run colours used: 000000 text, 3F3F3F captions (darker than the #444444 limit), E6E6E6 rules and shading, 28598A on all 68 hyperlinks, all with single underline in 28598A. Latent risk: styles.xml keeps Word defaults (Heading1/2 #2E74B5, Hyperlink #0563C1); they are overridden at run level now but would show if an editor re-applies styles. |
| No placeholder or draft text | FAIL in Online Resources | Manuscript clean. OR1: "M (V6) = P10_s0 + nnU-Net" (internal version tag). OR2 README "Sources" section lists internal paths and script names: "rebuild/results/perimage32", "rebuild/results/kaggle*/scores", "rebuild/scripts/perimage_v7.py", "rebuild/results/v7/stats_v7.json", "pools_v5", "online_resources_v6.py". |
| No author names | FAIL | Reference 60 (see Critical C1). |

## 4. Issues by severity, with location and fix

### Critical (fix before submission)

**C1. Self-identifying reference breaks double-blind review.**
Location: Methods, Synthetic benchmarks: "Generator A places a lesion on a uniform drape and reproduces the benchmark released with an earlier system description [60]." Reference 60: "Islam L: Surgical Vision ROS2: a ROS2 Humble pipeline for intraoperative wound assessment and variable-autonomy arm control. Research Square preprint, 2026."
Why: JIIM says self-identifying citations should be avoided or left blank. A single-author preprint whose benchmark the paper reproduces points straight at the author.
Fix: in the blinded build only, replace the entry with "60. Reference removed for double-blind review" (keep the number so numbering is unchanged) and keep the sentence in third person. Restore the full entry in the unblinded version.

**C2. No page numbers.**
Location: `document.xml` sectPr has `<w:pgNumType/>` but no footer; the PDF pages are unnumbered.
Why: JIIM asks for automatic page numbering.
Fix in `build_v6.js` (docx library): add to the section `footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: L.FONT, size: 18 })] })] }) }`. While there, consider continuous line numbers for reviewers: `properties: { lineNumbers: { countBy: 1, restart: LineNumberRestartFormat.CONTINUOUS } }`.

### Major

**M1. Fact mismatch about the FUSeg duplicates between manuscript and Online Resources.**
Manuscript (Methods, Clinical datasets): "In FUSeg, 8 validation images are byte-identical to FUSeg training images (7) or to another validation image (1)."
OR1 Table S9a caption: "Exclusions: 8 FUSeg test images are byte-identical to FUSeg training images".
OR2 README "Kept images": "the clinical sets exclude 8 FUSeg test images that duplicate FUSeg training images".
Fix: in both Online Resources write "8 FUSeg validation images that duplicate FUSeg training images (7) or another validation image (1)". Also use "validation" consistently (Table 1 says "FUSeg (validation)").

**M2. Internal labels and file paths in the Online Resources.**
OR2 README "Sources" block (four lines of repository paths and script names, "to 1e-12", "pools_v5"). OR1 notation "M (V6) = P10_s0 + nnU-Net"; codes "C16 = P9 + N7", "N8_sk", "P10_s0" in Table S4 and S5 captions.
Fix: replace the OR2 "Sources" block with two plain sentences (for example: "Clinical and control scores are the per-image scores used in the main analyses; nnU-Net uses the first-session scores. FUSeg at 256 px uses the rescored probability maps described in Online Resource 1, Table S6f."). In OR1 drop "(V6)" and give each code a plain name at first use ("C16, the 16 architectures with three seeds").

**M3. No statement on AI-assisted writing.**
JIIM requires LLM use to be documented in Methods unless it was copy editing only. This manuscript and its Online Resources were drafted and built with AI agents.
Fix (author decision): add one sentence at the end of Materials and Methods, for example "A large language model (Claude, Anthropic) was used to help draft text and analysis scripts; the author checked all content and takes full responsibility for it." Wording must reflect what was actually done.

**M4. Reference style departs from JIIM.**
Examples: ref 5 "Nature Communications 9, 5217, 2018" should read "Nat Commun 9:5217, 2018"; ref 6 "Nature Methods 21(2), 182-194, 2024" should read "Nat Methods 21:182-194, 2024"; ref 1 "Badano A, Graff CG, Badal A et al." should list all authors.
Fix in the formatter (`lib.js` ref function plus a journal-abbreviation field in `refs.json`): "Authors: Title. Abbrev Journal Vol:pages, year. DOI". Keep DOIs. Also:
- Ref 30 (Medetec): the URL is plain black text while all other URLs are links; make it a link.
- arXiv entries are written two ways: ref 31 "arXiv preprint arXiv:2408.10827, 2024. https://doi.org/10.48550/..." versus others "arXiv preprint, 2026. https://arxiv.org/abs/...". Pick one.
- Ref 61 (NSD, Nikolov et al., arXiv 2018) has a peer-reviewed version in J Med Internet Res (2021); citing it is stronger for a journal readership. Verify before switching.

**M5. Figures not to JIIM specification.**
- Fonts: set matplotlib to Arial or Helvetica (`rcParams["font.family"] = "Arial"`, or Liberation Sans if Arial is not installed, then check pdffonts).
- Fig. 2 is 140 mm wide: re-render at 129 mm (or 174 mm) and place it at that width in the docx (`fig(file, 129)`).
- Supply vector EPS (or PDF) for Figs. 1 to 3, or TIFF at 1200 dpi for the line-art Figs. 1 and 2. Fig. 3 (grey band) is fine at 600 dpi.
- Rename to `Fig1.tif`, `Fig2.tif`, `Fig3.tif` (and EPS equivalents).
- Fig. S1: cell numbers are about 5 pt at 174 mm; raise to at least 7 pt or drop the numbers in panels where OR2 already gives values. FigS1 TIFF is RGBA; save as RGB.
- Fig. 1: the panel titles nearly touch ("(domain: wound photography)" of a and b) and "c Untrained with Kvasir-trained" runs to the right edge. Shorten to two short lines or reduce title size slightly.

**M6. A citation to Online Resource 1 does not support its sentence; sub-tables not named.**
Results, "Modality structure ...": "For these three groups the difference holds without the three within-source pairs (...), under NSD (...) and with empty masks excluded (...; Online Resource 1, Tables S2 and S3)." Table S3 is leave-one-dataset-out and leave-one-method-out, which the sentence does not mention; the support is Tables S2b (NSD), S2e (52-pair test) and S2f (empty masks excluded). Table S3 is cited nowhere else, so its content is never described in the main text.
Fix: cite "Tables S2b, S2e and S2f", and either add one sentence on the leave-one-out results with "Table S3" or drop the citation. Generally cite sub-tables where a single one is meant: "Table S9b" (nnU-Net re-prediction), "Table S9d" (WoundSeg), "Table S5g" (leave one method out), "Table S6g" (pooled robustness), "Table S7b" (synthetic chooser by seed), "Table S4d" (seed averages).

**M7. The stratification derivation is typeset as plain text with underscores.**
Methods, "Stratified agreement and why it equals the within-group average": "n_g", "c_g", "v_g = (1 − 1/n_g²)/12", "Σ w_g r_g / Σ w_g with w_g = n_g − 1/n_g". In the PDF the final formula breaks across lines ("w_g = n_g" / "− 1/n_g").
Fix: use real subscripts (TextRun with `subScript: true`) and set the two key results as displayed equations on their own lines: stratified agreement = Σ_g w_g r_g / Σ_g w_g, with w_g = n_g − 1/n_g.

**M8. Abstract is at the 250-word limit and has three unclear phrases.**
- "We measured how much this figure depends on the methods compared" : "this figure" has no antecedent. Write "how much this rank correlation depends on ...".
- "and different wound datasets agreed less (0.10)": this value is for different datasets at different seeds. Write "and two different wound datasets at different seeds agreed at 0.10".
- "on one dataset the pretrained architectures missed 27.9% of wounds": name it: "on the AZH dataset the 18 pretrained architectures missed 27.9% of wounds (Dice below 0.1)".
Trim about 15 words elsewhere so the total lands near 235 (for example drop "a value that" style connectors, or shorten the method-group list to "18 untrained, 14 from-scratch, 14 colonoscopy-trained and 19 modern models").

### Minor

**m1. British spelling.** "relabeling" to "relabelling" (3 in manuscript, 7 in OR1).

**m2. Abbreviations not expanded at first use.**
- "SIFT with RANSAC" (Methods, Clinical datasets): "scale-invariant feature transform (SIFT) keypoints matched with random sample consensus (RANSAC)".
- "GPU" (Methods, Training seeds): "graphics processing unit (GPU)".
- "CT" (Discussion, Relation to prior work: "reported for CT [17]"); Introduction says "computed tomography" without "(CT)". Add "(CT)" in the Introduction.
- "SAM [58] and MedSAM [59]": "Segment Anything Model (SAM) and its medical version MedSAM".
- "FUSeg": "the Foot Ulcer Segmentation (FUSeg) challenge validation set".
- "P0": OR1 uses P0 for the main post-processing pipeline but the manuscript only names P1 and P2. Add "(P0)" to the post-processing paragraph in Methods.
- "Dice": add "Dice similarity coefficient (Dice)" at its first Methods use.

**m3. Jargon that needs a short plain explanation at first use** (one clause each):
- Spearman correlation (abstract and Methods, Agreement between image sets): "a rank correlation: 1 means two image sets order the methods identically, 0 means no relation".
- NSD (same paragraph): "the share of the predicted and reference boundaries that lie within 2 pixels of each other".
- Stratified agreement (Methods heading paragraph): open with "Stratified agreement ranks methods within their own group first and then pools them." before the algebra.
- Method bootstrap and hierarchical bootstrap: defined in Methods; OK. Simplify "hierarchical intervals therefore sit partly below the estimate" to "hierarchical intervals are shifted towards zero, so they understate agreement".
- Oracle threshold: defined; add what calibration means: "how much a model's output probabilities differ in level from other models'".
- Seed average: defined (Methods, Training seeds). OK.
- Simpson's paradox: explained in the Introduction. OK.
- "the floor" (Results: "Both groups score above the floor on most non-wound datasets"): define as "near-zero Dice" at first use; the word is otherwise only explained in Limitations.
- "reliability (intraclass correlation)" (Results, Rankings among modern models): undefined; say "how well a ranking would repeat with fresh training seeds (intraclass correlation)".
- Split-half reliability with Spearman-Brown correction: add "an estimate of how well the full dataset would rank the methods again with new images".
- "wound-pair agreement": used from Results onwards; define in Methods as "mean agreement over the 6 pairs of wound datasets".

**m4. Terms used inconsistently.**
- "modern models" (26 lines) vs "modern methods" (abstract "0.21 among the modern methods"; Introduction contribution 1 "Pooling untrained with modern methods"; Results "pooled untrained and modern methods"; Conclusions "Pooling classical and modern methods"). Use "modern models" for the M19 group throughout, "methods" as the umbrella term.
- "untrained" (main term) vs "classical" (abstract twice, Discussion "whether classical methods are in the pool", Conclusions "Pooling classical and modern methods"). Define once "untrained classical methods" and then use "untrained" only.
- "method set" vs "pooled set" vs "pool" vs "method pool" (title) vs "pooled methods" (abstract). The Methods define "method sets" (groups and pooled sets) but the title's "method pool" is never tied to them. Add to the Introduction: "the method pool, that is the set of methods over which agreement is computed", and use "method set" in Methods and Results.
- Pre-registered rule: "One reading rule was fixed" (Methods), "The rule fixed before the seed experiment" (Results), "the criterion fixed in advance" (Discussion). Use "the pre-specified rule" each time.
- Control: "within-dataset control", "AZH train slice (control)", "disjoint AZH slice", "the AZH control", "AZH and its train slice" (Fig. 3). Pick "AZH control slice".
- "validity" vs "benchmark agreement" vs "validity figure": state once in the Introduction that the paper calls a benchmark's rank agreement with clinical data its "measured validity".
- Fig. S1 (manuscript) vs Figure S1 (OR1): harmless, but use "Fig. S1" in both.

**m5. Sentences too long or ambiguous** (split or reword):
- Methods, From-scratch training: "one decision threshold (0.90) was swept for the logistic regression on images 250 to 399 and applied to all three". Rewrite: "The threshold of one logistic regression was tuned on images 250 to 399 (best value 0.90) and this value was used for all three pixel classifiers."
- Methods, Method groups, 66-word sentence listing the 14 from-scratch methods: split after "117,681 parameters".
- Methods, Training seeds paragraph, 63-word sentence on GPU sessions and versions: move versions into its own sentence.
- Results, "Synthetic benchmark validity": "it agrees with each wound dataset less than the other wound datasets do, by −0.17 on average" is a double negative. Write "the difference (benchmark minus the other wound datasets) is −0.17 on average (−0.38 to −0.02)".
- Results, texture effect: "only 1 of the 16 per-reference intervals excludes zero, about what chance would give" : write "about the 1 in 20 expected by chance".
- Results, thresholds: "the mean Spearman correlation between the method means on the wound datasets and each model's own tuned threshold is 0.27 ..." : write "how well a model's tuned threshold predicts its rank: correlation 0.27 under the main protocol and 0.55 at a common threshold of 0.5".
- Results, choosing a model: "U-Net (EfficientNet-B3) (mean loss 0.067)": double brackets; write "U-Net with EfficientNet-B3, mean loss 0.067".
- Results, misses: "(0.79 against 0.23, 0.36 against 0.05 and 0.15 against 0.09)": say the tercile order ("from the smallest to the largest lesions").
- Results, Robustness, 72-word sentence starting "In all 12 conditions": split after the bracket.
- Results, modality in modern models: seed-0 value "0.46" has no p, while seed 1 and seed 2 do. Add the p or say why it is omitted.
- Introduction: "four homogeneous groups": "homogeneous" is unclear; write "four groups that each share one training regime".

**m6. Layout.**
- Page 10 is half empty: Table 4 (with keepNext caption) is pushed to page 11 after Fig. 2. Acceptable for review; could be avoided by placing Table 4 before Fig. 2.
- Body text is fully justified, giving wide word gaps in the formula paragraph on page 6. Left alignment would read better and avoids the gaps.
- Tables 1 to 5 fit the page width; headers are readable at 8 pt; no orphaned headings found; figures placed after their first citation; captions below figures and above tables.
- Table 3 footnote: convert to a superscript letter on the "Within modality" header ("a Different sources only") and move "smallest attainable p" into the caption.

**m7. Online Resource files.**
- Supply OR1 as PDF (ESM_1.pdf) and OR2 as ESM_2.xlsx; add the journal name under the title in both. For the final (unblinded) versions add author names and corresponding-author details as JIIM asks.
- OR2: negative values display with ASCII hyphen-minus in Excel. If the U+2212 rule must hold here too, set the number format to `0.000;"−"0.000`.
- OR2 README "Method sets" line defines sets circularly ("M (11 methods: M)"), and omits U+M11s and ALL57 which OR1 lists. Align the two lists.
- OR1 core properties say "generated by python-docx" with 2013 dates. Not identifying; set the title and current dates for tidiness.

**m8. Data availability.** "All clinical datasets are publicly available from their original distributors (Table 1)": Table 1 gives sources but no access points. Point to references [30 to 39]. The synthetic generators and benchmark images are not mentioned; reviewers will ask. State where they will be released (an anonymised repository link for review is acceptable).

## 5. What the author must supply separately

1. Title page (full version): title, all author names, affiliations (department, institution, city, country), corresponding author e-mail, ORCID iDs, Acknowledgements.
2. Statements and Declarations on the title page: Funding (agency and grant number, or "No funding was received"), Competing interests (ICMJE form; a "no competing interests" statement if none), Author contributions. Repeat Ethics approval, Consent, Data availability there as well.
3. AI-use statement, if AI tools were used beyond copy editing (see M3); it belongs in Methods, not only on the title page.
4. Cover letter: what the paper shows and why it suits JIIM, confirmation that it is not under consideration elsewhere, any preprint of this manuscript, and that the blinded reference 60 is the author's own earlier preprint (so the editor can see it).
5. Unblinded manuscript version with reference 60 restored, and unblinded Online Resources with author details.
6. Separate figure files named Fig1, Fig2, Fig3 (EPS or TIFF at the required resolution, Arial or Helvetica), and the graphics program used (Matplotlib 3.10.9) if the submission form asks.

## 6. Limitations of this review

- The JIIM ethics-and-disclosures page could not be read (HTTP 429), and the guidelines page was read through a summarising fetch with short quotes. Exact wording of the Index Medicus and ESM rules should be checked once on the live page before upload.
- Line and font sizes in figures were judged from the 300 dpi PNGs, not measured from the vector files.
- Numbers were not re-checked against the results JSON; this review covered compliance, language and layout only.
- The Online Resource 1 PDF used here was converted locally with LibreOffice; Word may paginate it differently.
