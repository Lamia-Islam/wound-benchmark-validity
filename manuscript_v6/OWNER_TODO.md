# What Lamia must supply or decide before submission (JIIM)

The blinded manuscript, the unblinded build, both Online Resources and the figures are ready. The items below are not in the blinded files, by design, or need your decision.

## 1. Title page (separate file, not blinded)
- Full title (current title, chosen from reviewer R2's suggestion; see item 9).
- All author names, affiliations (department, institution, city, country), ORCID iDs.
- Corresponding author: name, postal address, e-mail.
- Acknowledgements (people and institutions, if any).

## 2. Statements and Declarations (on the title page)
- **Funding:** agency and grant number, or "No funding was received for conducting this study."
- **Competing interests:** your statement, plus the ICMJE disclosure form if the journal asks for it.
- **Author contributions:** who did what (for a single author: "L.I. designed the study, ran the analyses and wrote the manuscript", or your own wording).
- Ethics, consent and data availability are already in the manuscript. Repeat them on the title page as well.

## 3. AI-use sentence (in Methods; please confirm or reword)
Current wording, at the end of Materials and Methods: "Large language model assistants were used to write analysis code and to draft and edit the text; all code, numbers and text were checked by the authors." JIIM requires that this describe what was actually done. Change it if needed (for example, name the tool).

## 4. Code and data availability (decision needed)
The manuscript currently says: "The synthetic generators, the benchmark images and the analysis code are available from the corresponding author on reasonable request." If you can give a repository URL (an anonymised link is fine for review), send it and the sentence will be changed.

## 5. Cover letter
- What the paper shows and why it suits JIIM.
- That it is not under consideration elsewhere, and any preprint of this manuscript.
- **Reference 60 is your own earlier preprint** (Surgical Vision ROS2, Research Square 2026). In the blinded manuscript it reads "Reference removed for double-blind review"; tell the editor that it is your work. The unblinded build (`manuscript_v6_unblinded.docx`) restores it.

## 6. Unblinded Online Resources
JIIM asks that each supplementary file carry the article title, journal name, author names and corresponding-author details. The current Online Resources are blinded (title and journal only). Add the author block to the final, accepted versions.

## 7. Figure files
`figures/submission/Fig1-3` (.eps, .pdf, .tif at 1200 dpi). The font is Liberation Sans, which has the same metrics as Arial; no Arial is installed on the build machine. If the editorial office insists on Arial or Helvetica by name, re-render on a machine that has Arial. The graphics program is Matplotlib.

## 8. Items not done, for your judgement
- **JIIM reference style:** journal names are still in full, not Index Medicus abbreviations, and "et al." follows three authors. The guidelines ask for abbreviations and all authors; changing this needs verified metadata for every entry that is cut to three authors.
- **Preprints in the reference list** (every arXiv and Research Square entry, and the segmentation_models_pytorch GitHub repository): JIIM says only works accepted for publication should be listed. Ask the editorial office or move them to the text.
- **tremblay2018** pages (CVF pagination with the IEEE DOI) and **raina2023** (IEEE DOI missing): confirm on IEEE Xplore.
- **TN3K:** reviewer R5 suggests adding the Computers in Biology and Medicine journal paper. Its metadata were not confirmed, so it was not added.
- **Reviewer R2's optional new work** (not done): rescaling AZH images to FUSeg magnification and rescoring misses; more nnU-Net seeds; a colonoscopy-trained pretrained group; an image figure of example wound images.

## 9. Changes you should know about
- **Title** is now R2's: "Validating a synthetic segmentation benchmark by rank correlation: how the methods compared, the clinical reference and training noise change the answer in wound imaging". The previous title generalised from one generator. Revert it if you prefer.
- **Exclusions (new check).** Every AZH test and control image was matched against all FUSeg photographs. One rule removes copies or crops of photographs used elsewhere in the study:
  - **AZH test:** 4 images are crops of FUSeg training photographs and were removed, leaving 146. The 12 AZH crops of FUSeg validation photographs are **not** in our FUSeg reference (the first 120 validation images), so they were kept. HANDOFF.md had assumed they were.
  - **Control:** 72 images were removed, leaving 78.
  - The headline numbers moved only slightly.
- **"Confirmatory"** is now "primary", and the pre-specified seed rule is described as written after training but before the seed analysis.
