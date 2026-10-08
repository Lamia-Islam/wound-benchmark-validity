// Response to the review report, one row per comment.
const fs = require("fs");
const { d, p, h1, h2, table, rule } = require("./lib");
const { Document, Packer, Paragraph, TextRun, AlignmentType } = d;
const FONT = "Times New Roman";

const rows = [
  ["1", "Sec. 2.1: threshold protocol for the pixel classifiers.",
   "Stated explicitly for every group. The three pixel classifiers share one threshold (0.90), swept on FUSeg training images 250 to 399, which are disjoint from the 250 images they were fitted on. The small U-Net uses a fixed 0.5 with no sweep. The ten modern architectures each have their own threshold (0.10 to 0.45), swept on FUSeg training images 400 to 549. No sweep touches an evaluation image.",
   "Sec. 3.2"],
  ["2", "Sec. 2.2: do Kvasir-SEG and Kvasir-sessile share images?",
   "They did: the sessile set is a subset of Kvasir-SEG, and the two 120-image samples shared 31 images. The Kvasir reference is now drawn only from images outside the sessile subset, so the two share none. Agreement between them changes from 0.952 to 0.950, so no conclusion moves.",
   "Sec. 3.3, Sec. 4.6"],
  ["3", "Sec. 2.3: empty ground-truth masks may inflate correlations.",
   "Empty-mask counts are now a column of Table 2 (FUSeg 4, AZH 10, AZH train 8, all others 0). There are no exact ties in any reference. Every agreement was recomputed with those images excluded. One pair moves materially (FUSeg against AZH, 0.660 to 0.368 under Dice) and this is reported. Every modality test stays significant (p at most 0.0036).",
   "Table 2, Sec. 3.5, Sec. 4.6, Sec. 5.5"],
  ["4", "Table 3: \"same source\" and \"same dataset\" are undefined.",
   "Four levels are now defined once and used everywhere: within dataset (two disjoint slices of one dataset), within source (one clinic or group, no shared images), within modality (one modality, different sources) and across modality. Kvasir and its sessile subset are within source. FUSeg and AZH, both from the AZH Wound and Vascular Center, are also within source, not within modality as before.",
   "Sec. 3.5, Table 3, Appendix A"],
  ["5", "Sec. 3.3: the 98% and 49% figures use different ceilings.",
   "The comparison has been redone as a full design: three method pools (original 22, 18 untrained, all 32) against both reference pairs, each normalised by its own ceiling over the same pool. With the references held fixed, validity falls from 99 to 60 percent (Medetec and FUSeg) and from 116 to 48 percent (Medetec and AZH) as the pool grows. The ceiling is therefore not what drives the drop. Absolute agreement is reported next to every percentage.",
   "Sec. 4.3, Table 6"],
  ["6", "Fig. 3: the two ceilings are not explicit.",
   "Figure 3 is redrawn with one panel per reference pair. Every pool has its own ceiling line, in the same shade as its curve and labelled with its value, so no point is read against the wrong line.",
   "Figure 3"],
  ["7", "Sec. 4.3: domain randomisation advice is uncited.",
   "Cited: Tobin et al. (2017) and Tremblay et al. (2018), with Prakash et al. (2019) for the structured variant. The paragraph now also states why evaluation differs from training: a benchmark should match the deployment distribution rather than exceed it.",
   "Sec. 5.4"],
];

const other = [
  "Clinical references grew from six sets in two modalities to eleven in four: CVC-ClinicDB, CVC-ColonDB and ETIS-LaribPolypDB (colonoscopy), PH2 (dermoscopy) and the TN3K test split (thyroid ultrasound) were added.",
  "Every result is reported for all 32 methods, for the 18 untrained methods and for the 14 trained methods separately. Because all trained methods were trained on FUSeg, this also separates the effect of training domain.",
  "Every result is computed under Dice and under normalised surface Dice.",
  "The modality test is an exact permutation over reference modality labels, so the dependence between pairs that share a reference is respected.",
  "Effects that appear only when trained and untrained methods are pooled are reported as such. The earlier framing reversal and texture dose-response do not survive within pools; the weakening of the boundary-metric preference with lesion fraction does.",
];

const kids = [
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({
    text: "Response to the review report", font: FONT, size: 32, bold: true })] }),
  p("How the Clinical Reference and the Method Pool Determine Synthetic Benchmark Validity in Medical Image Segmentation", { italics: true, after: 200 }),
  rule(200),
  p("Thank you for the careful reading. Every comment has been addressed in the revised manuscript. Section, table and figure numbers below refer to the revised version."),
  h2("Point-by-point response"),
  table(["#", "Comment", "Response", "Where"], rows, [400, 2200, 5260, 1500],
        { leftCols: 4, size: 18, keep: false }),
  h2("Other changes in this revision"),
  ...other.map(t => p("\u2022  " + t, { indent: { left: 360, hanging: 240 }, after: 90 })),
];

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 22 } } } },
  sections: [{ properties: { page: { margin: { top: 1300, bottom: 1300, left: 1300, right: 1300 } } },
               children: kids }]
});
Packer.toBuffer(doc).then(b => {
  fs.writeFileSync("response_to_review.docx", b);
  console.log("wrote response_to_review.docx");
});
