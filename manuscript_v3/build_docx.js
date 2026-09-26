const fs = require("fs");
const d = require("docx");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, PageOrientation, convertInchesToTwip
} = d;

const INK = "000000";
const GREY = "3F3F3F";          // dark enough to read
const RULE = "9A9A9A";
const HEADSHADE = "EDEDED";
const W = 9360;                 // 6.5in content width in DXA

const FONT = "Times New Roman";

function p(text, o = {}) {
  return new Paragraph({
    alignment: o.align || AlignmentType.JUSTIFIED,
    spacing: { after: o.after === undefined ? 140 : o.after, line: 276 },
    indent: o.indent,
    children: [new TextRun({
      text, font: FONT, size: o.size || 22, bold: o.bold, italics: o.italics,
      color: o.color || INK
    })]
  });
}

function rich(runs, o = {}) {
  return new Paragraph({
    alignment: o.align || AlignmentType.JUSTIFIED,
    spacing: { after: o.after === undefined ? 140 : o.after, line: 276 },
    children: runs.map(r => new TextRun({
      text: r.t, font: FONT, size: r.size || o.size || 22, bold: r.b,
      italics: r.i, color: r.c || o.color || INK
    }))
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 340, after: 160 },
    children: [new TextRun({ text, font: FONT, size: 26, bold: true, color: INK })]
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 260, after: 120 },
    children: [new TextRun({ text, font: FONT, size: 23, bold: true, color: INK })]
  });
}

function cell(text, o = {}) {
  return new TableCell({
    width: { size: o.w, type: WidthType.DXA },
    shading: o.head ? { type: ShadingType.CLEAR, fill: HEADSHADE, color: "auto" } : undefined,
    margins: { top: 70, bottom: 70, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: o.align || (o.head ? AlignmentType.LEFT : AlignmentType.LEFT),
      spacing: { after: 0, line: 240 },
      children: [new TextRun({
        text: String(text), font: FONT, size: 19,
        bold: o.head || o.bold, color: INK
      })]
    })]
  });
}

function table(header, rows, widths, opts = {}) {
  const bold = opts.boldRows || [];
  return new Table({
    columnWidths: widths,
    width: { size: W, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 6, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "D5D5D5" },
      insideVertical: { style: BorderStyle.NONE }
    },
    rows: [
      new TableRow({
        tableHeader: true,
        children: header.map((t, i) => cell(t, {
          w: widths[i], head: true,
          align: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT
        }))
      }),
      ...rows.map((r, ri) => new TableRow({
        children: r.map((t, i) => cell(t, {
          w: widths[i], bold: bold.includes(ri),
          align: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT
        }))
      }))
    ]
  });
}

function tableCaption(n, text) {
  return new Paragraph({
    spacing: { before: 170, after: 90, line: 260 },
    children: [
      new TextRun({ text: `Table ${n}. `, font: FONT, size: 19, bold: true, color: INK }),
      new TextRun({ text, font: FONT, size: 19, color: GREY })
    ]
  });
}

function figure(file, w, h, n, caption) {
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 200, after: 80 },
      children: [new ImageRun({
        type: "png", data: fs.readFileSync(file),
        transformation: { width: w, height: h }
      })]
    }),
    new Paragraph({
      alignment: AlignmentType.LEFT,
      spacing: { after: 220, line: 260 },
      children: [
        new TextRun({ text: `Figure ${n}. `, font: FONT, size: 19, bold: true, color: INK }),
        new TextRun({ text: caption, font: FONT, size: 19, color: GREY })
      ]
    })
  ];
}

const body = [];

// ------------------------------------------------------------------ header
body.push(new Paragraph({
  alignment: AlignmentType.LEFT,
  spacing: { after: 150, line: 300 },
  children: [new TextRun({
    text: "Synthetic benchmark validity in medical image segmentation depends "
        + "on the clinical reference and the method pool",
    font: FONT, size: 30, bold: true, color: INK
  })]
}));
body.push(rich([{ t: "Lamia Islam" }], { align: AlignmentType.LEFT, after: 40 }));
body.push(rich([{
  t: "Department of Information and Communication Engineering, Pabna University "
   + "of Science and Technology, Pabna, Bangladesh", size: 19, c: GREY
}], { align: AlignmentType.LEFT, after: 30 }));
body.push(rich([{ t: "ORCID 0009-0006-2866-5022", size: 19, c: GREY }],
  { align: AlignmentType.LEFT, after: 60 }));
body.push(new Paragraph({
  spacing: { after: 260 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 8 } },
  children: [new TextRun({ text: "", font: FONT, size: 2 })]
}));
body.push(rich([
  { t: "Draft for supervisory review. ", b: true, size: 19 },
  { t: "This document contains the introduction, methodology, results and "
     + "discussion. Abstract, related work, references and declarations follow "
     + "in the full submission draft.", size: 19, c: GREY }
], { after: 260 }));

// ------------------------------------------------------------ 1 introduction
body.push(h1("1. Introduction"));

body.push(p(
  "Rankings of segmentation algorithms are not stable. Maier-Hein and colleagues "
+ "analysed 150 biomedical image analysis challenges in Nature Communications in "
+ "2018 and showed that an algorithm's rank moves with the test cases used, with "
+ "the ranking scheme, with the annotators, and with how per-case scores are "
+ "aggregated. Metrics Reloaded, in Nature Methods in 2024, subsequently gave the "
+ "field a framework for choosing metrics appropriate to a problem."));

body.push(p(
  "That work establishes that rankings are fragile and names the sources. It does "
+ "not answer the question a practitioner actually faces, which is what one would "
+ "have to measure about a benchmark in order to know how far to trust it. This "
+ "study asks that question for synthetic benchmarks in medical image "
+ "segmentation, and reports that the answer has three parts rather than one."));

body.push(p(
  "The first part is the clinical reference. We show that the agreement between "
+ "two clinical datasets about which method is better is not a fixed quantity. It "
+ "falls from +0.944 between two disjoint halves of one dataset, to +0.563 on "
+ "average between different datasets of the same modality, to no detectable "
+ "agreement between modalities. Because the within-dataset value establishes that "
+ "the measurement reproduces itself, the decline is genuine disagreement between "
+ "references rather than noise. Any claim of the form \"this benchmark agrees "
+ "with clinical data at rho = x\" is therefore incomplete until it says which "
+ "clinical data."));

body.push(p(
  "The second part is the method pool, and this is the result we did not expect. "
+ "The same synthetic benchmark, scored against the same clinical references, "
+ "reaches 98 percent of the clinical-to-clinical ceiling when validity is "
+ "assessed over a pool of 21 classical algorithms and one small network, and 49 "
+ "percent of it when eleven modern architectures are added. A benchmark that "
+ "appears nearly perfect for choosing between thresholding methods is roughly "
+ "half as good at ordering the methods a practitioner would deploy. We also show "
+ "that in a pool mixing trained and untrained methods, a cross-domain rank "
+ "comparison measures which methods had domain-matched training rather than "
+ "anything about the data, and that this can invert the sign of the reported "
+ "correlation."));

body.push(p(
  "The third part is the benchmark data itself, and here the result is negative in "
+ "a way we think is useful. We built a measurable background-texture statistic as "
+ "a candidate design specification, calibrated it against clinical images, and "
+ "tested it by constructing benchmarks from three generator families matched on "
+ "it. Benchmarks matched to within 0.03 on that statistic differ by 0.115 in "
+ "validity. The spread between generator families at a fixed value of the "
+ "statistic is as large as the entire within-family effect of the statistic. The "
+ "reduction fails."));

body.push(p(
  "Within a single modality, one property of the reference does predict the "
+ "direction of its bias. The fraction of the frame occupied by the lesion "
+ "determines whether a reference rewards high or low lesion-to-surround contrast, "
+ "monotonically, and the relationship holds across four generator seeds, under "
+ "both an overlap metric and a boundary metric, and over the enlarged method "
+ "pool. A clinical dataset that took no part in constructing the relationship "
+ "falls on it with the correct sign. We report the relationship and explicitly "
+ "decline to report its crossover point as a portable constant, because that "
+ "point moves with the pool, the reference and the metric."));

body.push(h2("1.1 Contributions"));
body.push(p("1. A measured hierarchy of rank agreement between clinical "
+ "references, from within-dataset to across-modality, anchored by a "
+ "within-dataset control that makes the decline interpretable.", { indent: { left: 240 } }));
body.push(p("2. Evidence that benchmark validity depends on the method pool it "
+ "is assessed over, with the size of the effect quantified.", { indent: { left: 240 } }));
body.push(p("3. A demonstration that cross-domain rank comparison in a mixed "
+ "pool measures training-domain match, with a case in which this inverts the "
+ "sign of the reported correlation.", { indent: { left: 240 } }));
body.push(p("4. A tested and failed reduction of benchmark validity to a single "
+ "measured image statistic, with the residual quantified across three generator "
+ "families.", { indent: { left: 240 } }));
body.push(p("5. A within-modality relationship between a reference's lesion "
+ "fraction and the direction of its design preference, confirmed across seeds, "
+ "metrics and pool composition.", { indent: { left: 240 } }));

// ------------------------------------------------------------- 2 methodology
body.push(h1("2. Methodology"));

body.push(h2("2.1 Method pool"));
body.push(p(
  "Thirty-two segmentation methods in eight families. Every method runs through "
+ "one interface with one post-processing step, so no part of the ranking reflects "
+ "differences in how outputs were cleaned up."));

body.push(tableCaption(1, "Composition of the 32-method pool."));
body.push(table(
  ["Family", "n", "Members"],
  [
    ["Background-referenced", "5", "adaptive Gaussian, adaptive mean, Otsu on grey, BGR and HSV corner distance"],
    ["Colour-referenced", "4", "Otsu on a*, 92nd percentile of a*, redness ratio, Otsu on saturation"],
    ["Clustering", "4", "k-means in Lab at k = 3, 4, 5; mean shift with Otsu on a*"],
    ["Region and contour", "5", "watershed, GrabCut, flood fill, Chan-Vese, Canny with fill"],
    ["Pixel classifiers", "3", "logistic regression (all and colour-only), gradient boosting"],
    ["Small CNN", "1", "three-level U-Net, 117,681 parameters"],
    ["Deep encoder-decoder", "5", "U-Net, U-Net++, LinkNet on ResNet-18; U-Net on MobileNetV2 and EfficientNet-B0"],
    ["Deep other", "5", "DeepLabV3+, PSPNet, FPN, MAnet on ResNet-18; SegFormer on MiT-B0"]
  ],
  [2400, 600, 6360]
));

body.push(p(
  "Twenty-one of the methods are classical algorithms with no learned parameters. "
+ "They span a wide range of segmentation principles at a wide range of "
+ "performance, which gives the rank correlations something to order, and being "
+ "untrained they carry no dependence on any dataset.", { after: 100 }));

body.push(p(
  "Eleven methods are trained. Three pixel classifiers are fitted on 264,292 "
+ "pixels sampled from 250 FUSeg training images, balanced between lesion and "
+ "non-lesion. The remaining eight architectures, plus the small U-Net, are "
+ "trained on 393 FUSeg training images at 128 by 128 pixels. The ten modern "
+ "architectures were chosen to vary along the two axes that modern segmentation "
+ "varies along: seven decoders sharing one encoder, and three alternative "
+ "encoders including a transformer."));

body.push(rich([
  { t: "No pretrained weights are used anywhere in this study. ", b: true },
  { t: "Every architecture is randomly initialised, so no method enters the pool "
     + "carrying knowledge of an external dataset and the pool is not confounded "
     + "by differences in pretraining corpora. All ten see identical data under an "
     + "identical protocol, copied from the small U-Net so that new and old "
     + "methods are comparable: Dice plus binary cross-entropy, Adam, horizontal "
     + "flip augmentation, batch size 8, 30 epochs. The single deliberate "
     + "difference is a learning rate of 1e-3 rather than 3e-3, because 3e-3 with "
     + "Adam is not stable for models of 14 to 22 million parameters." }
]));

body.push(p(
  "Decision thresholds are swept per model on FUSeg training images 400 to 550, a "
+ "slice disjoint from the 393 used for training and from every evaluation split. "
+ "Held-out Dice at the selected threshold ranges from 0.507 to 0.641, so the pool "
+ "has genuine spread rather than ten near-identical models."));

body.push(h2("2.2 Clinical reference sets"));
body.push(tableCaption(2, "The five clinical reference sets, from three independent sources."));
body.push(table(
  ["Reference", "n scored", "Modality", "Median lesion fraction"],
  [
    ["Medetec", "152", "Wound photography", "0.102"],
    ["FUSeg validation", "120", "Wound photography", "0.011"],
    ["AZH test", "150", "Wound photography", "0.029"],
    ["AZH train slice", "150", "Wound photography", "0.026"],
    ["Kvasir-SEG", "120", "Colonoscopy", "0.106"],
    ["Kvasir-SEG sessile", "120", "Colonoscopy", "0.043"]
  ],
  [2700, 1400, 3060, 2200]
));

body.push(p(
  "The AZH train slice is a disjoint 150-image slice of the same dataset as AZH "
+ "test and exists solely as a within-dataset control. FUSeg and AZH come from the "
+ "same clinic, since the Foot Ulcer Segmentation Challenge was built from AZH "
+ "Wound and Vascular Center data. Template matching 25 AZH patches against all "
+ "1,010 FUSeg frames at six scales gives a maximum normalised cross-correlation "
+ "of 0.927 with no exact matches, so they are distinct images under different "
+ "curation conventions rather than re-crops of one another. We treat them as one "
+ "source throughout. Kvasir-SEG and its sessile subset likewise share a source."));

body.push(p(
  "Mask conventions differ and the difference matters. Medetec masks are 0/1 and "
+ "FUSeg masks 0/255, so wound masks are binarised on any non-zero value; a fixed "
+ "mid-range threshold silently empties the Medetec masks and produces a Dice of "
+ "exactly zero for every method. Kvasir masks are JPEG-compressed and carry "
+ "anti-aliased edge values, so binarising on any non-zero value inflates every "
+ "lesion; Kvasir is thresholded at 127."));

body.push(h2("2.3 Synthetic benchmarks"));
body.push(p(
  "Three generator families, built on different principles, each producing images "
+ "with ground-truth masks and exposing its design properties as explicit "
+ "parameters. Parametric generators are used rather than a learned generative "
+ "model because the ablation and dose-response experiments require individual "
+ "design properties to be switched off and swept independently, and a diffusion "
+ "model does not offer those controls."));

body.push(p(
  "Generator A places a lesion on a uniform drape background and has no background "
+ "texture parameter. Generator B renders textured skin with a shared luminance "
+ "mottling field, a clinically calibrated tissue palette and clutter such as hair "
+ "and gauze; its texture amplitude is the swept parameter. Generator C shares no "
+ "code path with generator B and differs in every mechanism: spectral-filtered "
+ "noise rather than upsampled Gaussian mottling, a radial vignette rather than a "
+ "linear ramp, distance-field alpha blending rather than filled polygons, and a "
+ "CIELAB Gaussian mixture rather than an HSV offset from skin tone."));

body.push(p(
  "We describe these as constructed on different principles rather than as "
+ "independently constructed. They were written by the same author and share "
+ "whatever unstated assumptions we hold about what a wound photograph looks like. "
+ "Generator C exists specifically to break the shared-implementation assumption."));

body.push(p(
  "The background texture statistic is the median local standard deviation of "
+ "shading-corrected luminance in a 9-pixel window, measured over background "
+ "pixels only. Every image is resampled to a canonical 512 pixels before "
+ "measurement. This is not cosmetic: measured natively, Medetec at 224 pixels "
+ "reads 5.07 and FUSeg at 512 pixels reads 3.88, while resampled to a common 512 "
+ "they read 2.69 and 3.88, which reverses their order. A statistic whose ordering "
+ "of two datasets depends on the resolution they happen to be distributed at is "
+ "not a property of the images."));

body.push(h2("2.4 Statistical protocol"));
body.push(p(
  "Agreement is the Spearman rank correlation between two vectors of per-method "
+ "mean Dice over the same method pool, with two-sided permutation p-values at "
+ "20,000 permutations and confidence intervals by bootstrap over methods at 5,000 "
+ "resamples. At n = 32 methods a Spearman requires approximately |rho| of 0.35 to "
+ "reach p < 0.05; at n = 22 the requirement is 0.43. Where we report an absence of "
+ "agreement we state it as an absence of detectable agreement at this sample size."));

body.push(p(
  "Every relationship claimed in this paper is computed twice: with Dice, and with "
+ "normalised surface Dice at a 2-pixel tolerance. The second is included because "
+ "Dice carries a region-size bias and the framing experiment varies region size, "
+ "so the two would otherwise be confounded by construction. Both metrics treat "
+ "two empty masks as perfect agreement."));

body.push(p(
  "Every synthetic benchmark used in a reported relationship is built at four "
+ "generator seeds and the relationship is computed independently for each. "
+ "Single-seed results overturned two conclusions during this work and are not "
+ "reported as findings anywhere in this paper."));

body.push(p(
  "A reference's preference over a design property is the slope of its rank "
+ "agreement against the measured value of that property, fitted by ordinary least "
+ "squares over the sweep and expressed per 10 units. An earlier version of this "
+ "work used the Spearman correlation of agreement against the property instead. "
+ "That statistic saturates: once the underlying relationship is monotone it "
+ "returns exactly +1 or -1 regardless of magnitude, and at n = 32 it returned "
+ "exactly +1.000 for five of eight conditions where the underlying agreements "
+ "ranged from -0.33 to +0.36. The slope is continuous and does not saturate."));

// ---------------------------------------------------------------- 3 results
body.push(h1("3. Results"));

body.push(h2("3.1 Rank agreement falls in steps, and the decline is real"));
body.push(tableCaption(3, "Pairwise rank agreement between clinical references over the 32-method pool."));
body.push(table(
  ["Pair", "rho", "p", "95% CI", "Relationship"],
  [
    ["Kvasir vs Kvasir-sessile", "+0.952", "<0.0001", "[+0.871, +0.982]", "Same source"],
    ["AZH test vs AZH train", "+0.944", "<0.0001", "[+0.835, +0.984]", "Same dataset"],
    ["FUSeg vs AZH", "+0.660", "0.0001", "[+0.430, +0.804]", "Same modality"],
    ["Medetec vs AZH", "+0.568", "0.0012", "[+0.256, +0.781]", "Same modality"],
    ["Medetec vs FUSeg", "+0.461", "0.0081", "[+0.085, +0.758]", "Same modality"],
    ["Medetec vs Kvasir", "+0.019", "0.9172", "[-0.318, +0.360]", "Across modality"],
    ["Medetec vs Kvasir-sessile", "-0.042", "0.8185", "[-0.397, +0.338]", "Across modality"],
    ["AZH vs Kvasir", "-0.191", "0.2991", "[-0.506, +0.164]", "Across modality"],
    ["AZH vs Kvasir-sessile", "-0.277", "0.1289", "[-0.579, +0.081]", "Across modality"],
    ["FUSeg vs Kvasir", "-0.477", "0.0065", "[-0.706, -0.152]", "Across modality"],
    ["FUSeg vs Kvasir-sessile", "-0.552", "0.0011", "[-0.741, -0.224]", "Across modality"]
  ],
  [2700, 1000, 1100, 2360, 2200]
));

body.push(rich([
  { t: "The within-dataset control is what makes this interpretable. ", b: true },
  { t: "Splitting one dataset into two disjoint 150-image halves and scoring each "
     + "independently gives +0.944. The measurement therefore reproduces itself at "
     + "0.94, and the drop to 0.56 between different datasets of the same modality "
     + "is genuine disagreement between references rather than sampling noise. All "
     + "three same-modality pairs are significant with confidence intervals "
     + "excluding zero; over the earlier 22-method pool the weakest of them sat at "
     + "p = 0.0385 with an interval touching zero, so the enlarged pool "
     + "strengthened this structure rather than weakening it." }
]));

body.push(...figure("fig1_hierarchy.png", 560, 407, 1,
  "Pairwise rank agreement between clinical references, ordered by relationship, "
+ "with 95 percent bootstrap intervals over methods. Filled markers are "
+ "significant at p < 0.05. The shaded band marks across-modality pairs."));

body.push(h2("3.2 The apparent cross-modality inversion is a composition artifact"));
body.push(p(
  "The pooled across-modality mean of -0.253 invites the reading that methods good "
+ "at wound segmentation are worse than chance at colonoscopy. That reading is "
+ "wrong, and the way it fails is itself a result."));

body.push(tableCaption(4, "Mean agreement by method subset. The negative value exists only in the mixed pool."));
body.push(table(
  ["Subset", "Same modality", "Across modality"],
  [
    ["All 32 methods", "+0.563", "-0.253"],
    ["22 classical and pixel-classifier methods", "+0.511", "+0.108"],
    ["10 modern architectures", "+0.321", "+0.236"]
  ],
  [4560, 2400, 2400], { boldRows: [0] }
));

body.push(tableCaption(5, "Mean Dice by subset on each reference. The two groups exchange places between modalities."));
body.push(table(
  ["Reference", "10 deep", "22 others", "Gap"],
  [
    ["Medetec", "0.374", "0.295", "+0.079"],
    ["FUSeg", "0.581", "0.132", "+0.449"],
    ["AZH", "0.396", "0.330", "+0.066"],
    ["Kvasir", "0.062", "0.200", "-0.138"],
    ["Kvasir-sessile", "0.024", "0.114", "-0.091"]
  ],
  [3360, 2000, 2000, 2000]
));

body.push(p(
  "The deep models were trained on wound photography. They beat the untrained "
+ "methods on all three wound references and collapse below them on both endoscopy "
+ "references. Two groups exchanging places between domains produces a negative "
+ "pooled rank correlation without any individual pair of methods inverting."));

body.push(p(
  "Two claims follow. First, within a homogeneous method pool there is no "
+ "detectable cross-modality rank agreement: +0.108 and +0.236, neither "
+ "significant, at a detection limit of 0.35. Second, and this is the transferable "
+ "warning, in a pool mixing trained and untrained methods a cross-domain rank "
+ "comparison measures which methods had domain-matched training. Its sign is set "
+ "by pool composition. A practitioner comparing benchmarks across domains with "
+ "such a pool will measure this artifact and can readily mistake it for a "
+ "property of the data."));

body.push(...figure("fig2_composition.png", 600, 259, 2,
  "Left: mean agreement within and across modality for the full pool and for each "
+ "subset; only the mixed pool goes negative. Right: mean Dice of each subset on "
+ "the five references, showing the exchange of places between modalities."));

body.push(h2("3.3 Benchmark validity depends on the method pool"));
body.push(p(
  "The clinical-to-clinical ceiling, measured on the two references no method was "
+ "trained on, is Medetec against AZH at +0.568 (p = 0.0012). Generator B against "
+ "those same references, over the 32-method pool:"));

body.push(tableCaption(6, "Generator B against the neutral clinical references, 32-method pool."));
body.push(table(
  ["Texture amplitude", "Measured texture", "vs Medetec", "vs AZH", "Mean", "% of ceiling"],
  [
    ["11", "3.45", "+0.234", "+0.058", "+0.146", "26%"],
    ["28", "4.30", "+0.366", "+0.157", "+0.261", "46%"],
    ["40", "5.07", "+0.360", "+0.194", "+0.277", "49%"]
  ],
  [1900, 1800, 1500, 1300, 1400, 1460]
));

body.push(p(
  "Over the earlier 22-method pool, and against Medetec and FUSeg, the best "
+ "configuration of the same generator reached 98 percent of the then-measured "
+ "ceiling. Two changes account for the gap and both matter. FUSeg is excluded "
+ "here because eleven methods were trained on its training split. And the pool "
+ "now contains modern architectures whose relative ordering the synthetic "
+ "benchmark does not reproduce."));

body.push(rich([
  { t: "A synthetic benchmark that appears to reach the clinical ceiling when "
     + "assessed over classical methods reaches about half of it once the methods "
     + "people deploy are in the pool. ", b: true },
  { t: "This is the most practically consequential result in the study. It says "
     + "that validity figures for synthetic benchmarks are not portable across "
     + "method pools, and that assessing validity over convenient classical "
     + "baselines will overstate it. The texture dose-response itself survives the "
     + "pool change: agreement rises monotonically with background texture, which "
     + "is the direction reported over the smaller pool." }
]));

body.push(...figure("fig3_pool_dependence.png", 580, 367, 3,
  "Generator B agreement with the neutral references as a function of measured "
+ "background texture, against the clinical-to-clinical ceiling. The open triangle "
+ "marks the same generator assessed over the 22-method pool with FUSeg included."));

body.push(h2("3.4 Within one modality, lesion fraction predicts the direction of bias"));
body.push(p(
  "The primary curve is built from AZH, the wound reference no method was trained "
+ "on. AZH frames were cropped around their own lesions to seven target lesion "
+ "fractions. Every crop uses the same 140 frames and the same annotations, so "
+ "lesion fraction is the only thing that differs between the seven references."));

body.push(tableCaption(7, "Contrast verdict against reference lesion fraction, AZH crops, four seeds."));
body.push(table(
  ["Lesion fraction", "Verdict, Dice", "sd", "Verdict, NSD", "sd"],
  [
    ["0.0353", "+0.077", "0.006", "+0.080", "0.005"],
    ["0.0550", "+0.029", "0.009", "+0.061", "0.005"],
    ["0.0801", "-0.002", "0.010", "+0.038", "0.005"],
    ["0.1101", "-0.037", "0.009", "+0.004", "0.005"],
    ["0.1598", "-0.060", "0.008", "-0.041", "0.005"],
    ["0.2194", "-0.078", "0.007", "-0.056", "0.006"],
    ["0.2981", "-0.086", "0.007", "-0.079", "0.007"]
  ],
  [2200, 2000, 1500, 2000, 1660]
));

body.push(rich([
  { t: "Monotonicity against lesion fraction is Spearman -1.000 at p = 0.0005 in "
     + "every one of the four seeds, under both metrics. ", b: true },
  { t: "Eight of eight seed and metric combinations are perfectly monotonic, and "
     + "the seed standard deviation never exceeds 0.010 against a curve range of "
     + "0.163. The boundary metric is the load-bearing control here: Dice is "
     + "biased by region size and this experiment varies region size, so a "
     + "Dice-only result would be confounded by construction. The same "
     + "construction on FUSeg crops gives monotonicity between -0.881 and -1.000, "
     + "and is reported as a replication rather than as the primary result because "
     + "eleven of the 32 methods were trained on FUSeg training images. Removing "
     + "that contamination sharpened the relationship." }
]));

body.push(p(
  "The crossover between rewarding high and low contrast sits between lesion "
+ "fraction 0.110 and 0.160 in three of the four curve and metric combinations, "
+ "and between 0.055 and 0.080 for the AZH curve under Dice. It moves with the "
+ "reference, with the metric and with the method pool. The direction of the "
+ "relationship moves with none of them, and we report the crossover as an "
+ "interval observed under stated conditions rather than as a threshold for others "
+ "to apply."));

body.push(p(
  "Medetec, at lesion fraction 0.1024 and taking no part in the construction, is "
+ "predicted at -0.028 and measures +0.038 under Dice, and is predicted at +0.013 "
+ "and measures +0.056 under normalised surface Dice. The residuals are 41 and 27 "
+ "percent of the curve range. Across two independently built curves and three "
+ "held-out datasets, every real clinical dataset lands above the curve at its own "
+ "lesion fraction, four placements in one direction. The mechanism is visible: a "
+ "crop taken to reach lesion fraction f is not the same object as a dataset "
+ "natively framed at f, because cropping discards the surrounding limb and "
+ "background that a natively framed dataset retains. Crop series therefore "
+ "overstate the framing effect. The construction establishes direction and "
+ "ordering, which is what is claimed, and is not a calibrated predictor."));

body.push(...figure("fig4_framing_both.png", 600, 264, 4,
  "Contrast verdict against reference lesion fraction, on a log axis, for the "
+ "AZH-built curve and the FUSeg-built replication, under both metrics. Medetec is "
+ "overlaid as open predicted and filled measured markers joined by a residual stem."));

body.push(h2("3.5 Validity is not reducible to the measured texture statistic"));
body.push(p(
  "If benchmark validity were a function of the background texture statistic, two "
+ "benchmarks matched on it would agree."));

body.push(tableCaption(8, "Four benchmarks spanning three generator families, against the neutral references."));
body.push(table(
  ["Benchmark", "Measured texture", "vs Medetec", "vs AZH", "Mean"],
  [
    ["Generator A, uniform drape", "2.10", "-0.045", "-0.261", "-0.153"],
    ["Generator C, no noise", "2.34", "+0.116", "-0.102", "+0.007"],
    ["Generator C, noise 3.4", "3.20", "+0.132", "-0.097", "+0.017"],
    ["Generator B, texture ablated", "3.23", "+0.201", "+0.063", "+0.132"]
  ],
  [3360, 1800, 1500, 1300, 1400]
));

body.push(p(
  "Two matched pairs. Generator C at 3.20 and generator B at 3.23 are matched to "
+ "within 0.03 on the statistic and differ by 0.115 in validity. Generator A at "
+ "2.10 and generator C at 2.34 are matched to within 0.24 and differ by 0.160. "
+ "For scale, the entire effect of sweeping texture within generator B across its "
+ "clinical range is 0.131. The spread between generator families at a fixed value "
+ "of the statistic is as large as the whole within-family effect of the "
+ "statistic. The statistic captures something real, since texture predicts "
+ "validity within a family, but it is not a specification a benchmark designer "
+ "could target."));

body.push(...figure("fig5_nonreducibility.png", 580, 367, 5,
  "Validity against measured background texture for four benchmarks from three "
+ "generator families. The two bracketed pairs are matched on the statistic and "
+ "still disagree."));

body.push(h2("3.6 Cross-modality verdicts are not interpretable"));
body.push(p(
  "Kvasir-SEG's median lesion fraction is 0.106, close to Medetec's 0.102, so the "
+ "framing curve makes a prediction for it. Placed on the wound-built curve, "
+ "Kvasir's residual was 0.040 and the sessile subset's was 1.607 on a 1.93 scale: "
+ "one apparent bullseye and one of the largest misses the scale allows. Neither is "
+ "meaningful. The underlying agreements between the synthetic wound benchmarks and "
+ "the endoscopy references run from -0.19 to +0.13 with p-values from 0.39 to "
+ "0.96, so the verdict is a trend fitted through values not distinguishable from "
+ "zero. This is a direct consequence of Section 3.1: a verdict about which "
+ "synthetic design a reference prefers exists only where the benchmark and the "
+ "reference agree about method ranking at all. The framing result is scoped to "
+ "within-modality use for this reason, and not by assumption."));

// -------------------------------------------------------------- 4 discussion
body.push(h1("4. Discussion"));

body.push(h2("4.1 What a validity figure means"));
body.push(p(
  "The common form of a synthetic-benchmark validity claim is a single number: "
+ "this benchmark agrees with clinical data at rho = x. This study shows that "
+ "number is underspecified in three ways at once, and gives the size of each."));

body.push(p(
  "It depends on which clinical reference. Two datasets of the same modality agree "
+ "with each other at 0.46 to 0.66, against 0.94 for two halves of one dataset. "
+ "Quoting agreement against one reference tells the reader roughly half of what "
+ "they need. It depends on the method pool: the same generator against the same "
+ "references reaches 98 percent of ceiling over a mostly classical pool and 49 "
+ "percent over a pool including modern architectures, which is the largest of the "
+ "three dependences and the least discussed in the literature. And it does not "
+ "depend on any single measured property of the benchmark images in a way that "
+ "could be turned into a design specification, at least not for the most natural "
+ "candidate."));

body.push(p(
  "The practical recommendation is narrow and we think defensible. Report "
+ "benchmark agreement as a matrix over references rather than as a scalar, report "
+ "the method pool it was computed over, and include a within-dataset control so a "
+ "reader can tell how much of any decline is measurement noise. That control "
+ "costs one extra scoring run on a disjoint slice and is what makes every other "
+ "number in this study interpretable."));

body.push(h2("4.2 Why the method-pool dependence is not obvious"));
body.push(p(
  "It would be reasonable to expect a benchmark that orders twenty-two methods "
+ "correctly to order thirty-two correctly, since the added methods are simply "
+ "more points on the same scale. That is not what happens, for a reason visible "
+ "in Section 3.2: the added methods are not more of the same. They are trained "
+ "and the others are not, and a benchmark's ability to reproduce the ordering "
+ "within each group is not the same as its ability to reproduce the ordering "
+ "between them."));

body.push(p(
  "This generalises beyond the present setting. Any evaluation that mixes trained "
+ "and untrained methods, or methods trained on different corpora, is partly "
+ "measuring training-domain match. Where the comparison crosses domains, Section "
+ "3.2 shows this can dominate to the point of flipping the sign of the reported "
+ "correlation. Pretrained foundation models make this more pressing rather than "
+ "less, since their training corpora are large, opaque and unequal across "
+ "methods."));

body.push(h2("4.3 The framing result and what it is for"));
body.push(p(
  "Within a modality, one measurable property of a reference predicts the "
+ "direction of its design preference. A reference whose lesions fill little of "
+ "the frame rewards high lesion-to-surround contrast; one whose lesions fill a "
+ "lot rewards low contrast."));

body.push(p(
  "The use is diagnostic rather than prescriptive. A researcher who concludes from "
+ "one clinical reference that a particular synthetic design property is realistic "
+ "can check whether that conclusion would survive a reference framed differently, "
+ "and the direction of the change is predictable from a quantity anyone can "
+ "compute in one line. That is a check on a conclusion, not a recipe for building "
+ "a benchmark."));

body.push(p(
  "It is also worth stating the contrast with domain randomization, because a "
+ "reader's prior will be that literature. For training transfer, the established "
+ "advice is to maximise variation, randomising textures, lighting and placement "
+ "well beyond realism. Our validation results point the other way: within a "
+ "generator family, agreement rises monotonically with background texture towards "
+ "the clinical range. The two are not in conflict, they are different objectives. "
+ "A benchmark exists to order methods as the deployment distribution would, so it "
+ "should resemble that distribution. A training set exists to produce a method "
+ "robust to distributions it has not seen, so it should exceed it. Section 3.2 "
+ "gives an unusually clean illustration, in which models trained on one narrow "
+ "domain reach 0.58 mean Dice on it and 0.02 to 0.06 on another."));

body.push(h2("4.4 Limitations"));
body.push(p(
  "Five reference sets from three independent sources, and one of those sources "
+ "supplies both endoscopy references, so every cross-modality conclusion rests on "
+ "a single endoscopy collection. A second, independent endoscopy source would "
+ "materially strengthen Sections 3.1 and 3.2 and is the most valuable single "
+ "addition to this study.", { after: 100 }));

body.push(p(
  "Both modalities are colour photography of surface tissue under illumination. "
+ "Nothing here establishes behaviour for volumetric modalities such as CT or MRI, "
+ "where background texture and lesion fraction would need to be redefined.",
  { after: 100 }));

body.push(p(
  "The eleven trained methods are trained from random initialisation on 393 images "
+ "at 128 by 128 pixels. They are real architectures under a real and identical "
+ "protocol, which is what the study requires, but they are not the fully trained, "
+ "augmented, pretrained models a practitioner would deploy. Neither nnU-Net nor a "
+ "segmentation foundation model is included. We claim the pool spans modern "
+ "architectural variation, not that it reproduces state-of-the-art performance.",
  { after: 100 }));

body.push(p(
  "Every agreement value is a Spearman over 32 points, so reported absences of "
+ "agreement are absences of detectable agreement. The comparison of "
+ "same-modality against across-modality pairs is a Mann-Whitney over 3 against 6 "
+ "values, for which p = 0.0238 is the smallest attainable value; the individual "
+ "pair significances carry that result, not the group test.", { after: 100 }));

body.push(p(
  "The framing references are cropped using the ground-truth mask centroid and "
+ "area, which is oracle localisation that no deployable preprocessing pipeline "
+ "could reproduce. The crops are instruments for varying one property in "
+ "isolation, not proposed pipelines, and the real clinical datasets, curated "
+ "without any oracle, land in the predicted places. AZH is distributed as 224 "
+ "pixel patches at a median lesion fraction of 0.029 and cropping can only raise "
+ "a lesion fraction, so the primary curve starts at 0.035 and only one held-out "
+ "dataset falls inside its range.", { after: 100 }));

body.push(p(
  "Eleven of 32 methods were trained on FUSeg training images. The evaluation "
+ "images are disjoint but the distribution is not, and the resulting inflation is "
+ "large, which is why headline figures are reported against Medetec and AZH and "
+ "the primary framing curve is built from AZH crops.", { after: 100 }));

body.push(p(
  "Finally, the three generator families were written by the same author. "
+ "Generator C was built specifically to break shared implementation assumptions "
+ "and shares no code path with generator B, but shared conceptual assumptions "
+ "cannot be excluded."));

body.push(h2("4.5 Conclusion"));
body.push(p(
  "Benchmark validity in medical image segmentation is a three-way property of the "
+ "benchmark, the clinical reference, and the method pool. Rank agreement between "
+ "clinical references falls from +0.944 within a dataset to +0.563 between "
+ "datasets of one modality and to no detectable agreement across modalities, with "
+ "a within-dataset control establishing that the decline is real. A synthetic "
+ "benchmark's measured validity halves when modern architectures enter the method "
+ "pool. The natural reduction of validity to a single measured image statistic "
+ "fails. And within one modality, the fraction of the frame a reference's lesions "
+ "occupy predicts the direction of its bias over synthetic design, monotonically, "
+ "across four seeds and under both an overlap and a boundary metric."));

body.push(p(
  "What follows for practice is that a benchmark validity figure should be "
+ "reported as a matrix over references, with the method pool named and a "
+ "within-dataset control included, and that cross-domain rank comparisons over "
+ "pools mixing trained and untrained methods should not be read as statements "
+ "about the data."));

const doc = new Document({
  creator: "Lamia Islam",
  title: "Synthetic benchmark validity depends on the clinical reference and the method pool",
  styles: {
    default: {
      document: { run: { font: FONT, size: 22, color: INK } }
    }
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: {
          top: convertInchesToTwip(1), bottom: convertInchesToTwip(1),
          left: convertInchesToTwip(1), right: convertInchesToTwip(1)
        }
      }
    },
    children: body
  }]
});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync("Islam_benchmark_validity_draft.docx", b);
  console.log("wrote Islam_benchmark_validity_draft.docx");
});
