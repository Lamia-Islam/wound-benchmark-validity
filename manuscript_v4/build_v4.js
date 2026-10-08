const fs = require("fs");
const L = require("./lib");
const { d, p, h1, h2, table, caption, figure, rule, ref } = L;
const { Document, Packer, Paragraph, TextRun, AlignmentType, PageBreak,
        convertInchesToTwip } = d;

const B = [];
const add = (...xs) => xs.forEach(x => B.push(x));
const P = (t, o) => add(p(t, o));

// ------------------------------------------------------------------ title
add(new Paragraph({
  alignment: AlignmentType.LEFT, spacing: { after: 160, line: 300 },
  children: [new TextRun({
    text: "How the clinical reference and the method pool determine synthetic "
        + "benchmark validity in medical image segmentation",
    font: L.FONT, size: 30, bold: true, color: L.INK })]
}));
add(rule(240));

// --------------------------------------------------------------- abstract
add(h2("Abstract"));
P("**Background.** A synthetic benchmark for medical image segmentation is usually "
+ "validated by showing that it ranks methods the way a clinical dataset does. That "
+ "validity figure depends on more than the benchmark, but which dependencies matter, "
+ "and by how much, has not been measured for segmentation.");
P("**Methods.** We scored 32 methods: 18 untrained classical algorithms and 14 methods "
+ "trained on chronic wound photographs (three pixel classifiers, a small U-Net, and "
+ "ten architectures trained from random initialisation). The clinical references are "
+ "eleven datasets from eight sources in four modalities (wound photography, "
+ "colonoscopy, dermoscopy, thyroid ultrasound) plus a within-dataset control. The "
+ "synthetic benchmarks are parametric wound images from three generator families. "
+ "Agreement is the Spearman rank correlation of per-method mean scores, computed "
+ "under Dice and under normalised surface Dice, over up to four generator seeds, with "
+ "exact permutation tests at the level of references.");
P("**Results.** Agreement between clinical references is organised by modality, not by "
+ "source. Same-modality agreement exceeds across-modality agreement by 0.60 among "
+ "untrained methods and by 0.47 among trained methods (exact p of 0.0001 and 0.004), "
+ "and colonoscopy datasets from three independent sources agree at 0.70 to 0.94 "
+ "among untrained methods. Pooling trained and untrained methods manufactures "
+ "structure in both directions: wound references appear anti-correlated with the "
+ "other modalities (mean -0.26) and the non-wound modalities appear to agree with each "
+ "other (+0.41), yet within either pool both values lie between 0.04 and 0.14. The "
+ "validity of a synthetic wound benchmark depends on the pool and on the reference. "
+ "Adding the 14 trained methods to the untrained pool lowers the benchmark's agreement "
+ "with clinical data by 0.21 to 0.24 against each of three reference pairs, and over "
+ "all 32 methods the same benchmark retains 48, 60 or 96 percent of the "
+ "clinical-to-clinical agreement depending on which pair is used. The effect of "
+ "background texture also depends on the reference, and benchmarks matched to within "
+ "0.03 on a measured texture statistic differ by 0.176 in validity against one "
+ "reference pair and by 0.013 against another. A reversal of a reference's preference "
+ "for lesion contrast found earlier in this work disappears within either pool. What "
+ "remains, under the boundary metric, is a monotonic weakening of the contrast "
+ "preference as the reference's lesion fraction rises, which holds in both pools and "
+ "places a held-out dataset within 4 to 15 percent of the curve range.");
P("**Conclusions.** A benchmark validity figure is a property of the benchmark, the "
+ "reference and the method pool together. Agreement should be reported as a matrix "
+ "over references, with the method pool named, trained and untrained methods "
+ "analysed separately, and a within-dataset control included.");
P("**Keywords.** benchmark validity; rank agreement; synthetic data; method pool; "
+ "evaluation methodology; medical image segmentation", { after: 200 });

// ------------------------------------------------------------ 1 introduction
add(h1("1. Introduction"));
P("Rankings of segmentation methods are known to be fragile. An analysis of 150 "
+ "biomedical image analysis challenges showed that a method's rank changes with the "
+ "test cases, the ranking scheme, the annotators and the aggregation rule "
+ "(Maier-Hein et al., 2018), and later work gave the field a framework for choosing "
+ "metrics (Maier-Hein et al., 2024) and a catalogue of metric pitfalls (Reinke et al., "
+ "2024). Leaderboards for the same task also change between test sets from "
+ "different hospitals (Bassi et al., 2024; Ali et al., 2024).");
P("Synthetic benchmarks add a further question. A parametric generator can produce "
+ "unlimited labelled images with every design property under control, which makes it "
+ "attractive for evaluating segmentation methods, but it is only useful if it orders "
+ "methods the way clinical data does. The usual way to show this is to report the rank "
+ "correlation between method scores on the benchmark and on one clinical dataset, "
+ "sometimes normalised by the agreement between two real datasets (Kang, 2022; "
+ "Borsukiewicz et al., 2026). The resulting figure is treated as a property of the "
+ "benchmark.");
P("This study measures how far that figure depends on things other than the benchmark. "
+ "We score 32 segmentation methods on eleven clinical references from eight sources in "
+ "four modalities and on synthetic wound benchmarks from three generator families. "
+ "Three dependences emerge, and a fourth effect that we originally reported turns out "
+ "to be an instance of one of them.");
P("The first dependence is the clinical reference. Agreement between two clinical "
+ "datasets about which method is better is high within a modality and much lower "
+ "across modalities, whichever method pool is used. Colonoscopy datasets collected at three "
+ "different centres agree with each other as strongly as two curations of one "
+ "collection. Wound photography is the exception: its references, including three "
+ "independent sources, agree only moderately with one another.");
P("The second dependence is the method pool, and it is the largest. When trained and "
+ "untrained methods are pooled, cross-domain rank correlations are dominated by which "
+ "methods had training matched to the domain. In our data this makes wound references "
+ "appear anti-correlated with the other modalities and makes those modalities "
+ "appear to agree with each other, although neither effect exists within a "
+ "homogeneous pool. The same mechanism lowers the measured validity of a synthetic "
+ "wound benchmark when trained methods join an untrained pool, by an amount that "
+ "depends on which clinical references are used. The "
+ "principle that benchmark agreement depends on the models compared is established "
+ "for language model benchmarks (Perlitz et al., 2024; Zhang et al., 2026) and for "
+ "in-distribution and out-of-distribution accuracy (Teney et al., 2023). We show it in "
+ "segmentation, show that it manufactures both agreement and disagreement, and show "
+ "that it can produce apparent design-property effects inside a synthetic benchmark "
+ "study.");
P("That last point is the reason for a correction we report openly. In earlier stages "
+ "of this work we found that a reference's preference for high or low "
+ "lesion-to-surround contrast reversed as its lesion fraction increased, and that "
+ "background texture raised benchmark validity. Both effects were measured over the "
+ "mixed pool. Within either homogeneous pool the reversal does not survive, and the "
+ "texture effect survives against some references and not others. What survives of "
+ "the reversal, under a "
+ "boundary-sensitive metric, is a smaller effect: the preference for contrast weakens "
+ "monotonically as lesion fraction rises, in both pools, and a held-out dataset falls "
+ "on the curve.");
P("The third dependence concerns the benchmark images. A measured background-texture "
+ "statistic does not determine validity: benchmarks from different generators "
+ "matched on it to within 0.03 differ substantially in how well they order methods "
+ "against one reference pair, although against another they are indistinguishable. "
+ "This agrees with reports that a single fidelity statistic does not determine the "
+ "usefulness or ranking validity of synthetic data (Deo et al., 2025; Borsukiewicz "
+ "et al., 2026), and extends them with a controlled matching design.");

add(h2("1.1 Contributions"));
[
  "A measured structure of rank agreement over eleven clinical references from eight "
  + "sources in four "
  + "modalities, with a within-dataset control, showing that agreement is organised by "
  + "modality rather than by source, with exact permutation tests that respect the "
  + "dependence between pairs.",
  "A demonstration in segmentation that pooling trained and untrained methods "
  + "manufactures cross-domain rank structure in both directions, traced to training "
  + "domain with per-reference evidence.",
  "A decomposition over three method pools and three reference pairs showing that the "
  + "measured validity of a synthetic benchmark depends on both: adding trained methods "
  + "lowers its agreement with clinical data by 0.21 to 0.24 against every pair, and over "
  + "one pool its validity ranges from 48 to 96 percent of clinical agreement with the "
  + "pair.",
  "A matched-statistic test showing that synthetic benchmark validity is not "
  + "determined by a measured background-texture statistic, and that the size of the "
  + "difference between matched benchmarks itself depends on the reference.",
  "A within-pool, boundary-metric relationship between a reference's lesion fraction "
  + "and the strength of its contrast preference, together with the finding that the "
  + "stronger reversal reported earlier is a pool-composition effect."
].forEach((t, i) => P(`${i + 1}. ${t}`, { indent: { left: 240 } }));

// ----------------------------------------------------------- 2 related work
add(h1("2. Related work"));
add(h2("2.1 Ranking stability and agreement between datasets"));
P("Maier-Hein et al. (2018) showed that challenge rankings depend on test cases, "
+ "metrics, aggregation and annotators, and the BIAS guideline set reporting standards "
+ "for challenges (Maier-Hein et al., 2020). Wiesenfarth et al. (2021) provided tools "
+ "for quantifying ranking uncertainty by bootstrapping test cases, and Eisenmann et "
+ "al. (2023) analysed which strategies distinguish challenge winners. Within-dataset "
+ "bootstrap analyses of this kind are the counterpart of our split-half control. They "
+ "measure how stable one dataset's ranking is, not whether independently collected "
+ "datasets agree.");
P("Evidence on agreement between datasets is mostly qualitative in medical "
+ "segmentation. The Touchstone benchmark found different winners on external test "
+ "sets from different hospitals without reporting a rank correlation (Bassi et al., "
+ "2024), and the EndoCV2021 challenge folded performance deviation between seen and "
+ "unseen colonoscopy centres into its ranking (Ali et al., 2024). The Medical "
+ "Segmentation Decathlon found top-team rankings fairly stable across tasks "
+ "(Antonelli et al., 2022), but those are rankings of self-configuring pipelines "
+ "retrained per task, not of a fixed method pool scored on several references. Work "
+ "on strong baselines shows that claimed improvements often disappear when baselines "
+ "are properly configured (Isensee et al., 2021, 2024). For wound images, a "
+ "three-model study reported consistent rankings across two external datasets "
+ "(Benfatah, 2026) and a seven-model study reported substantial variability across "
+ "datasets (Lucho et al., 2024). Outside medicine, rankings of 40 background "
+ "subtraction methods transferred poorly across 53 video domains (Piérard et al., "
+ "2025).");
P("In general machine learning, the question is usually posed as whether rankings "
+ "survive distribution shift. Rankings on a reproduced ImageNet test set were largely "
+ "preserved (Recht et al., 2019), in-distribution and out-of-distribution accuracy "
+ "are often strongly correlated (Miller et al., 2021; Baek et al., 2022), and "
+ "robustness to synthetic shifts transfers little to natural shifts (Taori et al., "
+ "2020). These studies use fixed, mostly neural model zoos on natural images, and "
+ "they flag medical data and non-neural models as exceptions to the strong "
+ "correlations. Our setting sits in exactly those conditions.");

add(h2("2.2 The method pool as a confound"));
P("Perlitz et al. (2024) showed for language model benchmarks that the agreement used "
+ "to validate a new benchmark changes substantially with which subset of models is "
+ "compared. Zhang et al. (2026) showed that much of the disagreement between "
+ "benchmark rankings is caused by differences in how much each model was trained on "
+ "each task, and that giving every model the same task-specific training raises "
+ "average agreement from 0.51 to 0.74. Dominguez-Olmedo et al. (2025) made the same "
+ "point about apparent capability gains. Teney et al. (2023) showed that the "
+ "correlation between in-distribution and out-of-distribution accuracy can be "
+ "negative on real datasets and that earlier positive findings partly reflected "
+ "which models were selected. Our results carry this principle into medical "
+ "segmentation with a pool that mixes untrained and trained methods, and show that the "
+ "mechanism produces both spurious agreement and spurious disagreement, and can "
+ "produce apparent design-property effects in a synthetic benchmark study.");

add(h2("2.3 Synthetic data for evaluation"));
P("Synthetic and simulated images have a long history in validating segmentation "
+ "against known ground truth, from digital brain phantoms (Collins et al., 1998; "
+ "Alfano et al., 2011) to in-silico imaging trials (Badano et al., 2018), and synthetic "
+ "data more broadly is reviewed by Chen et al. (2021). Much recent work synthesises "
+ "data for training rather than evaluation, for example domain randomisation for "
+ "sim-to-real transfer (Tobin et al., 2017; Tremblay et al., 2018; Prakash et al., "
+ "2019) and contrast-agnostic brain segmentation trained on synthetic scans (Billot "
+ "et al., 2023). Wound images have been synthesised with generative models (Sarp et "
+ "al., 2021; Basiri et al., 2023), but we found no prior study that uses synthetic "
+ "wound images to rank segmentation methods.");
P("Studies closest to ours ask whether a synthetic test set ranks models as real data "
+ "does. Shoshan et al. (2023) used synthetic data for model selection and gave a "
+ "condition for rank preservation, van Breugel et al. (2023) proposed synthetic test "
+ "data to improve model evaluation, and Maheshwari et al. (2024) tested synthetic data "
+ "as a benchmark for language tasks. Kang (2022) and Borsukiewicz et al. (2026) "
+ "compared synthetic-to-real rank agreement with real-to-real agreement for person "
+ "re-identification and face recognition, which is the normalisation we also use. "
+ "Borsukiewicz et al. (2026) further reported that FID is necessary but not sufficient "
+ "for benchmark reliability, and Deo et al. (2025) found that no-reference image "
+ "quality metrics correlate poorly with downstream suitability of generated medical "
+ "images. None of these varies the method pool.");

add(h2("2.4 Region size, framing and dataset conventions"));
P("Datasets carry signatures of how they were collected and framed (Torralba and "
+ "Efros, 2011), models exploit such signatures (Geirhos et al., 2020), and dataset "
+ "shift in medical imaging has causal structure that matters for evaluation (Castro "
+ "et al., 2020). Dice is biased by region size, both as a loss (Liu et al., 2024) and "
+ "as a metric whose value depends on lesion load (Raina et al., 2023), which is one "
+ "reason for boundary-based alternatives such as normalised surface Dice (Nikolov et "
+ "al., 2018, 2021). Cropping and preprocessing change which pipeline wins between "
+ "datasets (Alajmi et al., 2026), and annotation conventions themselves change scores "
+ "(Kendrick et al., 2022). The wound datasets used here differ markedly in framing: "
+ "FUSeg and AZH come from one clinic (Wang et al., 2020, 2024) but are distributed at "
+ "different crops, Medetec is a separate archive (Medetec, 2014), and CO2Wounds-V2 "
+ "was photographed with smartphones at a leprosy clinic in Colombia (Sanchez et al., "
+ "2024). Wound "
+ "segmentation methods are surveyed by Zhang et al. (2022), and diabetic foot ulcer "
+ "segmentation has its own challenge series (Yap et al., 2024).");

// ------------------------------------------------------------- 3 methodology
add(h1("3. Methodology"));
add(h2("3.1 Method pool"));
P("Thirty-two methods in two groups that differ in whether they were trained (Table 1). "
+ "Every method runs through one interface with one shared post-processing step, so "
+ "no part of a ranking reflects differences in how outputs were cleaned.");
add(caption("Table", 1, "The 32-method pool. Trained methods were all trained on FUSeg training images."));
add(table(
  ["Group", "Family", "n", "Members"],
  [
    ["Untrained (18)", "Background-referenced", "5", "adaptive Gaussian and mean thresholds, Otsu on grey, BGR and HSV corner distance"],
    ["", "Colour-referenced", "4", "Otsu on a*, 92nd percentile of a*, redness ratio, Otsu on saturation"],
    ["", "Clustering", "4", "k-means in Lab at k = 3, 4 and 5; mean shift with Otsu on a*"],
    ["", "Region and contour", "5", "watershed, GrabCut, flood fill, Chan-Vese, Canny with fill"],
    ["Trained (14)", "Pixel classifiers", "3", "logistic regression on all features and on colour only, gradient boosting"],
    ["", "Small CNN", "1", "three-level U-Net, 117,681 parameters"],
    ["", "Modern architectures", "10", "U-Net, U-Net++, DeepLabV3+, FPN, PSPNet, LinkNet, MA-Net on ResNet-18; U-Net on MobileNetV2 and EfficientNet-B0; SegFormer on MiT-B0"]
  ],
  [1500, 1900, 500, 5460], { leftCols: 4 }
));
P("The 18 untrained methods have no learned parameters and therefore no dependence on "
+ "any dataset. The 14 trained methods are all trained on FUSeg training images. The "
+ "three pixel classifiers are fitted on 264,292 pixels sampled from FUSeg training "
+ "images 0 to 249, balanced between lesion and background. The small U-Net and the "
+ "ten modern architectures are trained on images 0 to 392 at 128 by 128 pixels.", { after: 100 });
P("The ten modern architectures come from segmentation_models_pytorch (Iakubovskii, "
+ "2019) and were chosen to vary along the two axes along which segmentation networks "
+ "usually vary: seven decoders share one ResNet-18 encoder (Ronneberger et al., 2015; "
+ "Zhou et al., 2018; Chen et al., 2018; Lin et al., 2017; Zhao et al., 2017; "
+ "Chaurasia and Culurciello, 2017; Fan et al., 2020; He et al., 2016), and three "
+ "alternative encoders include a transformer (Sandler et al., 2018; Tan and Le, 2019; "
+ "Xie et al., 2021). **No pretrained weights are used anywhere.** Every architecture is "
+ "randomly initialised, so no method carries knowledge of an external dataset. All "
+ "ten share one protocol copied from the small U-Net: Dice plus binary cross-entropy, "
+ "Adam, horizontal flips, batch size 8, 30 epochs. The one deliberate difference is a "
+ "learning rate of 1e-3 rather than 3e-3, because 3e-3 was unstable for models of 14 to "
+ "22 million parameters.");

add(h2("3.2 Decision thresholds"));
P("Each group of trained methods has its own threshold protocol, and every sweep uses "
+ "images disjoint from the images the methods were fitted on and from every "
+ "evaluation set. The three pixel classifiers share one decision threshold, 0.90, "
+ "selected by sweeping on FUSeg training images 250 to 399, which are disjoint from "
+ "the 250 images they were fitted on. The small U-Net uses a fixed threshold of 0.5 "
+ "with no sweep. Each of the ten modern architectures has its own threshold, between "
+ "0.10 and 0.45, selected on FUSeg training images 400 to 549. Held-out Dice at the "
+ "selected threshold ranges from 0.507 to 0.641 across the ten, so the pool has real "
+ "spread.");

add(h2("3.3 Clinical reference sets"));
add(caption("Table", 2, "Clinical reference sets. Lesion fraction is the median over images of the share of pixels inside the ground-truth mask."));
add(table(
  ["Reference", "Modality", "Source", "n", "Empty GT", "Lesion fraction"],
  [
    ["Medetec", "Wound photography", "Medetec archive", "152", "0", "0.098"],
    ["CO2Wounds-V2", "Wound photography", "Leprosy clinic, Colombia", "150", "0", "0.045"],
    ["FUSeg (validation)", "Wound photography", "AZH Wound Center", "120", "4", "0.008"],
    ["AZH (test)", "Wound photography", "AZH Wound Center", "150", "10", "0.019"],
    ["AZH train slice (control)", "Wound photography", "AZH Wound Center", "150", "8", "0.016"],
    ["Kvasir", "Colonoscopy", "Kvasir collection", "120", "0", "0.128"],
    ["Kvasir-sessile", "Colonoscopy", "Kvasir collection", "120", "0", "0.043"],
    ["CVC-ClinicDB", "Colonoscopy", "CVC, Barcelona", "150", "0", "0.067"],
    ["CVC-ColonDB", "Colonoscopy", "CVC, Barcelona", "150", "0", "0.038"],
    ["ETIS-Larib", "Colonoscopy", "ETIS, Paris", "150", "0", "0.020"],
    ["PH2", "Dermoscopy", "Porto", "150", "0", "0.231"],
    ["TN3K (test)", "Thyroid ultrasound", "TN3K", "150", "0", "0.066"]
  ],
  [2350, 1800, 1750, 650, 1050, 1760], { leftCols: 3 }
));
P("The wound references are Medetec (Medetec, 2014), CO2Wounds-V2 (Sanchez et al., "
+ "2024), the FUSeg challenge validation "
+ "set (Wang et al., 2024) and the AZH chronic wound dataset (Wang et al., 2020). FUSeg "
+ "was built from AZH Wound and Vascular Center images, so the two are treated as one "
+ "source. Template matching of 25 AZH patches against all 1,010 FUSeg frames at six "
+ "scales gave a maximum normalised cross-correlation of 0.927 with no exact match, so "
+ "they are distinct images under different crops. The AZH train slice is a disjoint "
+ "slice of the same dataset as AZH test and serves only as the within-dataset "
+ "control. CO2Wounds-V2 holds 764 smartphone photographs of chronic wounds from 96 "
+ "leprosy patients at the Sanatorio de Contratación in Colombia (Sanchez et al., 2024). "
+ "It shares no source with any other reference, and no method was trained on it.", { after: 100 });
P("The colonoscopy references are Kvasir-SEG (Jha et al., 2020) and its sessile "
+ "subset, CVC-ClinicDB (Bernal et al., 2015), CVC-ColonDB (Bernal et al., 2012) and "
+ "ETIS-LaribPolypDB (Silva et al., 2014). The sessile images are a subset of "
+ "Kvasir-SEG, so the Kvasir reference was drawn only from images outside that subset; "
+ "the two Kvasir references share no image. CVC-ClinicDB and CVC-ColonDB come from "
+ "one research group and are treated as one source. The dermoscopy reference is PH2 "
+ "(Mendonça et al., 2013) and the ultrasound reference is the TN3K test split (Gong "
+ "et al., 2021). Content comparison found no overlap between any colonoscopy set and "
+ "Kvasir-SEG.", { after: 100 });
P("For CO2Wounds-V2 and the five non-wound references outside Kvasir, 150 pairs were taken evenly spaced "
+ "over the sorted file list, with numeric names sorted numerically; for CO2Wounds-V2 "
+ "the list is the 607 images with published masks. The colonoscopy sets are "
+ "video frames and consecutive frames are nearly identical, so taking the first 150 "
+ "would have covered only a few sequences. Masks are binarised on any non-zero value "
+ "for the wound sets, whose masks are 0 and 1 or 0 and 255, and above 127 for all other "
+ "sets, whose masks are anti-aliased or JPEG-compressed. Images in CO2Wounds-V2 and in "
+ "all seven non-wound references are resampled so the long side is at most 256 pixels.");

add(h2("3.4 Synthetic benchmarks"));
P("Three generator families produce wound images with ground-truth masks and expose "
+ "their design properties as explicit parameters. Parametric generators are used "
+ "rather than a learned generative model because the experiments require individual "
+ "properties to be switched off and swept independently. Generator A places a lesion "
+ "on a uniform drape and reproduces the 45-image benchmark released with an earlier "
+ "system description (Islam, 2026). Generator B renders textured skin with a shared luminance "
+ "mottling field, a calibrated tissue palette and clutter such as hair and gauze; its "
+ "texture amplitude is a swept parameter. Generator C shares no code with generator B "
+ "and differs in every mechanism: spectral-filtered noise, a radial vignette, "
+ "distance-field alpha blending and a CIELAB Gaussian mixture for tissue colour. The "
+ "three were written by the same authors, so we describe them as built on different "
+ "principles rather than as independent. Each benchmark contains 45 images, 15 per "
+ "wound class.", { after: 100 });
P("The background texture statistic is the median local standard deviation of "
+ "shading-corrected luminance in a 9-pixel window over background pixels, measured "
+ "after resampling every image to 512 pixels. Without the resampling, Medetec at 224 "
+ "pixels and FUSeg at 512 pixels change order on the statistic.");

add(h2("3.5 Agreement, levels and statistics"));
P("Agreement between two image sets is the Spearman correlation between their vectors "
+ "of per-method mean scores over the same pool. Pair-level p-values use 20,000 "
+ "permutations and confidence intervals use 5,000 bootstrap resamples over methods. At "
+ "n = 32 a Spearman needs roughly |rho| of 0.35 to reach p below 0.05, so an absence of "
+ "agreement is reported as an absence of detectable agreement.", { after: 100 });
P("Every pair of references is assigned to one of four levels, defined once and used "
+ "throughout: **within dataset**, two disjoint slices of one dataset; **within "
+ "source**, two references from one clinic or research group that share no images; "
+ "**within modality**, references of one modality from different sources; and "
+ "**across modality**.", { after: 100 });
P("Pairs of references share references, so pair-level tests overstate the evidence "
+ "for differences between levels. The difference between mean same-modality "
+ "agreement and mean across-modality agreement is therefore tested by permuting the "
+ "modality labels over the eleven non-control references, exactly, over all 13,860 "
+ "distinct relabelings.", { after: 100 });
P("Every result is computed under Dice and under normalised surface Dice at a "
+ "tolerance of 2 pixels (Nikolov et al., 2018). Both treat two empty masks as perfect "
+ "agreement. Because 3.3 to 6.7 percent of images in three wound references have "
+ "empty ground truth (Table 2), every agreement is also recomputed with those images "
+ "excluded. Results are reported for the full pool and separately for the 18 "
+ "untrained and 14 trained methods. Synthetic benchmarks are built at four generator "
+ "seeds for the framing analysis and three for the validity decomposition. The "
+ "matched-statistic comparison in Section 4.5 uses one seed per benchmark and is read "
+ "with that in mind.", { after: 100 });
P("A reference's preference over lesion-to-surround contrast is the ordinary least "
+ "squares slope of its agreement with a synthetic benchmark against that benchmark's "
+ "measured saturation contrast, over seven contrast levels, per 10 saturation units. "
+ "An earlier version used the Spearman correlation of agreement against contrast, "
+ "which saturates at plus or minus 1 once the relationship is monotone; the slope does "
+ "not saturate.");

// ----------------------------------------------------------------- 4 results
add(h1("4. Results"));
add(h2("4.1 Agreement between clinical references is organised by modality"));
add(caption("Table", 3, "Mean agreement by level, Dice and normalised surface Dice (NSD), all images."));
add(table(
  ["Level", "Pairs", "Dice, 32", "Dice, 18 untrained", "Dice, 14 trained", "NSD, 32", "NSD, 18", "NSD, 14"],
  [
    ["Within dataset", "1", "+0.944", "+0.973", "+0.947", "+0.960", "+0.979", "+0.908"],
    ["Within source", "3", "+0.857", "+0.689", "+0.648", "+0.898", "+0.855", "+0.927"],
    ["Within modality", "13", "+0.739", "+0.674", "+0.602", "+0.710", "+0.685", "+0.606"],
    ["Across modality", "39", "-0.073", "+0.080", "+0.142", "+0.032", "+0.298", "+0.103"]
  ],
  [1800, 700, 1050, 1300, 1250, 1050, 1050, 1160]
));
P("Agreement within a modality exceeds agreement across modalities in every pool and "
+ "under both metrics. The exact permutation test gives a same-modality minus "
+ "across-modality difference under Dice of 0.834 for all 32 methods, 0.597 for the "
+ "untrained pool and 0.468 for the trained pool (p of 0.0001, 0.0001 and 0.0045), and "
+ "under NSD of 0.713, 0.418 and 0.563 (each p = 0.0001). Excluding images with empty "
+ "ground truth leaves every difference significant (Dice: 0.761, 0.598 and 0.463, p "
+ "at most 0.0062).");
P("The within-dataset control reproduces itself at 0.944, so values well below it "
+ "reflect disagreement between references rather than noise in the measurement. The "
+ "boundary that matters is modality, not source. Colonoscopy references from three "
+ "independent sources agree at 0.80 to 0.95 over all 32 methods and 0.70 to 0.94 over "
+ "the untrained pool, as high as two curations of one source (Kvasir and its sessile "
+ "subset, 0.950; the two CVC sets, 0.961).");
P("Wound photography is the exception within modality. CO2Wounds-V2, photographed at "
+ "a clinic with no connection to any other reference or to the training data, agrees "
+ "with Medetec, FUSeg and AZH at 0.50, 0.53 and 0.46 over all 32 methods (each p below "
+ "0.01). Independent wound sources therefore agree, but at little more than half the "
+ "level of independent colonoscopy sources. Within the pools wound agreement is "
+ "unstable: among untrained methods the five different-source wound pairs range from "
+ "0.07 (CO2Wounds-V2 and AZH) to 0.74 (Medetec and FUSeg), and among trained methods "
+ "from -0.13 to 0.48. Under NSD, mean wound-to-wound agreement among untrained methods "
+ "rises from 0.37 to 0.62, which points to the region-size sensitivity of Dice combined "
+ "with very different lesion fractions (0.008 to 0.098) as part of the disagreement; "
+ "among trained methods it rises only from 0.16 to 0.31. FUSeg and AZH come from one "
+ "clinic yet agree at only 0.15 among untrained methods and 0.03 among trained "
+ "methods; their agreement of 0.66 over all 32 methods is produced by pooling, as "
+ "Section 4.2 explains.");
add(figure("fig1_matrix_print_dice.png", 620, 219));
add(caption("Figure", 1, "Agreement between the eleven clinical references under Dice, for the full pool and for each homogeneous pool. Black lines separate modalities (wound photography, colonoscopy, dermoscopy, thyroid ultrasound). Purple is positive and orange negative. Every value is listed in Appendix Table A1."));

add(h2("4.2 Mixing trained and untrained methods manufactures structure in both directions"));
P("The across-modality pairs in Figure 1 are not homogeneous. Over all 32 methods, "
+ "FUSeg and AZH are negatively correlated with every non-wound reference (-0.20 to "
+ "-0.74), Medetec is close to zero with most of them, CO2Wounds-V2 ranges from -0.47 "
+ "(TN3K) to +0.23, and the "
+ "non-wound modalities agree with each other. Table 4 splits the pairs by "
+ "whether they cross the boundary of the trained methods' domain.");
add(caption("Table", 4, "Mean agreement by category of pair, Dice, all images. NSD values follow in parentheses."));
add(table(
  ["Category", "Pairs", "All 32", "18 untrained", "14 trained"],
  [
    ["Wound with wound", "6", "+0.530 (+0.553)", "+0.373 (+0.616)", "+0.163 (+0.305)"],
    ["Non-wound, same modality", "10", "+0.899 (+0.860)", "+0.859 (+0.777)", "+0.879 (+0.882)"],
    ["Non-wound, across modality", "11", "+0.406 (+0.382)", "+0.035 (+0.071)", "+0.142 (+0.119)"],
    ["Wound with non-wound", "28", "-0.262 (-0.106)", "+0.098 (+0.387)", "+0.142 (+0.096)"]
  ],
  [3000, 800, 1850, 1850, 1860]
));
P("Both effects come from pooling. Under Dice, wound references appear anti-correlated "
+ "with the other modalities at -0.262 over all 32 methods, but the same pairs agree "
+ "at +0.098 among untrained methods and +0.142 among trained methods. Non-wound "
+ "references from different modalities appear to agree at +0.406, but the same pairs "
+ "give +0.035 and +0.142 within the pools. The pooled values lie outside the range of "
+ "both pool-specific values, which is the signature of a composition effect.");
add(caption("Table", 5, "Mean Dice of each pool on each reference."));
add(table(
  ["Reference", "14 trained", "18 untrained", "Difference"],
  [
    ["Medetec", "0.356", "0.292", "+0.064"],
    ["CO2Wounds-V2", "0.277", "0.188", "+0.090"],
    ["FUSeg", "0.543", "0.062", "+0.481"],
    ["AZH", "0.420", "0.297", "+0.122"],
    ["Kvasir", "0.127", "0.229", "-0.102"],
    ["Kvasir-sessile", "0.053", "0.111", "-0.057"],
    ["CVC-ClinicDB", "0.089", "0.180", "-0.090"],
    ["CVC-ColonDB", "0.051", "0.125", "-0.074"],
    ["ETIS-Larib", "0.015", "0.067", "-0.052"],
    ["PH2", "0.081", "0.490", "-0.409"],
    ["TN3K", "0.007", "0.102", "-0.095"]
  ],
  [3360, 2000, 2000, 2000]
));
P("Table 5 shows the mechanism. The trained methods lead on all four wound references, "
+ "most strongly on FUSeg, the distribution they were trained on, and fall below the "
+ "untrained methods on all seven non-wound references, where their mean Dice is "
+ "between 0.007 and 0.127. When two groups exchange places between domains, a pooled "
+ "rank correlation between a reference inside the domain and one outside it becomes "
+ "negative, and correlations between references outside the domain become positive, "
+ "without any pair of methods inverting.");
P("Two further observations follow. First, among trained methods the agreement between "
+ "FUSeg and the other wound references is close to zero or negative (-0.04 with "
+ "Medetec, -0.13 with CO2Wounds-V2, 0.03 with AZH), so a ranking measured on the training distribution did not predict the ranking "
+ "on shifted data of the same modality. Second, the high agreement among trained "
+ "methods on non-wound references (0.879 within colonoscopy) is agreement about how "
+ "methods fail: their Dice there is near zero, and their ordering reflects which "
+ "methods predict some foreground rather than which segment well.");
add(figure("fig2_composition_v4.png", 620, 237));
add(caption("Figure", 2, "Left: mean agreement by category of pair for the mixed pool and for each homogeneous pool. Right: mean Dice of each pool on each reference; the shaded band is wound photography, the trained methods' modality."));

add(h2("4.3 The validity of a synthetic benchmark depends on the pool and the reference"));
P("Generator B was scored at three texture amplitudes and three seeds. Its validity is "
+ "the mean of its agreement with two clinical references, expressed as a percentage "
+ "of the agreement between those two references over the same pool, so that the "
+ "normaliser never changes underneath a comparison. Earlier drafts compared a figure "
+ "computed over the original 22 methods (the 18 untrained methods, the three pixel "
+ "classifiers and the small U-Net) against Medetec and FUSeg with one computed over 32 "
+ "methods against Medetec and AZH, which changed two things at once. Table 6 "
+ "separates them and adds a third pair, Medetec and CO2Wounds-V2, neither of which "
+ "shares a source with the images the trained methods were fitted on.");
add(caption("Table", 6, "Generator B validity by method pool and reference pair. Best configuration shown; mean over three seeds."));
add(table(
  ["Method pool", "References", "Clinical agreement", "Benchmark agreement", "% of clinical"],
  [
    ["Original 22", "Medetec and CO2Wounds-V2", "+0.529", "+0.712 (sd 0.009)", "135%"],
    ["18 untrained", "Medetec and CO2Wounds-V2", "+0.540", "+0.698 (sd 0.006)", "129%"],
    ["All 32", "Medetec and CO2Wounds-V2", "+0.502", "+0.484 (sd 0.025)", "96%"],
    ["Original 22", "Medetec and FUSeg", "+0.567", "+0.562 (sd 0.047)", "99%"],
    ["18 untrained", "Medetec and FUSeg", "+0.740", "+0.511 (sd 0.028)", "69%"],
    ["All 32", "Medetec and FUSeg", "+0.461", "+0.276 (sd 0.007)", "60%"],
    ["Original 22", "Medetec and AZH", "+0.451", "+0.522 (sd 0.048)", "116%"],
    ["18 untrained", "Medetec and AZH", "+0.486", "+0.480 (sd 0.003)", "99%"],
    ["All 32", "Medetec and AZH", "+0.568", "+0.272 (sd 0.006)", "48%"]
  ],
  [1500, 2500, 1700, 2060, 1600], { leftCols: 2, boldRows: [2, 5, 8] }
));
P("With the references held fixed, adding the 14 trained methods to the untrained "
+ "pool lowers the benchmark's agreement with clinical data by a similar amount for "
+ "every pair: from 0.698 to 0.484 against Medetec and CO2Wounds-V2, from 0.511 to "
+ "0.276 against Medetec and FUSeg, and from 0.480 to 0.272 against Medetec and AZH, "
+ "a fall of 0.21 to 0.24. The percentage is less uniform. Over all 32 methods the same "
+ "benchmark retains 96 percent of clinical agreement against Medetec and CO2Wounds-V2, "
+ "but 60 and 48 percent against the two pairs that include the clinic whose images the "
+ "trained methods were fitted on. The clinical-to-clinical agreement over the full "
+ "pool is similar for the three pairs (0.461 to 0.568), so the difference lies in how "
+ "well the benchmark orders methods for each reference, not in the normaliser "
+ "(Figure 3). We do not have a mechanism for the difference between pairs. The gap "
+ "between the pools' mean Dice explains FUSeg, where trained methods lead by 0.48, but "
+ "not AZH, whose gap of 0.12 is close to that of CO2Wounds-V2 (0.09; Table 5).");
P("Two cautions apply to the percentage. It is not bounded by 100: a benchmark can "
+ "agree with each of two references more than they agree with each other, as the "
+ "values from 116 to 135 percent show. The absolute agreement is therefore reported "
+ "alongside it. Second, the effect of background texture depends on the pool and on "
+ "the reference. Over all 32 methods, agreement rises with texture amplitude against "
+ "every pair, for example from 0.133 to 0.272 against Medetec and AZH. Within the "
+ "untrained pool it is flat against Medetec and AZH (0.480, 0.443 and 0.453) but "
+ "rises against Medetec and FUSeg (0.417 to 0.511) and against Medetec and "
+ "CO2Wounds-V2 (0.482 to 0.698). Earlier versions of this work reported texture first "
+ "as a general driver of validity and then as a pure pool effect; neither statement "
+ "holds across references.");
add(figure("fig3_pool_dependence_v4.png", 640, 206));
add(caption("Figure", 3, "Generator B agreement with three pairs of clinical references at three texture amplitudes, for each method pool (mean and standard deviation over three seeds). Each horizontal line is the clinical-to-clinical agreement for the pool in the same shade, labelled with its value. Percentages are given in Table 6."));

add(h2("4.4 Lesion fraction weakens a reference's preference for contrast"));
P("AZH frames, which no method was trained on, were cropped around their "
+ "lesions to seven target lesion fractions between 0.035 and 0.30. The crops use the "
+ "same 140 frames and annotations, so lesion fraction is the only thing that changes "
+ "between the seven references. For each crop we measured its preference over "
+ "lesion-to-surround contrast against a seven-level contrast sweep of generator B, at "
+ "four seeds.");
add(caption("Table", 7, "Contrast preference against lesion fraction, by pool and metric. Monotonicity is the Spearman correlation of preference with lesion fraction, per seed."));
add(table(
  ["Metric", "Pool", "Preference at 0.035 and at 0.30", "Monotonicity, 4 seeds", "Sign change"],
  [
    ["Dice", "All 32", "+0.077, -0.086", "-1.000 in all (p 0.0005)", "0.055 to 0.080"],
    ["Dice", "18 untrained", "+0.002, -0.029", "-0.39 to -0.64 (p 0.14 to 0.40)", "none; starts at zero"],
    ["Dice", "14 trained", "+0.071, +0.088", "+0.61 to +0.96", "none"],
    ["NSD", "All 32", "+0.080, -0.079", "-1.000 in all (p 0.0005)", "0.110 to 0.160"],
    ["NSD", "18 untrained", "+0.033, +0.001", "-0.93 to -0.96 (p at most 0.006)", "none"],
    ["NSD", "14 trained", "+0.055, +0.032", "-0.86 to -0.89 (p at most 0.025)", "none"]
  ],
  [900, 1400, 2400, 3060, 1600], { leftCols: 2 }
));
P("Over the full pool, both metrics show a reference whose preference reverses: small "
+ "lesions reward high contrast and large lesions reward low contrast. That reversal "
+ "does not exist within either homogeneous pool. Under Dice, the untrained pool shows "
+ "a weak and non-significant trend and the trained pool trends the other way, so the "
+ "full-pool curve is a composition effect of the kind shown in Section 4.2. Under "
+ "NSD, the preference for contrast weakens monotonically as lesion fraction rises in "
+ "both pools and in every seed, but stays positive throughout.");
P("Medetec took no part in building the curve. Placed at its median lesion fraction, "
+ "0.098, its measured preference under NSD falls within 3 percent of the curve range "
+ "over all 32 methods, 4 percent over the untrained pool (predicted +0.024, measured "
+ "+0.023) and 15 percent over the trained pool. Under Dice the placements are poor, "
+ "consistent with the absence of a stable within-pool relationship under that metric. "
+ "The supported claim is therefore the weaker one: within a modality and under a "
+ "boundary metric, a reference with larger lesions rewards lesion-to-surround "
+ "contrast less, and a held-out dataset follows the relationship.");
add(figure("fig4_framing_pools.png", 620, 259));
add(caption("Figure", 4, "Contrast preference against the lesion fraction of AZH crops, for the mixed pool and each homogeneous pool. Stars mark held-out Medetec, measured. The reversal appears only in the mixed pool."));

add(h2("4.5 Validity is not determined by the measured texture statistic"));
add(caption("Table", 8, "Benchmarks from three generator families against two reference pairs, one seed each. M+A is Medetec and AZH, M+C is Medetec and CO2Wounds-V2; 32, 18 and 14 are the full, untrained and trained pools."));
add(table(
  ["Benchmark", "Texture", "M+A, 32", "M+A, 18", "M+A, 14", "M+C, 32", "M+C, 18", "M+C, 14"],
  [
    ["Generator A, uniform drape", "2.10", "-0.153", "+0.379", "+0.136", "-0.039", "+0.424", "+0.154"],
    ["Generator C, no noise", "2.34", "+0.007", "+0.246", "+0.024", "+0.235", "+0.389", "+0.266"],
    ["Generator C, noise 3.4", "3.20", "+0.017", "+0.232", "+0.112", "+0.245", "+0.357", "+0.343"],
    ["Generator B, texture ablated", "3.23", "+0.132", "+0.409", "+0.138", "+0.246", "+0.370", "+0.316"]
  ],
  [2560, 900, 980, 980, 980, 980, 980, 1000]
));
P("If validity were a function of the texture statistic, benchmarks matched on it "
+ "would agree. Against Medetec and AZH they do not. Generator C at 3.20 and generator B "
+ "at 3.23 are matched to within 0.03 and differ in validity by 0.176 among untrained "
+ "methods and by 0.115 over all 32. Generator A at 2.10 and generator C at 2.34 differ by "
+ "0.133 and 0.160. Among trained methods the closely matched pair differs by only "
+ "0.026, but every value in that pool is low.");
P("Against Medetec and CO2Wounds-V2 the picture is different. The matched pair differs "
+ "by 0.013 among untrained methods and by 0.001 over all 32, and the four benchmarks "
+ "lie within 0.07 of each other in the untrained pool across the whole texture range "
+ "from 2.10 to 3.23. Against this pair the four benchmarks are indistinguishable, even "
+ "though texture amplitude within generator B does raise agreement with it (Section "
+ "4.3). The texture statistic therefore does not determine validity, but whether two "
+ "benchmarks differ, and by how much, depends on the reference. Each benchmark here was "
+ "scored at one seed, and the seed-to-seed standard deviation of generator B in Table 6 "
+ "reaches 0.05, so differences of that size should not be read as real.");
add(figure("fig5_nonreducibility_v4.png", 600, 239));
add(caption("Figure", 5, "Validity against Medetec and AZH as a function of measured background texture, for four benchmarks from three generator families, over all 32 methods and over the untrained pool. Bracketed pairs are matched on the statistic."));

add(h2("4.6 Robustness"));
P("**Empty ground truth.** On an empty-ground-truth image a method scores 1 if it "
+ "predicts nothing and 0 otherwise. On AZH, all ten modern architectures leave at "
+ "least 70 percent of the empty images empty, while 16 of the 18 untrained methods "
+ "never do, which shifts the trained methods' mean Dice on AZH by up to 0.067. The "
+ "empty images create no exact ties in any reference. Excluding them changes one pair "
+ "materially: FUSeg against AZH falls from 0.660 to 0.368 over all 32 methods under "
+ "Dice, while Medetec against FUSeg and against AZH moves from 0.461 to 0.478 and from "
+ "0.568 to 0.510 (Medetec and CO2Wounds-V2 have no empty masks), "
+ "and the control from 0.944 to 0.920. Under NSD, FUSeg against AZH moves from 0.782 "
+ "to 0.712. Every modality test remains significant.", { after: 100 });
P("**Kvasir overlap.** An earlier draft scored the first 120 images of Kvasir-SEG and "
+ "of its sessile subset, which shared 31 images. With disjoint images the "
+ "within-source agreement changes from 0.952 to 0.950.", { after: 100 });
P("**Metric.** Every main result is reported under both Dice and NSD above. The "
+ "modality structure, the composition effect and the fall in benchmark agreement when "
+ "trained methods are added hold under both; the lesion-fraction relationship holds within pools only under "
+ "NSD.");

// ------------------------------------------------------------- 5 discussion
add(h1("5. Discussion"));
add(h2("5.1 What a validity figure measures"));
P("A synthetic benchmark's validity is usually reported as a single correlation with "
+ "clinical data. Our results show that number depends on three things besides the "
+ "benchmark. It depends on the clinical reference: within one modality, "
+ "independent sources agree at 0.80 to 0.95 in colonoscopy but at 0.46 to 0.57 in wound "
+ "photography over the same 32 methods, and the "
+ "same benchmark over the same pool retains 48 or 96 percent of clinical agreement "
+ "depending on which pair of references it is measured against. It depends on the "
+ "method pool: with the references fixed, adding trained methods lowers the "
+ "benchmark's agreement by about 0.2. And it is not determined by a natural image "
+ "statistic of the benchmark, while the effect of that statistic itself changes with "
+ "the reference. A figure that names none of the reference, the pool and the "
+ "composition of the pool does not transfer to another study.");
P("Four practices follow and each is inexpensive. Report agreement as a matrix over "
+ "several references rather than as a single number. Name the method pool. Analyse "
+ "trained and untrained methods, or methods trained on different data, separately "
+ "before pooling them. Include a within-dataset control, which costs one scoring run "
+ "on a disjoint slice and makes every other agreement interpretable.");

add(h2("5.2 Relation to prior work on pool effects"));
P("That benchmark agreement depends on the models compared, and that task-matched "
+ "training confounds rankings, is established for language model benchmarks "
+ "(Perlitz et al., 2024; Zhang et al., 2026; Dominguez-Olmedo et al., 2025), and the "
+ "sign of an in-distribution to out-of-distribution correlation is known to depend "
+ "on model selection (Teney et al., 2023). Our contribution is to show the same "
+ "mechanism in medical segmentation and to show three consequences that, as far as we "
+ "found, have not been reported together. Pooling manufactures spurious agreement as "
+ "well as spurious disagreement. It inflates agreement within a single clinic: FUSeg "
+ "and AZH agree at 0.66 over the full pool and at 0.15 and 0.03 within the pools. And "
+ "it produces apparent design-property effects inside a synthetic benchmark study, "
+ "including a reversal of contrast preference that we ourselves first reported as a "
+ "finding.");

add(h2("5.3 Why wound photography is the exception"));
P("Colonoscopy references from three centres agree almost as well as two halves of one "
+ "dataset, and wound references do not: the three independent wound sources agree at "
+ "0.46 to 0.57 over all 32 methods, against 0.80 to 0.95 for colonoscopy. Colonoscopy "
+ "frames share an acquisition device class and a field of view, whereas wound "
+ "photographs have no standard framing: the four wound references span a twelve-fold "
+ "range of median lesion fraction, two of them are distributed at different crops of "
+ "one clinic's images, and one was taken on smartphones in a leprosy programme. Under "
+ "NSD, mean wound agreement among untrained methods rises from 0.37 to 0.62, which is "
+ "consistent with Dice's region-size sensitivity interacting with these framing "
+ "differences. We offer this as an interpretation rather than a tested mechanism, "
+ "since four wound references are too few to test it. It is also consistent with "
+ "Section 4.4, where lesion fraction changes what a reference rewards. A three-model "
+ "study reported stable wound rankings across two external sets (Benfatah, 2026); "
+ "with three trained models, its pool is close to our trained pool, in which "
+ "agreement between FUSeg and Medetec is near zero, so the two results differ in pool "
+ "and in datasets as well as in conclusion.");

add(h2("5.4 Matching rather than maximising"));
P("The domain randomisation literature recommends maximising variation in synthetic "
+ "training data, randomising texture, lighting and placement well beyond realism, so "
+ "that a model trained on it transfers to real images (Tobin et al., 2017; Tremblay et "
+ "al., 2018), with structured variants that keep scene context (Prakash et al., "
+ "2019). Evaluation has a different objective. A benchmark exists to order methods as "
+ "the deployment distribution would, which argues for matching that distribution, "
+ "including its framing conventions, rather than exceeding it. Table 5 gives a clean "
+ "illustration of the training-side problem that randomisation addresses: methods "
+ "trained on one narrow domain reach 0.54 mean Dice on it and below 0.13 on every "
+ "other modality.");

add(h2("5.5 Limitations"));
[
  "Wound photography has four references from three sources, two of which share one "
  + "clinic. Statements about within-modality agreement for wounds rest on five "
  + "different-source pairs.",
  "All 14 trained methods were trained on one wound dataset. The composition effect is "
  + "shown for that configuration; pools trained on several domains, or pretrained "
  + "foundation models, were not tested.",
  "The modern architectures were trained from random initialisation on 393 images at "
  + "128 by 128 pixels. They span modern architectural variation under one protocol but "
  + "are not the fully trained, pretrained models a practitioner would deploy. Neither "
  + "nnU-Net nor a segmentation foundation model is included.",
  "Each agreement is a Spearman over 32, 18 or 14 points. The smaller pools have wider "
  + "intervals, and the trained pool's agreement outside its domain reflects failure "
  + "modes rather than segmentation quality.",
  "The framing crops use the ground-truth mask centroid and area, which no deployable "
  + "pipeline could reproduce. They are instruments for varying one property, not "
  + "proposed preprocessing. AZH patches start at a lesion fraction of about 0.02, so the "
  + "curve begins at 0.035, and only one held-out dataset falls in its range.",
  "The synthetic benchmarks contain 45 images each, the contrast preference is fitted "
  + "over seven contrast levels, the validity decomposition uses three seeds, and the "
  + "matched-statistic comparison uses one seed per benchmark.",
  "Both empty-mask conventions are reported, but other conventions, such as excluding "
  + "empty images only from the metric and not from ranking, were not explored.",
  "The three generator families were written by the same authors and may share "
  + "assumptions about what a wound photograph looks like."
].forEach(t => P(t, { indent: { left: 240 }, after: 90 }));

add(h2("5.6 Conclusion"));
P("The validity of a synthetic segmentation benchmark is a joint property of the "
+ "benchmark, the clinical reference and the method pool. Agreement between clinical "
+ "references is organised by modality, with wound photography a weak exception. "
+ "Pooling trained and untrained methods manufactures cross-domain agreement and "
+ "disagreement, lowers a synthetic benchmark's agreement with clinical data by about "
+ "0.2 against every reference pair tested, and produced an apparent reversal of "
+ "contrast preference in our own analysis that vanishes within homogeneous pools. The "
+ "reference pair alone moves the same benchmark's validity from 48 to 96 percent, and "
+ "decides whether background texture matters and whether two benchmarks matched on a "
+ "texture statistic differ. That statistic does not determine validity. Within a modality and under a boundary metric, a "
+ "reference with larger lesions rewards contrast less. Reporting agreement as a "
+ "matrix over references, naming the pool, separating trained from untrained methods "
+ "and including a within-dataset control would make validity figures comparable "
+ "across studies.");

add(h2("Data and code availability"));
P("All clinical reference sets are publicly available from their original "
+ "distributors and are not redistributed. The code for all 32 methods, the three "
+ "generators and every experiment, together with per-image scores for every "
+ "reference, is held in the project repository.");

// -------------------------------------------------------------- references
add(new Paragraph({ children: [new PageBreak()] }));
add(h1("References"));
const REFS = fs.readFileSync("references.txt", "utf8").split("\n")
  .map(s => s.trim()).filter(Boolean);
REFS.forEach(r => add(ref(r)));

// ---------------------------------------------------------------- appendix
add(new Paragraph({ children: [new PageBreak()] }));
add(h1("Appendix A. Agreement for every pair of references"));
const rows = JSON.parse(fs.readFileSync("appendix_pairs.json", "utf8"));
const f = x => (x >= 0 ? "+" : "") + x.toFixed(3);
const lvl = { "same dataset": "Within dataset", "same source": "Within source",
              "different source, same modality": "Within modality",
              "across modality": "Across modality" };
add(caption("Table", "A1", "Spearman agreement under Dice for every pair, by pool."));
add(table(["Pair", "Level", "All 32", "18 untrained", "14 trained"],
  rows.filter(r => r.metric === "dice").map(r =>
    [`${r.a} and ${r.b}`, lvl[r.level], f(r.all), f(r.untrained), f(r.trained)]),
  [3500, 1900, 1300, 1330, 1330], { leftCols: 2, size: 17, keep: false }));
add(caption("Table", "A2", "Spearman agreement under NSD for every pair, by pool."));
add(table(["Pair", "Level", "All 32", "18 untrained", "14 trained"],
  rows.filter(r => r.metric === "nsd").map(r =>
    [`${r.a} and ${r.b}`, lvl[r.level], f(r.all), f(r.untrained), f(r.trained)]),
  [3500, 1900, 1300, 1330, 1330], { leftCols: 2, size: 17, keep: false }));

const doc = new Document({
  creator: "",
  title: "How the clinical reference and the method pool determine synthetic benchmark validity in medical image segmentation",
  styles: { default: { document: { run: { font: L.FONT, size: 22, color: L.INK } } } },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: convertInchesToTwip(1), bottom: convertInchesToTwip(1),
                left: convertInchesToTwip(1), right: convertInchesToTwip(1) } } },
    children: B
  }]
});
Packer.toBuffer(doc).then(b => {
  fs.writeFileSync("benchmark_validity_manuscript_v4.docx", b);
  console.log("wrote benchmark_validity_manuscript_v4.docx");
});
