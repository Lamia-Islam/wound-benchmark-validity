// Manuscript V5 (blinded), Journal of Imaging Informatics in Medicine format.
// Citations are written [@key] or [@a;@b] and numbered in order of first use.
const fs = require("fs");
const L = require("./lib");
const { d, h1, h2, table, rule, ref } = L;
const { Document, Packer, Paragraph, TextRun, AlignmentType, PageBreak, ImageRun,
        convertInchesToTwip } = d;

const REFS = JSON.parse(fs.readFileSync("refs.json", "utf8"));
const order = [];
function cite(text) {
  return text.replace(/\[(@[^\]]+)\]/g, (_, inner) => {
    const keys = inner.split(";").map(s => s.trim().replace(/^@/, ""));
    const nums = keys.map(k => {
      if (!REFS[k]) throw new Error("unknown reference " + k);
      if (!order.includes(k)) order.push(k);
      return order.indexOf(k) + 1;
    }).sort((a, b) => a - b);
    const parts = [];
    for (let i = 0; i < nums.length; i++) {
      let j = i;
      while (j + 1 < nums.length && nums[j + 1] === nums[j] + 1) j++;
      parts.push(j - i >= 2 ? `${nums[i]}-${nums[j]}` : nums.slice(i, j + 1).join(", "));
      i = j;
    }
    return "[" + parts.join(", ") + "]";
  });
}

const B = [];
const add = (...xs) => xs.forEach(x => B.push(x));
const P = (t, o) => add(L.p(cite(t), o));
const cap = (kind, n, t) => add(L.caption(kind, n, cite(t)));
function fig(file, wmm) {
  const px = require("child_process").execSync(`python3 -c "from PIL import Image;i=Image.open('${file}');print(i.size[0],i.size[1])"`).toString().trim().split(" ").map(Number);
  const w = Math.round(wmm / 25.4 * 96);
  add(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 80 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(file),
      transformation: { width: w, height: Math.round(w * px[1] / px[0]) } })] }));
}

// ------------------------------------------------------------------ title
add(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 160, line: 300 },
  children: [new TextRun({ text: "How the clinical reference and the method pool determine synthetic benchmark validity in medical image segmentation",
    font: L.FONT, size: 30, bold: true, color: L.INK })] }));
add(rule(220));

add(h2("Abstract"));
P("**Purpose.** Synthetic test sets for medical image segmentation are usually validated by showing that they rank methods the way a clinical dataset does. We measured how much this figure depends on the clinical dataset and on the methods compared.");
P("**Methods.** We scored 57 segmentation methods in four groups: 18 classical methods without training, 14 methods trained from scratch on wound photographs, the same 14 trained on colonoscopy images, and 11 modern models (ten ImageNet-pretrained architectures and nnU-Net) trained on wound photographs. Each was scored on 11 public clinical datasets in four modalities, a within-dataset control and parametric synthetic wound benchmarks. Agreement was the Spearman correlation of per-method mean Dice, with method-bootstrap intervals and exact permutation tests.");
P("**Results.** Within every group, same-modality datasets agreed more than different-modality datasets (difference 0.46 to 0.60, p at most 0.005). Pooling classical and modern methods created agreement absent within either group: wound datasets agreed at 0.80 (95% interval 0.75 to 0.86) over the pooled 29 methods but at 0.13 (-0.12 to 0.41) among the modern models, although each dataset alone ranked them reproducibly. A synthetic benchmark agreed with clinical wound data at 0.72 (0.53 to 0.85) over the pool and at 0.35 (-0.07 to 0.63) among the modern models. Its apparent gain from background texture appeared only in pooled sets.");
P("**Conclusions.** A validity figure for a synthetic benchmark reflects the method pool and the clinical reference as much as the benchmark. Validation should be reported within the methods to be compared, against several clinical datasets.");
P("**Keywords** Benchmark validity; Rank agreement; Synthetic data; Wound imaging; Image segmentation; Evaluation methodology", { after: 200 });

// ------------------------------------------------------------ introduction
add(h1("Introduction"));
P("Segmentation software is increasingly evaluated before it reaches a clinic, by vendors comparing candidate models, by hospitals validating a product on local data and by regulators considering computational evidence. Synthetic and simulated images are attractive for this purpose because they come with exact ground truth and can be generated in any quantity, and in-silico trials and edited images have already been used to evaluate and stress-test medical models [@badano2018;@sizikova2023;@tushar2026a;@perezgarcia2024]. Rankings of segmentation methods are, however, known to be fragile. They change with the test cases, the metric, the aggregation rule and the annotators [@maierhein2018;@reinke2024], with the choice of benchmark [@dehghani2021;@pierard2025] and with the hospital the test data come from [@bassi2024;@ali2024;@urooj2026].");
P("The usual evidence that a synthetic test set is trustworthy is a rank correlation between method scores on the synthetic set and on one clinical dataset, sometimes expressed relative to the agreement between two real datasets [@kang2022;@borsukiewicz2026]. Synthetic data have been proposed for model validation and selection on the same basis [@hu2023;@vanbreugel2023;@shoshan2023], and in computed tomography the transfer of rankings from synthetic to real data has been shown to depend on the task [@tushar2026b]. Such a correlation is computed over a particular set of methods. Outside medical imaging it is established that agreement between benchmarks depends on which models are compared [@perlitz2024;@zhang2026], and that the correlation between in-distribution and out-of-distribution performance depends on how the models were selected and trained [@teney2023;@miller2021;@fang2022]. When groups of methods change order between domains, pooling them produces correlations that hold in neither group, a form of Simpson's paradox [@simpson1951;@kievit2013].");
P("Medical segmentation studies often pool exactly such groups. Wound segmentation studies, for example, compare classical colour and threshold methods with trained networks [@lucho2024;@benfatah2026], and a synthetic benchmark validated over such a pool inherits whatever the pooling does to the correlation. How large these effects are for segmentation, and what they imply for validating a synthetic benchmark, has not been measured.");
P("We scored 57 methods in four homogeneous groups on 11 clinical datasets from eight sources in four modalities and on synthetic wound benchmarks from three generator families, and asked three questions. First, whether agreement between clinical datasets is organised by modality within each group. Second, how much pooling groups changes that agreement. Third, what this implies for the measured validity of a synthetic benchmark. The contributions are:");
[
  "Evidence from four method groups, including pretrained architectures and nnU-Net, that rank agreement between clinical datasets is higher within a modality than across modalities, with exact tests that respect the dependence between pairs.",
  "A measurement of the composition effect in segmentation: pooling untrained with trained methods raises agreement between wound datasets from 0.13 to 0.80 and turns agreement between wound and other datasets negative, and a mirror experiment with colonoscopy-trained methods shows when the effect appears and when it does not.",
  "A demonstration that the measured validity of a synthetic benchmark follows the method pool, together with the finding that wound datasets that each rank modern models reproducibly do not rank them alike, so a benchmark cannot be validated against them for that purpose.",
].forEach((t, i) => P(`${i + 1}. ${t}`, { indent: { left: 240 } }));

// ---------------------------------------------------------------- methods
add(h1("Materials and Methods"));
add(h2("Clinical datasets"));
P("Eleven public datasets serve as clinical references (Table 1). The wound photographs are Medetec [@medetec2014], CO2Wounds-V2 [@sanchez2024], the FUSeg challenge validation set [@wang2024] and the AZH chronic wound dataset [@wang2020]. FUSeg and AZH were collected at one wound centre and are treated as one source; CO2Wounds-V2 contains 764 smartphone photographs from 96 leprosy patients in Colombia [@sanchez2024] and shares no source with any other dataset. The colonoscopy datasets are Kvasir-SEG [@jha2020] and its sessile subset, CVC-ClinicDB [@bernal2015], CVC-ColonDB [@bernal2012] and ETIS-LaribPolypDB [@silva2014]; the two CVC sets come from one group and are one source. PH2 [@mendonca2013] is the dermoscopy reference and the TN3K test split [@gong2021] the thyroid ultrasound reference. A disjoint slice of the AZH training set serves as a within-dataset control.", { after: 100 });
P("The Kvasir reference was drawn only from images outside the sessile subset, so the two Kvasir references share no image. For CO2Wounds-V2 and the five other non-wound datasets, 150 image and mask pairs were taken evenly spaced over the files sorted by number, because consecutive colonoscopy frames are near duplicates. Images in these six sets were resampled so that the long side is at most 256 pixels; wound masks were binarised at any non-zero value and all others above 127. Medetec's 152 images are 38 photographs with four variants each.", { after: 100 });
P("Duplicated evaluation images were removed before any analysis. Eight FUSeg validation images are byte-identical to FUSeg training images or to another validation image. Feature matching (SIFT with RANSAC, at least 25 geometrically consistent matches) showed that 47 of the 150 control images are crops of FUSeg photographs, 35 of them of images used for training; these were removed, leaving 103. The AZH test images match no FUSeg image used by any method.");
cap("Table", 1, "Clinical datasets after removal of duplicated images. Lesion fraction is the median share of image pixels inside the reference mask.");
add(table(
  ["Dataset", "Modality", "Source", "n", "Empty masks", "Lesion fraction"],
  [
    ["Medetec", "Wound photography", "Medetec archive", "152", "0", "0.098"],
    ["CO2Wounds-V2", "Wound photography", "Leprosy clinic, Colombia", "150", "0", "0.045"],
    ["FUSeg (validation)", "Wound photography", "AZH Wound Center", "112", "4", "0.007"],
    ["AZH (test)", "Wound photography", "AZH Wound Center", "150", "10", "0.019"],
    ["AZH train slice (control)", "Wound photography", "AZH Wound Center", "103", "8", "0.011"],
    ["Kvasir", "Colonoscopy", "Kvasir collection", "120", "0", "0.128"],
    ["Kvasir-sessile", "Colonoscopy", "Kvasir collection", "120", "0", "0.043"],
    ["CVC-ClinicDB", "Colonoscopy", "CVC, Barcelona", "150", "0", "0.067"],
    ["CVC-ColonDB", "Colonoscopy", "CVC, Barcelona", "150", "0", "0.038"],
    ["ETIS-Larib", "Colonoscopy", "ETIS, Paris", "150", "0", "0.020"],
    ["PH2", "Dermoscopy", "Porto", "150", "0", "0.231"],
    ["TN3K (test)", "Thyroid ultrasound", "TN3K", "150", "0", "0.066"],
  ],
  [2350, 1800, 1950, 600, 1100, 1560], { leftCols: 3 }));

add(h2("Method groups"));
P("Fifty-seven methods form four groups that differ in training and in training domain (Table 2). The 18 untrained methods are thresholding, colour, clustering and region-growing algorithms with no learned parameters. The 14 from-scratch methods are three pixel classifiers (two logistic regressions and gradient boosting on colour and texture features), a three-level U-Net with 117,681 parameters and ten architectures from segmentation_models_pytorch [@iakubovskii2019] with randomly initialised encoders: U-Net [@ronneberger2015], U-Net++ [@zhou2018], DeepLabV3+ [@chen2018], FPN [@lin2017], PSPNet [@zhao2017], LinkNet [@chaurasia2017] and MA-Net [@fan2020] on ResNet-18 [@he2016], U-Net on MobileNetV2 [@sandler2018] and EfficientNet-B0 [@tan2019], and SegFormer on MiT-B0 [@xie2021].", { after: 100 });
P("The from-scratch group was trained on FUSeg training images. The pixel classifiers were fitted on 247,975 pixels sampled from images 0 to 249, balanced between lesion and background; one decision threshold (0.90) was swept for the logistic regression on images 250 to 399 and applied to all three. The small U-Net uses a threshold of 0.5. The ten architectures were trained on images 0 to 392 at 128 by 128 pixels for 30 epochs (Dice plus binary cross-entropy, Adam, learning rate 1e-3, horizontal flips) and each threshold was swept on images 400 to 549. The Kvasir-trained group repeats this protocol exactly on 684 Kvasir-SEG images that appear in neither Kvasir reference nor the sessile subset (pixel threshold 0.60).", { after: 100 });
P("The modern group consists of the same ten architectures with ImageNet-pretrained encoders [@deng2009], trained at 256 pixels with the aspect ratio preserved (AdamW, learning rate 3e-4 with cosine decay, 40 epochs, flips and 90-degree rotations), and nnU-Net v2 [@isensee2021] in its 2D configuration trained for 100 epochs. All eleven used FUSeg training images 0 to 392 with the one image duplicated in the evaluation set removed, and thresholds for the ten architectures were swept on images 400 to 549. SAM [@kirillov2023] and MedSAM [@ma2024] were also run with box prompts taken from the reference masks; because those prompts give them the lesion location, they are reported only in Online Resource 1 and are not part of any pool.", { after: 100 });
P("Untrained methods receive morphological closing and opening with a 5-pixel elliptical element. Methods that output a probability map receive Gaussian smoothing with a standard deviation of 2 pixels before thresholding. Every method then has connected components smaller than 0.05% of the image removed.");
cap("Table", 2, "Method groups. Held-out Dice is on training-domain images not used for fitting; mean Dice is the range of group means over datasets.");
add(table(
  ["Group", "n", "Training data", "Held-out Dice", "Mean Dice, wound", "Mean Dice, other"],
  [
    ["Untrained", "18", "none", "", "0.06 to 0.30", "0.07 to 0.49"],
    ["From scratch", "14", "FUSeg", "0.51 to 0.64", "0.28 to 0.53", "0.01 to 0.13"],
    ["Kvasir-trained", "14", "Kvasir-SEG", "0.66 to 0.81", "0.08 to 0.29", "0.07 to 0.64"],
    ["Modern", "11", "FUSeg, ImageNet encoders", "0.75 to 0.84", "0.45 to 0.79", "0.01 to 0.09"],
  ],
  [1600, 500, 2500, 1500, 1630, 1630], { leftCols: 3 }));

add(h2("Synthetic benchmarks"));
P("Three parametric generator families produce wound images with exact masks, 45 images per benchmark in three severity classes. Generator A places a lesion on a uniform drape and reproduces the benchmark released with an earlier system description [@islam2026]. Generator B renders textured skin with clutter such as hair and gauze, and its background texture amplitude is a parameter; it was built at amplitudes 11, 28 and 40. Generator C shares no code with B and uses spectral noise, vignetting and a colour mixture model. Every benchmark was built at three seeds (2024, 7 and 101). Background texture is the median local standard deviation of shading-corrected luminance in a 9-pixel window over background pixels, measured at 512 pixels.");

add(h2("Agreement and statistics"));
P("Agreement between two image sets is the Spearman correlation between their vectors of per-method mean Dice over a given set of methods. Pairs of clinical datasets are assigned to four levels: within dataset (two disjoint slices of one dataset), within source (one clinic or group, no shared images), within modality (one modality, different sources) and across modality. Every analysis is repeated under normalised surface Dice (NSD) with a tolerance of 2 pixels [@nikolov2018], and with the images whose reference mask is empty excluded.", { after: 100 });
P("Pairs share datasets, so pair-level tests overstate the evidence. The difference between mean same-modality and mean across-modality agreement was therefore tested by assigning the modality labels to the 11 datasets in every distinct way (13,860 assignments, smallest attainable p 0.00014). Because singleton modalities contribute no same-modality pair, we also tested separately whether the four wound datasets (330 possible subsets) and the five colonoscopy datasets (462) agree more among themselves than with the rest.", { after: 100 });
P("Intervals are 95% percentile intervals from a bootstrap over methods, with methods resampled within each group so that a pooled set keeps its composition (2,000 draws; 1,000 for the level tables). Uncertainty from the images was assessed by split-half reliability: the Spearman correlation between method means on two random halves of a dataset, corrected with the Spearman-Brown formula. As a remedy for pooled sets we also computed stratified agreement, in which methods are ranked within their own group before pooling. The modality structure within groups and the effect of pooling are the confirmatory analyses; the synthetic-benchmark analyses are exploratory and reported with intervals rather than tests.");

// ---------------------------------------------------------------- results
add(h1("Results"));
add(h2("The groups differ in quality and in domain"));
P("The modern models reach Dice 0.45 to 0.79 on the wound datasets but 0.01 to 0.09 on the other modalities, and the from-scratch group shows the same pattern at a lower level (Table 2). The Kvasir-trained group leads on colonoscopy without falling below the untrained group on wounds, and the untrained group is moderate everywhere, with its best result on PH2 (0.49). Each dataset ranks the methods reproducibly: split-half reliability over all methods is 0.96 to 0.99 for every dataset, and among the 11 modern models it is 0.87 to 0.97 for the four wound datasets.");

add(h2("Agreement is organised by modality within every group"));
P("Within each of the four groups, datasets of the same modality agree more than datasets of different modalities, by 0.46 to 0.60 under Dice and 0.42 to 0.69 under NSD, every difference with p at most 0.005 (Table 3). Excluding empty reference masks changes no conclusion (Dice difference 0.46 to 0.60, p at most 0.0062). The structure is carried mainly by colonoscopy: the five colonoscopy datasets cluster in every group (p 0.002 to 0.009), whereas the four wound datasets do not cluster detectably within any group (p 0.085 to 0.37). Colonoscopy datasets from three independent sources agree at 0.80 to 0.95 among all 32 locally scored methods.");
cap("Table", 3, "Mean Spearman agreement by level under Dice, the same-modality minus across-modality difference with its 95% bootstrap interval and exact p, and the corresponding NSD difference. Pooled rows are discussed in the next section.");
add(table(
  ["Methods", "n", "Within dataset", "Within source", "Within modality", "Across modality", "Difference (95% interval)", "p", "NSD difference"],
  [
    ["Untrained", "18", "+0.96", "+0.69", "+0.67", "+0.08", "0.60 (0.35, 0.82)", "0.00014", "0.42"],
    ["From scratch", "14", "+0.95", "+0.65", "+0.60", "+0.14", "0.47 (0.23, 0.73)", "0.0045", "0.56"],
    ["Kvasir-trained", "14", "+0.85", "+0.54", "+0.68", "+0.16", "0.50 (0.25, 0.73)", "0.00058", "0.69"],
    ["Modern", "11", "+0.69", "+0.58", "+0.48", "+0.04", "0.46 (0.17, 0.76)", "0.00072", "0.44"],
    ["Untrained + from scratch", "32", "+0.89", "+0.86", "+0.74", "-0.07", "0.84 (0.64, 0.99)", "0.00014", "0.71"],
    ["Untrained + Kvasir-trained", "32", "+0.97", "+0.58", "+0.70", "+0.00", "0.68 (0.59, 0.75)", "0.00014", "0.72"],
    ["Untrained + modern", "29", "+0.97", "+0.86", "+0.84", "-0.24", "1.09 (0.95, 1.19)", "0.00014", "0.92"],
  ],
  [2050, 400, 900, 900, 950, 950, 1500, 700, 1010], { leftCols: 1, size: 16 }));
P("Among the modern models the within-dataset control is only 0.69 (95% interval 0.05 to 0.95) on 103 control images, against split-half reliability of 0.87 and 0.86 for the two AZH sets themselves, so with 11 closely matched methods a single comparison of two datasets is imprecise.");

add(h2("Pooling creates agreement and disagreement"));
P("Pooling the untrained group with a trained group changes agreement in both directions (Figure 1). With the modern models, wound datasets agree at 0.80 (0.75 to 0.86) over the pooled 29 methods, against 0.37 (0.14 to 0.61) among untrained methods and 0.13 (-0.12 to 0.41) among modern models. Agreement between wound and other datasets becomes -0.50 (-0.62 to -0.35), against 0.09 and -0.01 within the groups, and non-wound datasets of different modalities appear to agree at 0.40 (0.17 to 0.61), against 0.04 and 0.17. The from-scratch group produces the same pattern at a smaller size. In each case the pooled value lies outside the range of both group values.", { after: 100 });
P("The mechanism is visible in the group means. The modern models lead the untrained methods on every wound dataset and fall below them on every other dataset, so a pooled correlation between a wound and a non-wound dataset is dominated by which group each method belongs to, without any pair of methods changing order within a group. The Kvasir-trained group is the mirror experiment. It leads on colonoscopy, but on wound photographs its means (0.08 to 0.29) are close to those of the untrained methods (0.06 to 0.30), so the groups do not change places. Pooling it with the untrained group still pulls agreement between colonoscopy and other datasets outside the range of both groups (0.02, against 0.13 and 0.20), but does not reverse its sign (Figure 1c).", { after: 100 });
P("Stratified agreement removes the effect. Ranking methods within their own group before pooling gives a same-modality minus across-modality difference of 0.54, 0.55 and 0.55 for the three pooled sets (p at most 0.0003), close to the within-group values, and agreement between wound and other datasets of 0.05 to 0.13.");
fig("figures/Fig1_pooling.png", 174);
cap("Figure", 1, "Mean agreement between pairs of clinical datasets by category, for each group alone and pooled, with 95% bootstrap intervals over methods (Dice). The domain is the modality the trained group was trained on. a: untrained and modern methods. b: untrained and from-scratch methods. c: untrained and Kvasir-trained methods.");

add(h2("Synthetic benchmark validity follows the method pool"));
P("Generator B at texture amplitude 40 agrees with the four wound datasets at 0.72 (0.53 to 0.85) over the pooled untrained and modern methods, a value that would usually be read as strong validation (Figure 2, Table 4). Among the modern models alone it agrees at 0.35 (-0.07 to 0.63), and the wound datasets agree with each other at only 0.13 (-0.12 to 0.41). Among untrained methods the benchmark agrees at 0.53 (0.20 to 0.74). For the from-scratch and Kvasir-trained groups the intervals include zero.", { after: 100 });
P("Within a group the figure also depends on the reference. Among untrained methods the benchmark agrees with CO2Wounds-V2 at 0.80 and with AZH at 0.31; among Kvasir-trained methods it agrees with AZH at 0.56 and with FUSeg at -0.34. Reported against a single dataset, the same benchmark and pool could be described as valid or invalid.", { after: 100 });
P("Background texture appeared earlier in this work to raise validity. Moving generator B from amplitude 11 to 40 raises mean agreement over the four wound datasets by 0.30 (0.10 to 0.55) in the pooled untrained and modern set and by 0.21 (0.08 to 0.36) in the pooled untrained and from-scratch set, but by -0.03 to 0.08 within each group, every interval including zero. The one within-group exception is CO2Wounds-V2 among untrained methods (0.32, 0.09 to 0.60).");
fig("figures/Fig2_benchmark_validity.png", 140);
cap("Figure", 2, "Agreement of generator B (texture amplitude 40, mean over three seeds) with the four clinical wound datasets, and agreement of those datasets with each other, for each group and pooled set, with 95% bootstrap intervals over methods (Dice).");
cap("Table", 4, "Agreement of generator B (amplitude 40, three seeds) with each wound dataset, by method set (Dice).");
add(table(
  ["Methods", "Medetec", "CO2Wounds-V2", "FUSeg", "AZH", "Mean (95% interval)"],
  [
    ["Untrained", "+0.59", "+0.80", "+0.39", "+0.31", "0.53 (0.20, 0.74)"],
    ["From scratch", "+0.05", "+0.47", "+0.10", "+0.28", "0.22 (-0.04, 0.45)"],
    ["Kvasir-trained", "+0.16", "+0.06", "-0.34", "+0.56", "0.11 (-0.35, 0.54)"],
    ["Modern", "+0.30", "+0.45", "+0.28", "+0.39", "0.35 (-0.07, 0.63)"],
    ["Untrained + from scratch", "+0.34", "+0.63", "+0.21", "+0.20", "0.34 (0.07, 0.57)"],
    ["Untrained + Kvasir-trained", "+0.44", "+0.16", "-0.03", "+0.70", "0.32 (0.06, 0.54)"],
    ["Untrained + modern", "+0.75", "+0.81", "+0.66", "+0.66", "0.72 (0.53, 0.85)"],
  ],
  [2700, 1050, 1350, 1000, 1000, 2260], { leftCols: 1 }));

add(h2("Comparisons between generator families depend on the pool"));
P("Four benchmarks from the three families were compared at each seed: generator A, generator B with texture removed, and generator C without and with noise (amplitude 3.4). B without texture and C with noise had nearly the same texture statistic at seed 2024 (3.23 and 3.20) but not at seeds 7 and 101 (2.98 and 3.33, 3.06 and 3.28), so the statistic varies by seed as much as between generators. Among untrained methods, B without texture agreed with Medetec and AZH more than C with noise at every seed (by 0.18, 0.22 and 0.24), but with Medetec and CO2Wounds-V2 only by 0.01 to 0.08. Among modern models the order reversed: C with noise agreed with Medetec and AZH more than B, by 0.13 to 0.16 at every seed. Which generator family looks better therefore depends on the methods and the references used to judge it (Online Resource 1), consistent with reports that a single image statistic does not determine the usefulness of synthetic data [@deo2025;@borsukiewicz2026].");

// ------------------------------------------------------------- discussion
add(h1("Discussion"));
add(h2("What a validity figure measures"));
P("A rank correlation between a synthetic benchmark and clinical data is usually read as a property of the benchmark. In our data the same benchmark scores 0.72 or 0.35 depending on whether classical methods are included in the pool, and 0.80 or 0.31 depending on which wound dataset is the reference. The high value comes from the gap between groups of methods. A benchmark on which modern models beat classical ones, as they do on every wound dataset, will correlate with any of those datasets over a pooled set, whether or not it can order the modern models among themselves.");
P("That second question is the one a vendor, a hospital or a regulator usually faces. Among the 11 modern models, each wound dataset produces a reproducible ranking, yet the four datasets agree with each other at 0.13 on average. If clinical wound datasets do not agree on how modern models rank, a synthetic benchmark cannot be validated against them for that purpose, and a high pooled correlation should not be taken as evidence that it can.");

add(h2("Relation to prior work"));
P("The composition effect is Simpson's paradox [@simpson1951;@kievit2013], and its role in benchmark agreement is established for language models [@perlitz2024;@zhang2026] and for robustness of image classifiers, where models trained on different data lie on different lines relating in-distribution and out-of-distribution accuracy [@miller2021;@fang2022;@teney2023]. Our contribution is to measure its size in medical segmentation, to show that it inflates agreement within one modality as well as creating disagreement across modalities, and to show its consequence for validating synthetic benchmarks. The task dependence of synthetic-to-real transfer reported for CT [@tushar2026b] and the strong transfer reported for programmable lesion trials [@tushar2026a] are consistent with our finding that validity is not a fixed property of a synthetic set. Reporting intervals over methods follows recommendations to quantify uncertainty in medical imaging evaluation [@wiesenfarth2021;@christodoulou2024]. Finally, the advice to maximise variation in synthetic data comes from training, where domain randomisation helps models transfer to real images [@tobin2017;@tremblay2018;@prakash2019]. A benchmark has a different job: it should order methods as the deployment data would, which argues for matching the deployment distribution rather than exceeding it.");

add(h2("Why wound photography differs"));
P("Colonoscopy datasets from three centres agree almost as well as two slices of one dataset, while wound datasets do not cluster within any group. Colonoscopy frames share an acquisition device class and field of view. Wound photographs have no standard framing: the four wound datasets span a fourteen-fold range of median lesion fraction (0.007 to 0.098), two are different crops of one clinic's photographs and one was taken on smartphones in a leprosy programme. Dice depends on lesion size [@raina2023], and under NSD mean agreement between wound datasets among untrained methods rises from 0.37 to 0.62, which is consistent with framing conventions contributing to the disagreement [@torralba2011;@kendrick2022].");

add(h2("Recommendations"));
[
  "Report a benchmark's agreement within the group of methods it is meant to compare, and name that group.",
  "Report agreement against several clinical datasets, as a matrix, together with a within-dataset control.",
  "Report intervals from resampling methods, which here are wider than the uncertainty from images.",
  "If a mixed pool cannot be avoided, report stratified agreement, ranking methods within their own group before pooling.",
].forEach((t, i) => P(`${i + 1}. ${t}`, { indent: { left: 240 }, after: 80 }));

add(h2("Limitations"));
[
  "Wound photography is represented by four datasets from three sources, two of which share one clinic. Statements about wound datasets rest on five pairs from different sources.",
  "The synthetic benchmarks are parametric, contain 45 images each and were built at three seeds. Learned generators such as diffusion models were not tested.",
  "The modern group was trained once, on 392 images at 256 pixels, and nnU-Net for 100 epochs rather than its default 1,000. Its models are representative of current architectures but not of every deployed system. SAM and MedSAM were excluded from the pools because their prompts came from the reference masks.",
  "Groups of 11 to 18 methods give wide intervals, and the within-dataset control among modern models is imprecise.",
  "Images were not harmonised to one resolution across datasets, and NSD uses a fixed 2-pixel tolerance. Untrained methods and probability-output methods receive different post-processing, which coincides with the main contrast between groups; its parameters were not varied.",
  "The synthetic-benchmark analyses are exploratory and not corrected for multiple comparisons. The generator-family comparison rests on four benchmarks at three seeds.",
  "Medetec's 152 images are 38 photographs with four variants each, so its effective sample is smaller than its image count.",
].forEach(t => P(t, { indent: { left: 240 }, after: 80 }));

add(h1("Conclusions"));
P("Agreement between clinical datasets about which segmentation method is better is organised by modality within every group of methods we tested, but pooling classical and trained methods inflates it within a modality and reverses it across modalities. A synthetic benchmark validated over such a pool inherits the inflation. Among the modern models that practitioners choose between, wound datasets that each rank the models reproducibly do not agree with each other, and no synthetic benchmark can be validated against them. Validity should be reported within the group of methods to be compared, against several clinical datasets, with intervals over methods.");

add(h1("Statements and Declarations"));
P("**Ethics approval.** Not required. The study used only publicly available, de-identified datasets distributed for research use.", { after: 80 });
P("**Consent to participate and to publish.** Not applicable.", { after: 80 });
P("**Data availability.** All clinical datasets are publicly available from their original distributors (Table 1). Per-method scores for every method on every dataset and synthetic benchmark, and the agreement for every pair of datasets in every method set, are provided in Online Resource 2. Supplementary figures and tables are in Online Resource 1.", { after: 80 });

// ------------------------------------------------------------- references
add(new Paragraph({ children: [new PageBreak()] }));
add(h1("References"));
order.forEach((k, i) => add(ref(`${i + 1}. ${REFS[k]}`)));

const doc = new Document({
  creator: "", title: "How the clinical reference and the method pool determine synthetic benchmark validity in medical image segmentation",
  styles: { default: { document: { run: { font: L.FONT, size: 22, color: L.INK } } } },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 },
    margin: { top: convertInchesToTwip(1), bottom: convertInchesToTwip(1), left: convertInchesToTwip(1), right: convertInchesToTwip(1) } } },
    children: B }]
});
Packer.toBuffer(doc).then(b => {
  fs.writeFileSync("benchmark_validity_manuscript_v5.docx", b);
  const unused = Object.keys(REFS).filter(k => !order.includes(k));
  console.log(`wrote benchmark_validity_manuscript_v5.docx; ${order.length} references cited; ${unused.length} unused: ${unused.join(", ")}`);
});
