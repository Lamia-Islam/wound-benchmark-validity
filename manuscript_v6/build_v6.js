// Manuscript V6 (blinded), Journal of Imaging Informatics in Medicine (JIIM, Springer).
//
// JIIM submission guidelines, checked 12 October 2026 at
// https://link.springer.com/journal/10278/submission-guidelines :
//  - Abstract of 150 to 250 words, no undefined abbreviations or unspecified references;
//    no structured headings are required, so the abstract is one unstructured paragraph.
//  - 4 to 6 keywords.
//  - Hypothesis-driven research: Abstract, Introduction, Materials and Methods, Results,
//    Discussion, Conclusions, Acknowledgements. No word limit is stated. Limitations are a
//    subsection at the end of the Discussion so that this order is kept.
//  - Double-blind review: a blinded manuscript without author names and affiliations.
//  - Statements and Declarations (funding, competing interests, ethics approval, consent)
//    go on the separate title page; the owner supplies funding and competing interests there.
//    Ethics, consent and data availability are kept here as in V5.
//  - References numbered in order of first citation. Printed journal example:
//    "Zaidel M, Hopper K, Iyriboz T: Interactive web-based radiology teaching file.
//    J Digit Imaging 12:203-204, 1999". The list below is converted to this pattern
//    (surname and initials, colon after the authors, year at the end); journal names are
//    kept in full, not abbreviated to Index Medicus form.
//  - Figure captions begin with "Fig." and the number in bold; supplementary files are
//    cited as "Online Resource".
//
// Every result number comes from facts_v6.json (written by rebuild/scripts/v7/facts_v6.py)
// through {{key}} placeholders. Citations are written [@key] or [@a;@b].
const fs = require("fs");
const L = require("./lib");
const { d, h1, h2, table, rule, ref } = L;
const { Document, Packer, Paragraph, TextRun, AlignmentType, PageBreak, ImageRun,
        convertInchesToTwip } = d;

const REFS = JSON.parse(fs.readFileSync("refs.json", "utf8"));
const FACTS = JSON.parse(fs.readFileSync("facts_v6.json", "utf8"));
const used = new Set();

function S(text) {
  const out = text.replace(/\{\{([^}]+)\}\}/g, (_, k) => {
    if (!FACTS[k]) throw new Error("missing fact " + k);
    used.add(k);
    return FACTS[k].v;
  });
  if (out.includes("—")) throw new Error("em dash in: " + out.slice(0, 80));
  if (/(^|[\s(])-\d/.test(out)) throw new Error("ASCII minus before a number in: " + out.slice(0, 120));
  return out;
}

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
      parts.push(j - i >= 2 ? `${nums[i]}–${nums[j]}` : nums.slice(i, j + 1).join(", "));
      i = j;
    }
    return "[" + parts.join(", ") + "]";
  });
}

// word counts per part of the paper
let part = "front";
const words = {};
function count(t) {
  const n = t.replace(/\*\*/g, "").replace(/\[[\d,\s–]+\]/g, "").split(/\s+/).filter(w => /[A-Za-z0-9]/.test(w)).length;
  words[part] = (words[part] || 0) + n;
}

const B = [];
const add = (...xs) => xs.forEach(x => B.push(x));
const P = (t, o) => { const s = cite(S(t)); count(s); add(L.p(s, o)); };
const cap = (kind, n, t) => add(L.caption(kind, n, cite(S(t))));
const T = (head, rows, widths, opts) => add(table(head.map(S),
  rows.map(r => (r && r.section !== undefined) ? { section: S(r.section) } : r.map(S)), widths, opts));
const H1 = t => add(h1(t));
const H2 = t => add(h2(t));
const list = (items, numbered = true) => items.forEach((t, i) =>
  P(numbered ? `${i + 1}. ${t}` : t, { indent: { left: 240 }, after: 80 }));

function fig(file, wmm) {
  const px = require("child_process").execSync(`python3 -c "from PIL import Image;i=Image.open('${file}');print(i.size[0],i.size[1])"`).toString().trim().split(" ").map(Number);
  const w = Math.round(wmm / 25.4 * 96);
  add(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 80 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(file),
      transformation: { width: w, height: Math.round(w * px[1] / px[0]) } })] }));
}

const TITLE = "Method pool, clinical reference and training seed shape the measured validity of synthetic segmentation benchmarks: a study in wound imaging";

// ------------------------------------------------------------------ title
add(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 160, line: 300 },
  children: [new TextRun({ text: TITLE, font: L.FONT, size: 30, bold: true, color: L.INK })] }));
add(rule(220));

// ------------------------------------------------------------------ abstract
part = "abstract";
H2("Abstract");
P("Synthetic test sets for medical image segmentation are usually validated by showing that they rank methods as a clinical dataset does. We measured how much this figure depends on the methods compared, the clinical reference and training noise. We scored {{n.ALL}} methods on 11 public clinical datasets in four modalities and on parametric synthetic wound benchmarks: {{n.U}} untrained classical methods, {{n.F}} methods trained from scratch on wound photographs, the same {{n.K}} trained on colonoscopy images, and {{n.M19}} modern models ({{n.pre}} ImageNet-pretrained architectures, each trained with three random seeds, and nnU-Net). Agreement was the Spearman correlation of per-method mean Dice. Pooling classical and modern methods made four wound datasets agree at {{a.dice.UM19.wound}}, although they agreed at {{a.dice.U.wound}} among the classical and {{a.dice.M19.wound}} among the modern methods; ranking methods within their own group before pooling removed this effect. Among the modern models, two training seeds of the same architectures agreed at only {{sd.rbar}} on the same wound dataset, and different wound datasets agreed less ({{sd.cross}}). A synthetic benchmark agreed with the wound datasets at {{b.UM19.mean.bench}} over the pooled methods but at {{b.M19.mean.bench}} among the modern models, the level at which the wound datasets agreed with each other. A modern model chosen on one wound dataset lost {{sel.off}} Dice on average on another, against {{sel.rand}} for a random choice, and on one dataset the pretrained architectures missed {{miss.P18.azh}} of wounds against {{miss.nn.azh}} for nnU-Net. The measured validity of a synthetic benchmark depends on the method pool, the clinical reference and the training seed.");
part = "front";
P("**Keywords** Benchmark validity; Rank agreement; Synthetic data; Wound imaging; Image segmentation; Training variance", { after: 200 });

// ------------------------------------------------------------------ introduction
part = "main";
H1("Introduction");
P("Segmentation software is increasingly evaluated before it reaches a clinic, by vendors comparing candidate models, by hospitals validating a product on local data and by regulators considering computational evidence. Synthetic and simulated images are attractive for this purpose because they come with exact ground truth and can be generated in any quantity, and in-silico trials and edited images have already been used to evaluate and stress-test medical models [@badano2018;@sizikova2023;@tushar2026a;@perezgarcia2024]. Rankings of segmentation methods are, however, known to be fragile. They change with the test cases, the metric, the aggregation rule and the annotators [@maierhein2018;@reinke2024], with the choice of benchmark [@dehghani2021;@pierard2025] and with the hospital the test data come from [@bassi2024;@ali2024;@urooj2026].");
P("The usual evidence that a synthetic test set is trustworthy is a rank correlation between method scores on the synthetic set and on one clinical dataset, sometimes expressed relative to the agreement between two real datasets [@kang2022;@borsukiewicz2026]. Synthetic data have been proposed for model validation and selection on the same basis [@hu2023;@vanbreugel2023;@shoshan2023], and in computed tomography the transfer of rankings from synthetic to real data has been shown to depend on the task [@tushar2026b]. Such a correlation is computed over a particular set of methods. Outside medical imaging it is established that agreement between benchmarks depends on which models are compared [@perlitz2024;@zhang2026], and that the correlation between in-distribution and out-of-distribution performance depends on how the models were selected and trained [@teney2023;@miller2021;@fang2022]. When groups of methods change order between domains, pooling them produces correlations that hold in neither group, a form of Simpson's paradox [@simpson1951;@kievit2013].");
P("Medical segmentation studies often pool exactly such groups. Wound segmentation studies, for example, compare classical colour and threshold methods with trained networks [@lucho2024;@benfatah2026], and a synthetic benchmark validated over such a pool inherits whatever the pooling does to the correlation. A second source of variation lies in training itself. Retraining the same network with another random seed changes its score, and for closely matched models these changes can be as large as the differences between models [@bouthillier2021;@varoquaux2022]. A ranking of modern models on one dataset may then be partly a ranking of training runs, and two datasets cannot agree on an order that training noise does not preserve. Decision thresholds add a further choice, because networks differ in how confident their outputs are [@guo2017]. How large these effects are for segmentation, and what they imply for validating a synthetic benchmark, has not been measured.");
P("We scored {{n.ALL}} methods in four homogeneous groups on 11 clinical datasets from eight sources in four modalities and on synthetic wound benchmarks from three generator families. The modern group contains {{n.pre}} ImageNet-pretrained architectures, each trained with three random seeds, and nnU-Net. We asked how much pooling groups of methods changes agreement between clinical datasets, whether agreement is organised by modality within groups, how much of the disagreement among modern models is training noise, and what this implies for the measured validity of a synthetic benchmark and for choosing a model. The contributions are:");
list([
  "A measurement of the composition effect in segmentation benchmark agreement. Pooling untrained with modern methods raises agreement between wound datasets from {{a.dice.U.wound}} and {{a.dice.M19.wound}} within the groups to {{a.dice.UM19.wound}}, and an exact identity shows that ranking methods within their groups before pooling returns the size-weighted average of the within-group agreements.",
  "A demonstration that the measured validity of a synthetic benchmark follows the method pool and the clinical reference: the same benchmark agrees with wound data at {{b.UM19.mean.bench}} or at {{b.M19.mean.bench}} depending on whether untrained methods are in the pool, and its agreement with single wound datasets varies widely within one pool.",
  "Evidence from {{n.pre}} architectures trained with three seeds that rankings of modern models on wound data are dominated by small differences and training noise, with the consequences for model selection and a failure mode on one dataset that mean Dice hides.",
]);

// ------------------------------------------------------------------ methods
H1("Materials and Methods");
H2("Clinical datasets");
P("Eleven public datasets serve as clinical references (Table 1). The wound photographs are Medetec [@medetec2014], CO2Wounds-V2 [@sanchez2024], the FUSeg challenge validation set [@wang2024] and the AZH chronic wound dataset [@wang2020]. FUSeg and AZH were collected at one wound centre and are treated as one source; CO2Wounds-V2 contains 764 smartphone photographs from 96 leprosy patients in Colombia [@sanchez2024] and shares no source with any other dataset. The colonoscopy datasets are Kvasir-SEG [@jha2020] and its sessile subset, CVC-ClinicDB [@bernal2015], CVC-ColonDB [@bernal2012] and ETIS-LaribPolypDB [@silva2014]; the two CVC sets come from one group and are one source. PH2 [@mendonca2013] is the dermoscopy reference and the TN3K test split [@gong2021] the thyroid ultrasound reference. A disjoint slice of the AZH training set serves as a within-dataset control.", { after: 100 });
P("The Kvasir reference was drawn only from images outside the sessile subset, so the two Kvasir references share no image. For CO2Wounds-V2 and the five other non-wound datasets, 150 image and mask pairs were taken evenly spaced over the files sorted by number, because consecutive colonoscopy frames are near duplicates. Images in these six sets were resampled so that the long side is at most 256 pixels; wound masks were binarised at any non-zero value and all others above 127. Several images can come from one photograph: Medetec's {{n.img.med}} images are {{n.unit.med}} photographs with four variants each, and the {{n.img.azh}} AZH test images and {{n.img.ctrl}} control images are crops of {{n.unit.azh}} and {{n.unit.ctrl}} photographs. The photograph is therefore the resampling unit for these three sets in every analysis, and the image for all others.", { after: 100 });
P("Duplicated evaluation images were removed before any analysis. In FUSeg, {{n.excl.fus}} validation images are byte-identical to FUSeg training images ({{n.excl.fus.train}}) or to another validation image ({{n.excl.fus.eval}}). Feature matching (SIFT with RANSAC, at least {{n.excl.ctrl.mininl}} geometrically consistent matches) showed that {{n.excl.ctrl}} of the {{n.ctrl.before}} control images are crops of FUSeg photographs, {{n.excl.ctrl.train}} from its training split and {{n.excl.ctrl.val}} from its validation split; they were removed, leaving {{n.img.ctrl}}. A fifth wound dataset (WoundSeg) was prepared but excluded before any method was scored on it, because {{ws.bad}} of its {{ws.n}} masks did not match the size of their image, so the pairing of images and masks could not be trusted. The exclusions are listed in Online Resource 1 (Table S9).");
cap("Table", 1, "Clinical datasets after removal of duplicated images. Units are the resampling units (photographs for Medetec and the AZH sets, images otherwise). Lesion fraction is the median share of image pixels inside the reference mask");
const DS = [
  ["Medetec", "Wound photography", "Medetec archive", "med"],
  ["CO2Wounds-V2", "Wound photography", "Leprosy clinic, Colombia", "co2"],
  ["FUSeg (validation)", "Wound photography", "AZH Wound Center", "fus"],
  ["AZH (test)", "Wound photography", "AZH Wound Center", "azh"],
  ["AZH train slice (control)", "Wound photography", "AZH Wound Center", "ctrl"],
  ["Kvasir", "Colonoscopy", "Kvasir collection", "kva"],
  ["Kvasir-sessile", "Colonoscopy", "Kvasir collection", "kvs"],
  ["CVC-ClinicDB", "Colonoscopy", "CVC, Barcelona", "cvc1"],
  ["CVC-ColonDB", "Colonoscopy", "CVC, Barcelona", "cvc2"],
  ["ETIS-Larib", "Colonoscopy", "ETIS, Paris", "etis"],
  ["PH2", "Dermoscopy", "Porto", "ph2"],
  ["TN3K (test)", "Thyroid ultrasound", "TN3K", "tn3k"],
];
T(["Dataset", "Modality", "Source", "Images", "Units", "Empty masks", "Lesion fraction"],
  DS.map(([a, b, c, k]) => [a, b, c, `{{n.img.${k}}}`, `{{n.unit.${k}}}`, `{{n.empty.${k}}}`, `{{frac.${k}}}`]),
  [2150, 1750, 1950, 750, 700, 950, 1110], { leftCols: 3 });

H2("Method groups");
P("The {{n.ALL}} methods form four groups that differ in training and in training domain (Table 2; every method is listed in Online Resource 1, Table S1). The {{n.U}} untrained methods (U) are thresholding, colour, clustering and region-growing algorithms with no learned parameters. The {{n.F}} from-scratch methods (F) are three pixel classifiers (two logistic regressions and gradient boosting on colour and texture features), a three-level U-Net with 117,681 parameters and ten architectures from segmentation_models_pytorch [@iakubovskii2019] with randomly initialised encoders: U-Net [@ronneberger2015], U-Net++ [@zhou2018], DeepLabV3+ [@chen2018], FPN [@lin2017], PSPNet [@zhao2017], LinkNet [@chaurasia2017] and MA-Net [@fan2020] on ResNet-18 [@he2016], U-Net on MobileNetV2 [@sandler2018] and EfficientNet-B0 [@tan2019], and SegFormer on MiT-B0 [@xie2021].", { after: 100 });
P("The from-scratch group was trained on FUSeg training images. The pixel classifiers were fitted on 247,975 pixels sampled from images 0 to 249, balanced between lesion and background; one decision threshold (0.90) was swept for the logistic regression on images 250 to 399 and applied to all three. The small U-Net uses a threshold of 0.5. The ten architectures were trained on images 0 to 392 at 128 by 128 pixels for 30 epochs (Dice plus binary cross-entropy, Adam, learning rate 0.001, horizontal flips) and each threshold was swept on images 400 to 549. The Kvasir-trained group (K) repeats this protocol exactly on Kvasir-SEG images that appear in neither Kvasir reference nor the sessile subset (pixel threshold 0.60).", { after: 100 });
P("The modern group (M19) contains {{n.pre}} architectures with ImageNet-pretrained encoders [@deng2009] and nnU-Net. Ten are the architectures of the from-scratch group with pretrained encoders. Eight were added to cover larger encoders and current designs: U-Net, DeepLabV3+ and MA-Net on ResNet-50, U-Net on EfficientNet-B3 [@tan2019] and on Swin-T [@liu2021], SegFormer on MiT-B2 [@xie2021], UPerNet [@xiao2018] on ConvNeXt-T [@liu2022] and FPN on RegNetY-1.6GF [@radosavovic2020]. All {{n.pre}} were trained with one protocol on FUSeg training images 0 to 392 with the one image duplicated in the evaluation set removed ({{n.train}} images). Images were letterboxed so that the long side is 256 pixels, with the aspect ratio kept and the remainder padded with black pixels. Training used Dice plus binary cross-entropy, AdamW (learning rate 0.0003, weight decay 0.0001, cosine decay), batches of 16, {{ep.pre}} epochs, random horizontal and vertical flips and 90-degree rotations, and mixed-precision arithmetic with gradient scaling. Each model's decision threshold was chosen on a grid from 0.05 to 0.90 in steps of 0.05 to maximise mean Dice on {{n.sweep}} held-out FUSeg training images (images 400 to 549 without two duplicates of evaluation images). Held-out Dice of the {{n.pre.models}} individual models ranged from {{ho.pre}}.", { after: 100 });
P("Each architecture was trained with three random seeds (0, 1 and 2). The seed sets the initial weights of the parts not taken from ImageNet, the order of the training batches and the augmentation draws. Seed 0 of the ten original architectures was trained in a first GPU session; seeds 1 and 2 of these and all seeds of the eight new architectures were trained in a second session with the same training code (segmentation_models_pytorch {{ver.smp}}, timm {{ver.timm}} and PyTorch {{ver.torch}} were recorded in the second session; the first session did not record the PyTorch and timm versions). Of the {{n.run2}} trainings in the second session, {{n.run2.ok}} produced a usable model. The {{n.run2.fail}} that did not were MA-Net trainings ({{n.manet.fail}} of {{n.manet.run2}}: ResNet-18 at seed 2 and ResNet-50 at seeds 0 and 2), in which mixed-precision arithmetic produced non-finite values; the other {{n.other.run2}} trainings had none. Failed trainings were not imputed. In the main analyses each architecture is represented by the per-image mean of its scores over the seeds that exist (its seed average), so MA-Net on ResNet-18 averages two models and MA-Net on ResNet-50 is a single model.", { after: 100 });
P("nnU-Net v2 [@isensee2021] was trained once, in its 2D configuration with the default trainer, on all {{n.train}} training images (one model, no cross-validation folds) for 100 epochs instead of the default 1,000. Its foreground probability is thresholded at 0.5; its Dice on the {{n.sweep}} held-out images is {{ho.nn}}. To produce inputs for the robustness analyses nnU-Net was run again in the second session on the same images. Per-image Dice differed from the first run by at most {{nn.img.maxdiff}} and dataset means by at most {{nn.mean.maxdiff}}, with no change of rank, so all main analyses use the first-run scores (Online Resource 1, Table S9).", { after: 100 });
P("SAM [@kirillov2023] and MedSAM [@ma2024] were also run, with box prompts taken from the reference masks. Because those prompts give them the lesion location, they are reported only in Online Resource 1 (Table S8) and are not part of any method set.", { after: 100 });
P("Untrained methods receive morphological closing and opening with a 5-pixel elliptical element. Methods that output a probability map receive Gaussian smoothing with a standard deviation of 2 pixels before thresholding. Every method then has connected components smaller than 0.05% of the image removed.");
cap("Table", 2, "Method groups. Mean Dice is the range of group means (mean over methods of the per-method mean) over the four wound datasets and over the seven other clinical datasets");
T(["Group", "n", "Training data", "Training seeds", "Mean Dice, wound", "Mean Dice, other"],
  [
    ["Untrained (U)", "{{n.U}}", "none", "none", "{{md.U.wound}}", "{{md.U.other}}"],
    ["From scratch (F)", "{{n.F}}", "FUSeg", "one", "{{md.F.wound}}", "{{md.F.other}}"],
    ["Kvasir-trained (K)", "{{n.K}}", "Kvasir-SEG", "one", "{{md.K.wound}}", "{{md.K.other}}"],
    ["Modern (M19)", "{{n.M19}}", "FUSeg, ImageNet encoders", "three per architecture; one for nnU-Net", "{{md.M19.wound}}", "{{md.M19.other}}"],
  ],
  [1800, 500, 2100, 2000, 1480, 1480], { leftCols: 4 });

H2("Synthetic benchmarks");
P("Three parametric generator families produce wound images with exact masks, 45 images per benchmark in three severity classes. Generator A places a lesion on a uniform drape and reproduces the benchmark released with an earlier system description [@islam2026]. Generator B renders textured skin with clutter such as hair and gauze, and its background texture amplitude is a parameter; it was built at amplitudes 11, 28 and 40. Generator C shares no code with B and uses spectral noise, vignetting and a colour mixture model. Every benchmark was built at three synthetic seeds (2024, 7 and 101). Background texture is the median local standard deviation of shading-corrected luminance in a 9-pixel window over background pixels, measured at 512 pixels.");

H2("Agreement between image sets");
P("Agreement between two image sets is the Spearman correlation between their vectors of per-method mean Dice over a given set of methods. The 55 pairs of the 11 clinical datasets are grouped by level: within source (one clinic or group, no shared images; 3 pairs), within modality from different sources (13 pairs) and across modality (39 pairs). The within-dataset control compares AZH with the disjoint AZH slice. The modality difference is mean same-modality agreement (16 pairs) minus mean across-modality agreement (39 pairs). For the composition analyses each pair is assigned to one of four categories relative to a domain, the modality a trained group was trained on. For wound photography these are domain with domain (the 6 wound pairs), domain with other datasets (28 pairs), other datasets of the same modality (the 10 colonoscopy pairs) and other datasets across modality (11 pairs). Method sets are the four groups alone and the pooled sets U+F ({{n.UF}} methods), U+K ({{n.UK}}) and U+M19 ({{n.UM19}}). Every analysis was repeated with normalised surface Dice (NSD) at a tolerance of 2 pixels [@nikolov2018] and with the images whose reference mask is empty excluded.", { after: 100 });
P("Pairs share datasets, so tests on pairs overstate the evidence. The modality difference was therefore tested by assigning the modality labels to the 11 datasets in every distinct way ({{n.relabel}} assignments) and counting how often the difference was at least as large as observed. Swapping the labels of the two single-dataset modalities gives the same partition of pairs, so the smallest attainable p is 2/{{n.relabel}} = {{p.min}}. Because the three within-source pairs could carry the result alone, the test was repeated with them dropped under every relabeling (52 pairs).", { after: 100 });
P("Two bootstrap designs give 95% percentile intervals. The method bootstrap resamples methods with replacement within each group, so a pooled set keeps its composition (2,000 draws). The hierarchical bootstrap also resamples the units of every dataset independently (1,000 draws), using photographs as units for Medetec ({{n.unit.med}} photographs), AZH ({{n.unit.azh}}) and the control ({{n.unit.ctrl}}), and images for the other datasets. Resampling units adds noise to every method mean and so pulls agreement towards zero in each draw; hierarchical intervals therefore sit partly below the estimate and are conservative for claims that agreement exceeds zero. Intervals in the text are method-bootstrap intervals unless marked hierarchical. Analyses of the synthetic benchmarks that involve untrained or from-scratch methods use the method bootstrap only, because per-image synthetic scores exist only for the GPU-trained groups. Image sampling within a dataset was also summarised by split-half reliability: the Spearman correlation between method means on two random halves of the units, corrected with the Spearman-Brown formula (median of 500 splits).");

H2("Stratified agreement and why it equals the within-group average");
P("As a remedy for pooled sets we computed stratified agreement. Within each group g of size n_g, a method's mean on a dataset is replaced by its within-group percentile (r − 0.5)/n_g, where r is its rank in the group, and agreement is the Pearson correlation of these percentiles over the pooled set of N = Σ n_g methods. The percentiles have mean 1/2 in every group, so the pooled mean is also 1/2 and the between-group part of the pooled covariance is zero. The pooled covariance of the percentiles on datasets A and B is therefore (1/N) Σ n_g c_g, and each pooled variance is (1/N) Σ n_g v_g, where c_g and v_g are the covariance and variance within group g. Without ties, v_g = (1 − 1/n_g²)/12 on every dataset, and within a group the correlation of percentiles is the Spearman correlation r_g. Hence stratified agreement = Σ w_g r_g / Σ w_g with w_g = n_g − 1/n_g, a size-weighted average of the within-group agreements. The weights do not depend on the pair, so the identity also holds for means over pairs. With ties the variances differ slightly between datasets and the identity holds approximately. Stratified agreement thus adds no information beyond the within-group agreements; what it shows is how much of a pooled value comes from differences in level between groups. In our data the Pearson value matched the identity to rounding error, and ranking the pooled percentiles again (a Spearman version) changed single pairs by at most {{s.maxpair}} under Dice.");

H2("Training-seed analyses");
P("In total {{n.C16}} architectures have all three seeds: nine of the original ten and seven of the eight new ones. The two MA-Net architectures lack seeds and nnU-Net has one training, so these three are left out of the seed analyses. For each wound dataset, same-dataset cross-seed agreement is the mean Spearman correlation between the architecture means of seeds s and t on the same images, over the three seed pairs. Its counterpart across datasets correlates seed s on one wound dataset with a different seed t on another (36 correlations over the 6 wound pairs), so that training noise is present in both and only the dataset differs. The gap between the two measures how much more two datasets disagree than two trainings do. We also compared wound-pair agreement at single seeds (mean over the three seeds) with agreement of the seed averages, and set the spread of architecture means on each dataset against the seed-to-seed standard deviation of one architecture's mean (pooled over the {{n.C16}} architectures). Intervals resample architectures (2,000 draws); hierarchical intervals also resample units (1,000 draws). One reading rule was fixed before the new trainings were scored: the statement that modern models disagree across wound datasets because the datasets differ, not because of training noise, would be supported only if same-dataset cross-seed agreement exceeded 0.7 while cross-dataset agreement stayed low. Details are in Online Resource 1 (Table S4).");

H2("Synthetic benchmark analyses");
P("For each wound reference, benchmark agreement is the Spearman correlation between generator B at amplitude 40 and that reference, averaged over the three synthetic seeds. It is compared with the mean agreement of the other three wound datasets with the same reference; the difference (benchmark minus other) is positive when the benchmark ranks methods more like the reference than another wound dataset does. Averaged over the four references, the second term equals the mean agreement of the six wound pairs. The texture effect is benchmark agreement at amplitude 40 minus agreement at amplitude 11. Benchmark agreement was also computed for single-seed versions of the modern group, in which every architecture is represented by its model at one training seed, together with nnU-Net.");

H2("Model selection and failure modes");
P("For the modern models, the selection loss from dataset A to dataset B is the best mean Dice on B minus the mean Dice on B of the candidate ranked first on A. The expected loss of a random choice is the mean over candidates of the best mean Dice minus the candidate's. The synthetic chooser picks the candidate with the highest mean Dice on generator B (mean over synthetic seeds). Selection losses are small Dice differences and are given to three decimals. A miss is a non-empty reference mask segmented with Dice below 0.1. Miss rates were computed for every individual trained model and averaged over seeds per architecture, because a seed-averaged score is not the output of any model; their intervals resample units (2,000 draws) with the models fixed. To look for the cause of the misses on AZH we related, for each image, the share of pretrained models that miss it to its share of black padding pixels (every colour channel at most 8) and to its lesion fraction.");

H2("Robustness analyses");
P("The main analyses were repeated with two other post-processing pipelines (P1: threshold, closing and opening with a 5-pixel element and small-component removal; P2: threshold and small-component removal only), with NSD at tolerances of 1, 3 and 5 pixels and at 1% of the image diagonal, and with FUSeg resampled so that its long side is 256 pixels, the resolution of most other datasets. For the modern models the threshold protocol was also varied: one common threshold for all models (each value of the grid from 0.05 to 0.90), and a per-dataset oracle threshold for every model, chosen on the evaluation images themselves to remove calibration differences. These reanalyses used probability maps rescored in the second session; for the pooled sets the untrained and from-scratch methods were rescored locally with the same variants. Three Kvasir-trained pixel classifiers were not rescored, so robustness results for the Kvasir group use the remaining 11 methods (Online Resource 1, Table S6).");

H2("Confirmatory and exploratory analyses");
P("The composition effect of pooling and the modality structure within groups were the confirmatory analyses of the study design and are tested with the exact relabeling test. All other analyses, including the training-seed, benchmark, selection and robustness analyses, are exploratory and are reported with intervals; no correction for multiple comparisons was applied.");

// ------------------------------------------------------------------ results
H1("Results");
H2("The groups differ in level and in domain");
P("The groups differ in level and in domain (Table 2). The modern models reach group mean Dice of {{md.M19.wound}} on the wound datasets but {{md.M19.other}} on the seven other datasets, and the from-scratch group shows the same pattern at a lower level. The Kvasir-trained group leads on colonoscopy ({{md.K.colon}}) and is close to the untrained group on wound photographs ({{md.K.wound}} against {{md.U.wound}}). Each dataset ranks the methods reproducibly: split-half reliability over all {{n.ALL}} methods is {{rel.ALL}} on the 11 datasets, and among the {{n.M19}} modern models it is {{rel.M19.wound}} on the four wound datasets.");

H2("Pooling creates agreement and disagreement");
P("Pooling the untrained group with the modern group changes agreement in both directions (Fig. 1a, Table 3). Over the pooled {{n.UM19}} methods the four wound datasets agree at {{a.dice.UM19.wound}} ({{a.dice.UM19.wound.m}}), against {{a.dice.U.wound}} ({{a.dice.U.wound.m}}) among untrained methods and {{a.dice.M19.wound}} ({{a.dice.M19.wound.m}}) among modern models; the pooled value exceeds the higher of the two by {{rp.UM19.gap.P0}} ({{rp.UM19.gap.P0.m}}). Agreement between wound and other datasets becomes {{a.dice.UM19.w_in_out}} ({{a.dice.UM19.w_in_out.m}}), against {{a.dice.U.w_in_out}} and {{a.dice.M19.w_in_out}} within the groups, and non-wound datasets of different modalities appear to agree at {{a.dice.UM19.w_out_across}} ({{a.dice.UM19.w_out_across.m}}), against {{a.dice.U.w_out_across}} and {{a.dice.M19.w_out_across}}. Under NSD the pooled wound-pair agreement is {{a.nsd.UM19.wound}}, against {{a.nsd.U.wound}} and {{a.nsd.M19.wound}} within the groups. The full agreement matrices are shown in Online Resource 1 (Fig. S1).", { after: 100 });
P("The from-scratch group moves agreement in the same direction by less (Fig. 1b). Pooled with the untrained methods, wound datasets agree at {{a.dice.UF.wound}}, against {{a.dice.U.wound}} and {{a.dice.F.wound}} within the groups, but the excess over the untrained group, {{rp.UF.gap.P0}} ({{rp.UF.gap.P0.m}}), is not distinguishable from zero. Agreement between wound and other datasets turns negative, {{a.dice.UF.w_in_out}} ({{a.dice.UF.w_in_out.m}}), against {{a.dice.U.w_in_out}} and {{a.dice.F.w_in_out}} within the groups.", { after: 100 });
P("The mechanism is visible in the group means. The modern models lead the untrained methods on every wound dataset and fall below them on every other dataset (Table 2; per-dataset means in Online Resource 2), so a pooled correlation between a wound and a non-wound dataset is dominated by which group each method belongs to, without any pair of methods changing order within a group. The Kvasir-trained group is the mirror experiment. It leads on colonoscopy, but on wound photographs its means ({{md.K.wound}}) are close to those of the untrained methods ({{md.U.wound}}), so the two groups do not change places across modalities. Pooling it with the untrained group pulls agreement between colonoscopy and other datasets below both groups ({{a.dice.UK.c_in_out}}, against {{a.dice.U.c_in_out}} and {{a.dice.K.c_in_out}}) but does not reverse its sign (Fig. 1c).", { after: 100 });
P("Stratified agreement removes the effect and, as derived in the Methods, equals the size-weighted average of the within-group agreements. For U+M19 the stratified wound-pair agreement is {{s.dice.UM19.wound}} (size-weighted within-group average {{s.dice.UM19.wound.sw}}; pooled {{s.dice.UM19.wound.pool}}), and the stratified modality difference is {{s.dice.UM19.mdiff}}, against {{s.dice.UM19.mdiff.pool}} pooled. For U+F and U+K the stratified modality differences are {{s.dice.UF.mdiff}} and {{s.dice.UK.mdiff}}, against {{s.dice.UF.mdiff.pool}} and {{s.dice.UK.mdiff.pool}} pooled.");
fig("figures/Fig1_pooling.png", 165);
cap("Figure", 1, "Mean Spearman agreement between pairs of clinical datasets by category, for each group alone and pooled (Dice; 95% method-bootstrap intervals). The domain is the modality the trained group was trained on. a: untrained (U) and modern (M19) methods. b: untrained and from-scratch (F) methods. c: untrained and Kvasir-trained (K) methods");
cap("Table", 3, "Mean Spearman agreement between clinical datasets by level (Dice), wound-pair agreement, and the modality difference (same-modality minus across-modality agreement) with its 95% method-bootstrap interval, exact relabeling p and NSD value. Method sets: U untrained, F from scratch, K Kvasir-trained, M19 modern; + denotes pooling");
const T3 = [["U", "U"], ["F", "F"], ["K", "K"], ["M19", "M19"], ["U+F", "UF"], ["U+K", "UK"], ["U+M19", "UM19"]];
T(["Methods", "n", "Within dataset", "Within source", "Within modality", "Across modality", "Wound pairs", "Modality difference (95% interval)", "p", "NSD difference"],
  T3.map(([lab, k]) => [lab, `{{n.${k}}}`, `{{a.dice.${k}.ctrl}}`, `{{a.dice.${k}.src}}`, `{{a.dice.${k}.dsrc}}`,
    `{{a.dice.${k}.across}}`, `{{a.dice.${k}.wound}}`, `{{a.dice.${k}.mdiff}} ({{a.dice.${k}.mdiff.m}})`,
    `{{p.dice.${k}}}`, `{{a.nsd.${k}.mdiff}}`]),
  [880, 450, 850, 850, 900, 900, 850, 1700, 830, 950], { leftCols: 1, size: 16 });
part = "note";
P("Within modality: different sources. The smallest attainable p is {{p.min}}.", { size: 16, after: 160, color: L.GREY });
part = "main";

H2("Modality structure is clear within the untrained and Kvasir-trained groups");
P("Within the untrained and Kvasir-trained groups, datasets of the same modality agree more than datasets of different modalities: the modality difference is {{a.dice.U.mdiff}} (p = {{p.dice.U.t}}) and {{a.dice.K.mdiff}} (p = {{p.dice.K}}) (Table 3). Both groups score above the floor on most non-wound datasets. The from-scratch group shows the same structure ({{a.dice.F.mdiff}}, p = {{p.dice.F}}), although its group mean Dice is below 0.1 on {{md.F.nfloor}} of the seven non-wound datasets. For these three groups the difference holds without the three within-source pairs ({{d52.UFK.range}}, p at most {{p52.UFK.max}}), under NSD ({{mdiff.nsd.UFK.range}}, p at most {{p.nsd.UFK.max}}) and with empty masks excluded ({{ne.UFK.range}}, p at most {{ne.UFK.pmax}}; Online Resource 1, Tables S2 and S3). Colonoscopy datasets from three sources agree at {{a.dice.U.colon}}, {{a.dice.F.colon}} and {{a.dice.K.colon}} within the three groups, whereas the wound datasets agree at {{a.dice.U.wound}}, {{a.dice.F.wound}} and {{a.dice.K.wound}}.", { after: 100 });
P("For the modern models the same statistic cannot be interpreted. Their group mean Dice is at most {{md.M19.other.max}} on each of the seven non-wound datasets, so their rankings there order near-zero scores. The modality difference among the {{n.M19}} modern models is {{a.dice.M19.mdiff}} (p = {{p.dice.M19}}), but it depends on the training seed. For the ten original architectures with nnU-Net it is {{sdg.M.mdiff}} with the seed-0 models, {{sdg.Ms1.mdiff}} with the seed-1 models (p = {{p.Ms1}}) and {{sdg.Ms2.mdiff}} with the seed-2 models (nine architectures; p = {{p.Ms2}}), and the eight new architectures show none ({{a.dice.N8s.mdiff}}, p = {{p.dice.N8s}}). We therefore do not count the modern group as evidence of modality structure.");

H2("Synthetic benchmark validity follows the method pool");
P("Generator B at texture amplitude 40 agrees with the four wound datasets at {{b.UM19.mean.bench}} ({{b.UM19.mean.bench.m}}) over the pooled untrained and modern methods, a value that would usually be read as good validation (Fig. 2, Table 4). Among the modern models alone it agrees at {{b.M19.mean.bench}} ({{b.M19.mean.bench.m}}), and the wound datasets agree with each other at the same level ({{b.M19.mean.other}}; difference {{b.M19.mean.diff}}, {{b.M19.mean.diff.m}}; hierarchical {{bh.M19.mean.diff}}). Over the pooled set the benchmark is a worse proxy than another wound dataset: it agrees with each wound dataset less than the other wound datasets do, by {{b.UM19.mean.diff}} on average ({{b.UM19.mean.diff.m}}). Among untrained methods it agrees at {{b.U.mean.bench}} ({{b.U.mean.bench.m}}); for the from-scratch and Kvasir-trained groups the intervals include zero.", { after: 100 });
P("Validity per reference varies widely. Among untrained methods the benchmark agrees with CO2Wounds-V2 at {{b.U.co2.bench}} and with AZH at {{b.U.azh.bench}}; among Kvasir-trained methods it agrees with AZH at {{b.K.azh.bench}} and with FUSeg at {{b.K.fus.bench}}; among modern models its agreement with single wound datasets ranges from {{b.M19.range}} (Table 4). Reported against a single dataset, the same benchmark and pool could be described as valid or invalid.", { after: 100 });
P("Background texture raises validity only in pooled sets. Moving generator B from amplitude 11 to 40 raises mean benchmark agreement by {{tx.UM19}} ({{tx.UM19.m}}) in U+M19, with every per-reference interval above zero, and by {{tx.UF}} ({{tx.UF.m}}) in U+F. Within the groups the change is {{tx.U}}, {{tx.F}}, {{tx.K}} and {{tx.M19}} (U, F, K and M19), and only {{tx.within.n}} of the {{tx.within.tot}} per-reference intervals excludes zero, about what chance would give.", { after: 100 });
P("Benchmark agreement among modern models also depends on the training seed. With single-seed versions of the ten original architectures it ranges from {{bs.P10.range}} over the three seeds, for the eight new architectures from {{bs.N8.range}}, and for the full modern group from {{bs.M19.range}}; the agreement between the wound datasets themselves moves with the seed as well ({{bs.P10.other.range}} for the original ten). Leaving out one modern method moves benchmark agreement within {{lomo.M19.bench}}. Comparisons between generator families also changed with the pool and the references (Online Resource 1, Table S5).");
fig("figures/Fig2_benchmark_validity.png", 140);
cap("Figure", 2, "Agreement of generator B (texture amplitude 40, mean over three synthetic seeds) with the four clinical wound datasets, and agreement of those datasets with each other, for each group and pooled set (Dice; 95% method-bootstrap intervals)");
cap("Table", 4, "Agreement of generator B (amplitude 40, mean over three synthetic seeds) with each wound dataset by method set, the mean over the four, the mean agreement of the wound datasets with each other, and the difference between the two (Dice; 95% method-bootstrap intervals)");
T(["Methods", "Medetec", "CO2Wounds-V2", "FUSeg", "AZH", "Mean (95% interval)", "Wound with wound", "Difference (95% interval)"],
  T3.map(([lab, k]) => [lab, `{{b.${k}.med.bench}}`, `{{b.${k}.co2.bench}}`, `{{b.${k}.fus.bench}}`, `{{b.${k}.azh.bench}}`,
    `{{b.${k}.mean.bench}} ({{b.${k}.mean.bench.m}})`, `{{b.${k}.mean.other}}`, `{{b.${k}.mean.diff}} ({{b.${k}.mean.diff.m}})`]),
  [900, 900, 1150, 850, 850, 1750, 1150, 1810], { leftCols: 1, size: 16 });

H2("Rankings among modern models are weak and noisy");
P("Among the {{n.M19}} modern models, the four wound datasets agree at {{a.dice.M19.wound}} on average (hierarchical interval {{a.dice.M19.wound.h}}), although each ranks the models reproducibly (split-half reliability {{rel.M19.wound}}) and the AZH control agrees with AZH at {{a.dice.M19.ctrl}} (hierarchical {{a.dice.M19.ctrl.h}}). Under NSD wound-pair agreement is {{a.nsd.M19.wound}} (hierarchical {{a.nsd.M19.wound.h}}). The models are close together: on each wound dataset the {{n.M19}} method means span {{sp.range}} Dice, and the median gap between neighbours in the ranking is {{sp.gap}}.", { after: 100 });
P("Training noise is large (Fig. 3a, Table 5). Two seeds of the same {{n.C16}} architectures, scored on the same wound images, agree at {{sd.rbar}} on average (hierarchical interval {{sd.rbar.h}}; {{sd.rbar.range}} for the individual datasets), against {{sd.rbar.ctrl}} on the control images. Retraining one architecture with another seed moves its dataset mean with a pooled standard deviation of {{sp.seedsd}} Dice, several times the median gap between neighbours. The rule fixed before the seed experiment required same-dataset cross-seed agreement above 0.7. It was not met, so the statement that modern models disagree across wound datasets because the datasets differ, rather than because of training noise, is not supported in that form.", { after: 100 });
P("The datasets nevertheless disagree more than the seeds do. Agreement between different wound datasets at different seeds is {{sd.cross}} (hierarchical {{sd.cross.h}}), so the gap to same-dataset agreement is {{sd.gap}} ({{sd.gap.m}}; hierarchical {{sd.gap.h}}). The gap comes from the nine original architectures ({{sd.gap.P9}}); among the seven new ones it is {{sd.gap.N7}}, so for them disagreement across datasets is no larger than disagreement between seeds. Under NSD the gap is {{sdn.gap}} ({{sdn.gap.m}}). Averaging three seeds did not raise agreement across datasets: wound-pair agreement was {{sd.single}} for single seeds (mean over three) and {{sd.avg}} for the seed averages, a difference of {{sd.avgminus}} ({{sd.avgminus.m}}). A single-seed ranking on one wound dataset has a reliability (intraclass correlation) of {{sp.icc1}}, and a three-seed average one of {{sp.icc3}} (Online Resource 1, Table S4).", { after: 100 });
P("The level of agreement also depends on how thresholds are set (Fig. 3b). With each model at its own FUSeg-tuned threshold, the protocol used throughout, wound-pair agreement is {{a.dice.M19.wound}}. With one common threshold of 0.5 for all models it rises to {{thr.c50}} ({{thr.c50.m}}; change {{thr.c50.d}}, {{thr.c50.d.m}}), and the rise has an interval above zero for every common threshold from {{thr.above.lo}} to {{thr.above.hi}}, reaching {{thr.max}} at {{thr.max.t}}. At a common threshold the ranking follows how confident each model's outputs are: the mean Spearman correlation between the method means on the wound datasets and each model's own tuned threshold is {{cal.default}} under the main protocol and {{cal.c50}} at a common threshold of 0.5. With a per-dataset oracle threshold for every model, which removes calibration differences, agreement falls to {{thr.oracle}} ({{thr.oracle.m}}).");
fig("figures/Fig3_modern_noise.png", 165);
cap("Figure", 3, "Agreement among modern models on the four wound datasets (Dice). a: the {{n.C16}} architectures with three training seeds; agreement between two seeds on the same dataset (per dataset and mean), between two datasets at the same seed (each of the six pairs and their mean), between two datasets at different seeds, and on the control (AZH train slice), with 95% architecture-bootstrap intervals. b: mean wound-pair agreement of the {{n.M19}} modern models when one common threshold is applied to every model, with 95% method-bootstrap band; filled markers show a rise over each model's own tuned threshold with interval above zero; the dashed line marks each model's own threshold and the dotted line a per-dataset oracle threshold");

H2("Consequences for choosing a modern model");
P("The best model differs between wound datasets: {{sel.best.med}} on Medetec, {{sel.best.co2}} on CO2Wounds-V2, {{sel.best.fus}} on FUSeg and {{sel.best.azh}} on AZH (seed averages), and with single-seed models the best architecture changes with the seed on {{sel.bestchange.n}} of the four datasets. Choosing on one wound dataset and deploying on another loses {{sel.off}} Dice on average over the 12 ordered pairs (at most {{sel.off.max}}), against {{sel.rand}} for a random choice (Table 5). The synthetic benchmark picks {{sel.syn.pick}} at every texture amplitude, with a mean loss of {{sel.syn}}. This pick depends on the training seed. With the seed-0 models the benchmark picks {{sel.s0.pick}} (mean loss {{sel.s0.loss}}), an architecture whose synthetic Dice is {{eff.s0}}, {{eff.s1}} and {{eff.s2}} at seeds 0, 1 and 2; with the seed-1 models the pick is {{sel.s1.pick}} again, but its mean loss is {{sel.s1.loss}} because the best models change (Online Resource 1, Table S7).", { after: 100 });
P("Mean Dice hides a failure mode. On AZH the {{n.pre}} pretrained architectures miss (Dice below 0.1) {{miss.P18.azh}} ({{miss.P18.azh.ci}}) of the {{miss.azh.nonempty}} images with a non-empty reference on average, against {{miss.nn.azh}} ({{miss.nn.azh.ci}}) for nnU-Net; on FUSeg the rates are {{miss.P18.fus}} and {{miss.nn.fus}}. The new architectures miss less than the original ten ({{miss.N8.azh}} against {{miss.P10.azh}}), but even the best, {{miss.b2.name}}, misses {{miss.b2.azh}} ({{miss.b2.azh.ci}}), and one architecture's AZH miss rate varied by up to {{miss.seedvar.max}} percentage points between its seeds. AZH images are small wound crops pasted on a black canvas ({{pad.black.azh}} of pixels black on average, against {{pad.black.fus}} for FUSeg). The share of the {{miss.nmodels}} pretrained models that miss an image correlates with its black-pixel share (Spearman {{pad.rho.black}}) more than with its lesion fraction ({{pad.rho.frac}}). Within each lesion-fraction tercile, images with more padding than the median are missed by a larger share of models ({{pad.t0.more}} against {{pad.t0.less}}, {{pad.t1.more}} against {{pad.t1.less}} and {{pad.t2.more}} against {{pad.t2.less}}). In {{pad.maj}} AZH images most pretrained models miss the wound and nnU-Net detects it; the reverse occurs in {{pad.rev}}.");
cap("Table", 5, "Modern models: ranking reliability across training seeds, model selection and misses. Agreement is mean Spearman correlation over the four wound datasets (Dice); intervals are 95% hierarchical bootstrap intervals for agreement, unit-bootstrap intervals for miss rates. Losses are mean Dice below the best candidate on the deployment dataset");
T(["Quantity", "Estimate", "95% interval or detail"],
  [
    { section: "Ranking reliability ({{n.C16}} architectures with three seeds)" },
    ["Same dataset, different seed", "{{sd.rbar}}", "{{sd.rbar.h}}"],
    ["Different dataset, different seed", "{{sd.cross}}", "{{sd.cross.h}}"],
    ["Gap (same dataset minus different dataset)", "{{sd.gap}}", "{{sd.gap.h}}"],
    ["Gap, {{n.P9}} original architectures", "{{sd.gap.P9}}", "{{sd.gap.P9.m}} (architecture bootstrap)"],
    ["Gap, {{n.N7}} new architectures", "{{sd.gap.N7}}", "{{sd.gap.N7.m}} (architecture bootstrap)"],
    ["Different dataset, same seed (mean over the three seeds)", "{{sd.single}}", "{{sd.single.h}}"],
    ["Different dataset, seed-averaged models", "{{sd.avg}}", "{{sd.avg.h}}"],
    { section: "Model selection ({{n.M19}} candidates; Dice lost against the best)" },
    ["Chosen on another wound dataset (mean of 12 ordered pairs)", "{{sel.off}}", "range {{sel.off.min}} to {{sel.off.max}}"],
    ["Random choice (expected)", "{{sel.rand}}", ""],
    ["Synthetic pick, seed averages", "{{sel.syn}}", "{{sel.syn.pick}}"],
    ["Synthetic pick, seed-0 models", "{{sel.s0.loss}}", "{{sel.s0.pick}}"],
    ["Synthetic pick, seed-1 models", "{{sel.s1.loss}}", "{{sel.s1.pick}}"],
    ["Synthetic pick, seed-2 models", "{{sel.s2.loss}}", "{{sel.s2.pick}}"],
    { section: "Misses on AZH (non-empty reference, Dice below 0.1)" },
    ["{{n.pre}} pretrained architectures", "{{miss.P18.azh}}", "{{miss.P18.azh.ci}}"],
    ["{{n.P10}} original architectures", "{{miss.P10.azh}}", "{{miss.P10.azh.ci}}"],
    ["{{n.N8}} new architectures", "{{miss.N8.azh}}", "{{miss.N8.azh.ci}}"],
    ["Best architecture, {{miss.b2.name}}", "{{miss.b2.azh}}", "{{miss.b2.azh.ci}}"],
    ["nnU-Net", "{{miss.nn.azh}}", "{{miss.nn.azh.ci}}"],
  ],
  [4900, 1300, 3160], { leftCols: 1, size: 17 });

H2("Robustness of the findings");
P("The pooled-set findings survive the analysis choices (Online Resource 1, Table S6). In all {{rp.UM19.ncond}} conditions (three post-processing pipelines under Dice and NSD, four further NSD tolerances, and FUSeg at 256 pixels under Dice and NSD), U+M19 wound-pair agreement is {{rp.UM19.pooled}} and exceeds the higher within-group value by {{rp.UM19.gap}}, each time with an interval above zero; agreement between wound and other datasets is {{rp.UM19.vsother}}; and stratified agreement ({{rp.UM19.strat}}) equals the size-weighted within-group average in every condition. The weaker U+F effect does not reach this standard in any condition (excess over the higher within-group value {{rp.UF.gap}}). The modality difference of every pooled set stays above zero in all conditions (U+M19 {{rp.UM19.mdiff}}).", { after: 100 });
P("Among modern models, wound-pair agreement stays within {{rob.M19.dice.range}} under Dice and {{rob.M19.nsd.range}} under NSD across post-processing pipelines, tolerances, the resampled FUSeg and the oracle threshold, and none of these changes has an interval that excludes zero; the common threshold described above is the only exception. Post-processing barely reorders the modern models under Dice (lowest rank correlation between pipelines on a non-floor dataset {{rob.pp.minrho}}), more under NSD ({{rob.pp.minrho.nsd}}). Excluding empty reference masks leaves the modality differences unchanged (U+M19 {{ne.UM19.mdiff}}, M19 {{ne.M19.mdiff}}) and wound-pair agreement nearly so (U+M19 {{ne.UM19.wound}}, M19 {{ne.M19.wound}}).");

// ------------------------------------------------------------------ discussion
H1("Discussion");
H2("What a validity figure measures");
P("A rank correlation between a synthetic benchmark and clinical data is usually read as a property of the benchmark. In our data the same benchmark scores {{b.UM19.mean.bench}} or {{b.M19.mean.bench}} depending on whether classical methods are in the pool, and {{b.U.co2.bench}} or {{b.U.azh.bench}} among untrained methods depending on which wound dataset is the reference. The high pooled value comes from the gap between groups. A benchmark on which modern models beat classical ones, as they do on every wound dataset, will correlate with any of those datasets over a pooled set, whether or not it can order the modern models among themselves. The stratification identity makes this exact: once the difference in level between groups is removed, what remains is the average of the within-group agreements, and among modern models that average is low.");
P("Ordering modern models among themselves is the question a vendor, a hospital or a regulator usually faces, and its ceiling is lower than it seems. Wound datasets agree with each other at {{a.dice.M19.wound}} among modern models, so a benchmark that agrees with them at {{b.M19.mean.bench}} is as good a proxy as another clinical dataset and no better. A validity figure is informative only next to the agreement between clinical datasets for the same methods.");
H2("Modern models: dataset differences, training noise and thresholds");
P("Three things limit agreement among modern models. First, the models are close: the median gap between neighbouring models is below a hundredth of Dice on each wound dataset, much less than retraining moves a model, as reported for other benchmarks [@bouthillier2021]. Two seeds of the same architectures rank them only moderately alike on one dataset. Second, the datasets do differ: for the original architectures, two wound datasets disagree more than two seeds do, by a margin whose interval excludes zero. That part of the disagreement is a property of the data rather than of training. It did not appear among the new architectures, and the criterion fixed in advance for calling the disagreement data-driven was not met, so the evidence supports only this narrower statement. Third, the threshold protocol matters. A common threshold raises agreement, but the raised ranking follows how confident each model's outputs are [@guo2017], a trait that carries across datasets, rather than which model segments wounds best; an oracle threshold that removes calibration differences lowers agreement. Our main protocol tunes each model's threshold on held-out images from its training domain, as is usual before deployment, and the low agreement applies to that protocol.");
P("For practice, a single dataset, a single training seed and a single threshold protocol can each change which modern model looks best. In our data the best model differed between wound datasets and changed with the seed, the synthetic benchmark's choice changed with the seed, and a model chosen on one wound dataset did only modestly better on another than a random choice.");
H2("Relation to prior work");
P("The composition effect is Simpson's paradox [@simpson1951;@kievit2013], and its role in benchmark agreement is established for language models [@perlitz2024;@zhang2026] and for robustness of image classifiers, where models trained on different data lie on different lines relating in-distribution and out-of-distribution accuracy [@miller2021;@fang2022;@teney2023]. Our contribution is to measure its size in medical segmentation, to show that it inflates agreement within one modality as well as creating disagreement across modalities, to give the exact identity behind stratification, and to show the consequence for validating synthetic benchmarks. That training seeds and small test sets make model comparisons uncertain has been argued for machine learning benchmarks and for medical imaging [@bouthillier2021;@varoquaux2022]; here the seed variation is large enough to change the validity figure of a benchmark and the model it selects. The task dependence of synthetic-to-real transfer reported for CT [@tushar2026b] and the strong transfer reported for programmable lesion trials [@tushar2026a] are consistent with our finding that validity is not a fixed property of a synthetic set. Reporting intervals over methods follows recommendations to quantify uncertainty in medical imaging evaluation [@wiesenfarth2021;@christodoulou2024]. Finally, the advice to maximise variation in synthetic data comes from training, where domain randomisation helps models transfer to real images [@tobin2017;@tremblay2018;@prakash2019]. A benchmark has a different job: it should order methods as the deployment data would, which argues for matching the deployment distribution rather than exceeding it.");
H2("Why wound photography differs");
P("Colonoscopy datasets from three centres agree strongly among the untrained and Kvasir-trained methods, while wound datasets do not. Colonoscopy frames share a device class and a field of view. Wound photographs have no standard framing: the median lesion fraction of the four wound datasets ranges from {{frac.wound.range}}, two are different crops of one clinic's photographs, and one was taken on smartphones in a leprosy programme. The AZH crops are mostly black padding, and that framing is associated with whether the pretrained models detect the wound at all. Dice depends on lesion size [@raina2023], and under NSD mean agreement between wound datasets among untrained methods rises from {{a.dice.U.wound}} to {{a.nsd.U.wound}}, which is consistent with framing conventions contributing to the disagreement [@torralba2011;@kendrick2022].");
H2("Recommendations");
list([
  "Report a benchmark's agreement within the group of methods it is meant to compare, and name that group.",
  "Report agreement against several clinical datasets, as a matrix, together with the agreement between those clinical datasets and a within-dataset control.",
  "Report intervals that resample methods and the images or photographs of each dataset.",
  "For trained models, train several seeds and report how well the ranking on one dataset is reproduced across seeds before interpreting differences between datasets.",
  "State the threshold protocol, because a common threshold can create agreement that reflects calibration rather than segmentation quality.",
  "Report failure rates, such as the share of missed lesions, next to mean Dice.",
  "If a mixed pool cannot be avoided, report stratified agreement, keeping in mind that it equals the size-weighted average of the within-group agreements.",
]);
H2("Limitations");
list([
  "The modern sets are small ({{n.M19}} methods; {{n.C16}}, {{n.P9}} and {{n.N7}} architectures in the seed analyses), so Spearman correlations move in coarse steps and intervals are wide. Several statements about modern models rest on intervals that include zero.",
  "In the second session {{n.manet.fail}} of the {{n.manet.run2}} MA-Net trainings diverged under mixed precision and were not imputed. MA-Net on ResNet-18 is averaged over two seeds and MA-Net on ResNet-50 is a single model, so these columns carry more training noise than the others, although they are the best seed-averaged models on Medetec and FUSeg.",
  "Seed 0 of the ten original architectures was trained in a different GPU session from seeds 1 and 2, so for these architectures the effects of the session and of the seed cannot be separated. The first session did not record the PyTorch and timm versions.",
  "nnU-Net was trained once, in its 2D configuration, for 100 epochs rather than the default 1,000, with its threshold fixed at 0.5. Its scores carry the noise of a single training, which matters because it was the synthetic benchmark's pick in most settings. The pretrained architectures were trained on {{n.train}} images at 256 pixels; they represent current architectures, not every deployed system.",
  "The level of agreement among modern models depends on the threshold protocol. Our statements about low agreement apply to each model at its own tuned threshold; a common threshold gives higher agreement, and the oracle threshold, chosen on the evaluation images and limited by the grid, gives lower agreement.",
  "The modern models score near zero Dice on all seven non-wound datasets, so their colonoscopy agreement and modality difference rank floor-level scores, and the from-scratch group is near the floor on most of these datasets too.",
  "The synthetic benchmarks are parametric, contain 45 images each and were built at three synthetic seeds. Learned generators such as diffusion models were not tested. Benchmark intervals for sets that include untrained or from-scratch methods resample methods only.",
  "A fifth wound dataset was excluded before scoring because many of its masks did not match their images. Wound photography is therefore represented by four datasets from three sources, two of which share one clinic, and statements about wound datasets rest on five pairs from different sources.",
  "Medetec's {{n.img.med}} images are {{n.unit.med}} photographs with four variants each, so its effective sample is smaller than its image count; the AZH sets also contain several crops per photograph.",
  "Images were not harmonised to one resolution across datasets, and NSD uses a fixed pixel tolerance, which is looser at lower resolution. Resampling FUSeg and varying the tolerance did not change the conclusions, but other datasets were not resampled. Untrained methods and probability-output methods receive different post-processing, which coincides with the main contrast between groups.",
  "The seed, benchmark, selection and robustness analyses are exploratory and are not corrected for multiple comparisons. The selection losses are point estimates, and the link between padding and misses is correlational.",
  "SAM and MedSAM received box prompts from the reference masks, an oracle that makes their scores incomparable with automatic segmentation; they were therefore kept out of every method set.",
], false);

H1("Conclusions");
P("Agreement between clinical datasets about which segmentation method is better depends strongly on which methods are compared. Pooling classical and modern methods inflates agreement between wound datasets and reverses it across modalities, and a synthetic benchmark validated over such a pool inherits the inflation. Among the modern models that practitioners choose between, wound datasets agree weakly, training noise is of the same order as the differences between models, and the threshold protocol shifts the level of agreement. A synthetic benchmark can then be validated no better than one clinical dataset predicts another. Validity should be reported within the group of methods to be compared, against several clinical datasets, with intervals over methods and images, and with the training seeds and threshold protocol stated.");

// ------------------------------------------------------------------ declarations
part = "back";
H1("Statements and Declarations");
P("**Ethics approval.** Not required. The study used only publicly available, de-identified datasets distributed for research use.", { after: 80 });
P("**Consent to participate and to publish.** Not applicable.", { after: 80 });
P("**Data availability.** All clinical datasets are publicly available from their original distributors (Table 1). Online Resource 2 gives the per-method mean Dice and NSD on every image set and the agreement for every pair of image sets in every method set. Online Resource 1 contains Tables S1 to S9 and Fig. S1.", { after: 80 });

// ------------------------------------------------------------------ references (JIIM pattern)
function jiim(s) {
  const m = s.match(/^(.*?), (\d{4})\. (.*)$/);
  if (!m) throw new Error("cannot parse reference: " + s);
  const [, auth, year, rest0] = m;
  const um = rest0.match(/^(.*?)\s(https?:\/\/\S+)$/);
  let rest = um ? um[1] : rest0;
  const url = um ? um[2] : "";
  const toks = auth.split(", ");
  const names = [];
  for (let i = 0; i < toks.length;) {
    const t = toks[i];
    if (t === "et al.") { names.push("et al."); i++; continue; }
    const nx = toks[i + 1];
    if (nx && /^(?:[A-ZÀ-Ý][a-z]?\.-?)+$/.test(nx)) {
      names.push(`${t} ${nx.replace(/\./g, "")}`);
      i += 2;
    } else { names.push(t); i++; }
  }
  let a = names.join(", ").replace(/, et al\.$/, " et al.");
  const acc = rest.match(/^(.*)\. Web resource, accessed ([^.]+)\.$/);
  if (acc) return `${a}: ${acc[1]}, ${year}. Available at ${url}. Accessed ${acc[2]}`;
  rest = rest.replace(/\.$/, "");
  return `${a}: ${rest}, ${year}.` + (url ? ` ${url}` : "");
}
add(new Paragraph({ children: [new PageBreak()] }));
H1("References");
const refText = order.map((k, i) => `${i + 1}. ${jiim(REFS[k])}`);
refText.forEach(t => { if (t.includes("—")) throw new Error("em dash in reference " + t); });
refText.forEach(t => add(ref(t)));

// ------------------------------------------------------------------ write
const doc = new Document({
  creator: "", title: TITLE,
  styles: { default: { document: { run: { font: L.FONT, size: 22, color: L.INK } } } },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 },
    margin: { top: convertInchesToTwip(1), bottom: convertInchesToTwip(1), left: convertInchesToTwip(1), right: convertInchesToTwip(1) } } },
    children: B }]
});
Packer.toBuffer(doc).then(b => {
  fs.writeFileSync("manuscript_v6_blinded.docx", b);
  const unusedRefs = Object.keys(REFS).filter(k => !order.includes(k));
  const unusedFacts = Object.keys(FACTS).filter(k => !used.has(k));
  if (process.argv.includes("--refs")) refText.forEach(t => console.log(t));
  console.log(`wrote manuscript_v6_blinded.docx`);
  console.log(`words: abstract ${words.abstract}, main text (Introduction to Conclusions) ${words.main}`);
  console.log(`references cited ${order.length}; not cited (left out of the list) ${unusedRefs.length}: ${unusedRefs.join(", ")}`);
  console.log(`facts used ${used.size} of ${Object.keys(FACTS).length}; unused ${unusedFacts.length}: ${unusedFacts.join(", ")}`);
});
