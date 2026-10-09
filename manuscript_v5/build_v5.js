// Manuscript V5: JIIM structure, numbered references, every number from facts.json.
const fs = require("fs");
const L = require("./lib");
const { d, p, h1, h2, table, caption, figure, rule, runs } = L;
const { Document, Packer, Paragraph, TextRun, AlignmentType, PageBreak,
        ExternalHyperlink, convertInchesToTwip, PageNumber, Footer } = d;

const BLIND = process.argv.includes("--blind");
const F = JSON.parse(fs.readFileSync("facts.json", "utf8"));
const REFS = Object.fromEntries(fs.readFileSync("refs_v5.tsv", "utf8").split("\n")
  .filter(Boolean).map(l => { const i = l.indexOf("\t"); return [l.slice(0, i), l.slice(i + 1)]; }));
const order = [];

// {key} -> "v (95% CI a to b)"; {key|v}; {key|ci}; plain string facts as is.
function fact(spec) {
  const [k, part] = spec.split("|");
  const x = F[k];
  if (x === undefined) throw new Error("missing fact " + k);
  if (typeof x !== "object") return String(x);
  return part ? x[part] : x.vci;
}
// [@a; @b] -> [3, 7]
function cite(spec) {
  const keys = spec.split(";").map(s => s.trim().replace(/^@/, ""));
  const nums = keys.map(k => {
    if (!REFS[k]) throw new Error("missing ref " + k);
    if (BLIND && k === "islam2026") return null;
    if (!order.includes(k)) order.push(k);
    return order.indexOf(k) + 1;
  }).filter(n => n !== null).sort((a, b) => a - b);
  if (!nums.length) return "[blinded]";
  const parts = []; let i = 0;
  while (i < nums.length) {
    let j = i;
    while (j + 1 < nums.length && nums[j + 1] === nums[j] + 1) j++;
    parts.push(j - i >= 2 ? `${nums[i]}–${nums[j]}` : nums.slice(i, j + 1).join(", "));
    i = j + 1;
  }
  return `[${parts.join(", ")}]`;
}
const T = s => s.replace(/\{([^}]+)\}/g, (_, k) => fact(k)).replace(/\[(@[^\]]+)\]/g, (_, k) => cite(k));
const B = [];
const add = (...xs) => xs.forEach(x => B.push(x));
const P = (t, o) => add(p(T(t), o));
const TT = rows => rows.map(r => r.map(c => T(String(c))));

// ------------------------------------------------------------------ title
add(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 160, line: 300 },
  children: [new TextRun({ text: "Mixing method types inflates rank agreement between clinical datasets and synthetic benchmarks in medical image segmentation",
    font: L.FONT, size: 30, bold: true, color: L.INK })] }));
add(rule(240));

// --------------------------------------------------------------- abstract
add(h2("Abstract"));
P("**Purpose.** A synthetic test set for segmentation is usually trusted because it ranks methods the way a clinical dataset does. We asked how much that agreement depends on which methods are ranked.");
P("**Methods.** We scored 57 methods in four groups, 18 classical algorithms, 14 networks trained from scratch on wound photographs, 11 ImageNet-pretrained networks and nnU-Net trained on the same photographs, and 14 networks trained from scratch on colonoscopy images, on 11 public clinical datasets in four modalities and on parametric synthetic wound images. Agreement is the Spearman correlation of per-method mean Dice, with 95% intervals from resampling methods.");
P("**Results.** Within every group, colonoscopy datasets from different centres agreed strongly, while wound datasets did not agree more with each other than with other modalities. Mixing groups manufactured structure. Among the 11 modern networks, wound datasets agreed at {dice.M.wound_x_wound|v} ({dice.M.wound_x_wound|ci}); adding the 18 classical methods raised this to {dice.U+M.wound_x_wound|v} ({dice.U+M.wound_x_wound|ci}) and made wound datasets anti-correlated with every other modality ({dice.U+M.dom:in_out|v}). The synthetic benchmark's agreement with the wound datasets rose from {dice.M.bench_wound|v} ({dice.M.bench_wound|ci}) to {dice.U+M.bench_wound|v} ({dice.U+M.bench_wound|ci}) in the same way.");
P("**Conclusion.** Rank agreement computed over a pool that mixes classical and modern methods mainly measures the gap between the two groups. A synthetic benchmark validated that way says little about which modern model to deploy. Agreement should be reported within groups of comparable methods, with intervals.");
P("**Keywords.** segmentation evaluation; benchmark validity; synthetic data; rank agreement; wound imaging; Simpson's paradox", { after: 200 });

// ----------------------------------------------------------- introduction
add(h1("Introduction"));
P("Hospitals, vendors and regulators increasingly test segmentation software on images other than those it was developed on. Synthetic images are attractive for this because they are unlimited, labelled exactly and free of patient data, and regulatory science already accepts in-silico evidence for some imaging questions [@badano2018; @n_sizikova2023]. Before a synthetic test set is used to choose between methods, it has to be shown to rank methods the way clinical data would. The usual evidence is a rank correlation between method scores on the synthetic set and on one clinical dataset, sometimes compared with the correlation between two clinical datasets [@n_kang2022; @borsukiewicz2026; @n_hu2023].");
P("That evidence is only as good as the rank correlation it rests on, and rankings in medical image analysis are known to be fragile. Challenge rankings change with the test cases, metric, aggregation and annotators [@maierhein2018; @n_wiesenfarth2021], leaderboards change between hospitals [@bassi2024; @ali2024], and confidence intervals on reported performance are often missing or too narrow [@n_christodoulou2024]. Outside medicine, whether benchmarks agree depends strongly on which models are compared [@n_perlitz2024; @n_dehghani2021], and part of the disagreement between benchmarks comes from differences in how much each model was trained on each task [@zhang2026; @dominguezolmedo2025]. The correlation between in-distribution and out-of-distribution accuracy can even change sign with the set of models chosen [@teney2023; @n_fang2022].");
P("These observations point to a specific risk for segmentation. A typical method pool mixes classical algorithms with trained networks, and trained networks are fitted to one domain. When two groups of methods swap places between domains, a correlation computed over both groups can be large and of either sign even when there is no relationship within either group. This is Simpson's paradox [@n_simpson1951; @n_kievit2013], and it would make a validity figure measure the gap between method groups rather than the benchmark.");
P("We measured the size of this effect for segmentation, using wound photography as the case study. Wound photographs have no standard framing or acquisition protocol, several public datasets exist, and synthetic wound images have been proposed for evaluation [@n_hu2023; @sarp2021; @basiri2023]. We scored four groups of methods, including pretrained networks and nnU-Net [@isensee2021], on 11 clinical datasets in four modalities and on synthetic wound benchmarks from three generators. The study asks three questions: whether agreement between clinical datasets follows modality within groups of comparable methods; how much mixing method groups changes that agreement; and how much it changes the measured validity of a synthetic benchmark.");

// ------------------------------------------------------ materials and methods
add(h1("Materials and Methods"));
add(h2("Clinical datasets"));
P("Eleven public datasets with expert masks were used (Table 1). Wound photography: Medetec [@medetec2014], CO2Wounds-V2 [@sanchez2024], the FUSeg challenge validation set [@wang2024] and the AZH chronic wound test set [@wang2020]. FUSeg and AZH come from one clinic and are treated as one source; CO2Wounds-V2 was photographed with smartphones in a leprosy programme in Colombia. Colonoscopy: Kvasir-SEG [@jha2020] and its sessile subset, CVC-ClinicDB [@bernal2015], CVC-ColonDB [@bernal2012] and ETIS-LaribPolypDB [@silva2014]. The two Kvasir sets share no image, and the two CVC sets come from one group. Dermoscopy: PH2 [@mendonca2013]. Thyroid ultrasound: the TN3K test split [@gong2021].");
P("Up to 150 images were taken from each dataset, evenly spaced over the sorted file list so that video datasets cover many sequences. Masks were binarised at any non-zero value for wound sets and above 127 otherwise, and every dataset except FUSeg, Medetec and AZH was resampled to a long side of 256 pixels. {n_excl_fuseg} FUSeg images that are byte-identical to FUSeg training images or to each other were removed. A within-dataset control was built from AZH training images that no method saw; {n_excl_ctrl} of the first 150 were removed because image matching (SIFT features with RANSAC, 25 or more inliers) showed they are crops of FUSeg photographs.");
add(caption("Table", 1, "Clinical datasets. n is the number of images scored after exclusions; lesion fraction is the median share of image pixels inside the mask."));
add(table(["Dataset", "Modality", "Source", "n", "Empty masks", "Lesion fraction"], [
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
  ["TN3K (test)", "Thyroid ultrasound", "TN3K", "150", "0", "0.066"]
], [2300, 1750, 1900, 600, 1060, 1750], { leftCols: 3, size: 17 }));

add(h2("Methods and method groups"));
P("Fifty-nine methods were scored in five groups (Table 2). Classical methods have no learned parameters: thresholding on grey level, colour channels and corner-referenced colour distances, k-means and mean-shift clustering, watershed, GrabCut, flood fill, Chan-Vese and edge-based filling. Scratch-FUSeg methods are three pixel classifiers on colour and texture features, a three-level U-Net and ten architectures from segmentation_models_pytorch [@iakubovskii2019] (U-Net, U-Net++, DeepLabV3+, FPN, PSPNet, LinkNet and MA-Net on ResNet-18, U-Net on MobileNetV2 and EfficientNet-B0, and SegFormer) [@ronneberger2015; @zhou2018; @chen2018; @lin2017; @zhao2017; @chaurasia2017; @fan2020; @he2016; @sandler2018; @tan2019; @xie2021], all randomly initialised and trained at 128 by 128 pixels. Modern methods are the same ten architectures with ImageNet encoders [@n_deng2009], fine-tuned at 256 pixels with the aspect ratio kept, plus nnU-Net v2 in its 2D configuration [@isensee2021]. Scratch-Kvasir methods repeat the Scratch-FUSeg recipe on Kvasir-SEG images outside both Kvasir test sets. SAM and MedSAM [@n_kirillov2023; @n_ma2024] were prompted with one box per ground-truth component; because the prompt uses the answer, they are reported only as a performance reference and are not part of any ranking.");
P("All FUSeg-trained methods used FUSeg training images 0 to 392, minus three that duplicate evaluation images; pixel classifiers used images 0 to 249. Decision thresholds were chosen on held-out training images (250 to 399 for the pixel classifiers, 400 to 549 for the networks); the three-level U-Net and nnU-Net used 0.5. The same slices of the Kvasir pool were used for Scratch-Kvasir. Classical methods are post-processed with a morphological closing and opening (5-pixel ellipse) and removal of components smaller than 0.05 percent of the image; trained methods have their probability map smoothed (Gaussian, sigma 2 pixels) before thresholding and the same component removal.");
add(caption("Table", 2, "Method groups. Held-out Dice is on FUSeg or Kvasir training images not used for fitting."));
add(table(["Group", "n", "Training data", "Initialisation", "Held-out Dice"], [
  ["Classical", "18", "none", "none", "not applicable"],
  ["Scratch-FUSeg", "14", "FUSeg training images", "random", "0.51 to 0.64 (networks)"],
  ["Modern", "11", "FUSeg training images", "ImageNet (10); nnU-Net (1)", "{pt_heldout} (pretrained)"],
  ["Scratch-Kvasir", "14", "Kvasir-SEG training images", "random", "0.66 to 0.81 (networks)"],
  ["SAM, MedSAM", "2", "none (box prompts from ground truth)", "public weights", "reported separately"]
].map(r => r.map(c => T(c))), [1800, 500, 2700, 2400, 1960], { leftCols: 4, size: 17 }));

add(h2("Synthetic benchmarks"));
P("Three parametric generators produce wound images with exact masks. Generator A places a lesion on a uniform drape and reproduces a published 45-image set [@islam2026]. Generator B renders textured skin with a luminance mottling field whose amplitude is a parameter, a calibrated tissue palette and clutter such as hair and gauze. Generator C shares no code with B and uses spectral noise, vignetting, distance-field blending and a CIELAB colour model. Each benchmark has 45 images in three wound classes. Generator B was built at mottling amplitudes 11, 28 and 40 with seeds 2024, 7 and 101, giving nine benchmarks; benchmark agreement is the mean over these nine unless stated.");

add(h2("Agreement and statistics"));
P("Agreement between two image sets is the Spearman correlation between their vectors of per-method mean Dice over the same methods. Pairs of clinical datasets are labelled within source (same clinic or group, no shared images), within modality (different sources) or across modality. Normalised surface Dice at 2 pixels [@nikolov2018] and Dice with empty-mask images excluded were computed for every result.");
P("Uncertainty comes mainly from which methods happen to be in a pool. Every interval is therefore a 95% percentile interval from 2,000 resamples of methods with replacement, stratified by group. Whether same-modality pairs agree more than cross-modality pairs was tested exactly over all {n_relabel} assignments of the observed modality labels to the 11 datasets (smallest attainable p {p_min}); whether the four wound datasets cluster was tested over all {n_wound_relabel} ways of choosing four of the 11. The reliability of each dataset's ranking was estimated by split-half resampling of its images with the Spearman-Brown correction. For mixed pools we also report a stratified agreement, in which methods are ranked within their own group before the correlation is pooled. The pre-specified questions are the three listed in the Introduction; the texture analysis of generator B and the matched-benchmark comparison are exploratory.");

// ----------------------------------------------------------------- results
add(h1("Results"));
add(h2("Method groups differ most between domains"));
P("Each FUSeg-trained group led on wound photographs and fell below the classical methods on every other modality (Fig. 1). Modern methods reached mean Dice {md.M.fuseg} on FUSeg and {md.M.medetec} on Medetec, but {mdr.M.colon} on colonoscopy. Scratch-Kvasir methods reversed the pattern on colonoscopy ({md.K.kvasir_disjoint} on Kvasir) and stayed close to the classical methods on wound photographs ({mdr.K.wound} against {mdr.U.wound}). Box-prompted SAM and MedSAM averaged {mdr.S.all} across datasets. Each dataset ranked methods reliably within every group: split-half reliability for the modern group was {rel.M.all} across datasets and {rel.M.wound} on the wound datasets.");
add(figure("fig1_group_dice.png", 560, 330));
add(caption("Figure", 1, "Mean Dice of each method group on each clinical dataset. The shaded area is wound photography, the training domain of the FUSeg-trained groups."));

add(h2("Clinical datasets agree within colonoscopy, not within wound photography"));
P("Within every group of comparable methods, same-modality pairs of datasets agreed more than cross-modality pairs (Table 3). The difference was {dice.U.modality_diff} among classical methods, {dice.F.modality_diff} among Scratch-FUSeg, {dice.M.modality_diff} among modern methods and {dice.K.modality_diff} among Scratch-Kvasir methods ({dice.U.p}, {dice.F.p}, {dice.M.p} and {dice.K.p} by the exact test).");
P("This structure comes from colonoscopy. Colonoscopy datasets agreed with each other at {colon_range} on average in every group. The four wound datasets did not cluster in any group of comparable methods ({dice.U.pw}, {dice.F.pw}, {dice.M.pw} and {dice.K.pw} for classical, Scratch-FUSeg, modern and Scratch-Kvasir), and among modern methods they agreed at {dice.M.wound_x_wound}, below the agreement of each dataset with itself on split halves. Wound datasets therefore rank comparable methods differently, and this is not explained by too few images per dataset.");
add(caption("Table", 3, "Agreement between clinical datasets by level, per method group, under Dice. Intervals are 95% method-bootstrap intervals."));
const LV = ["level:same source", "level:different source, same modality", "level:across modality"];
add(table(["Group", "n", "Control", "Within source", "Within modality", "Across modality", "Same minus across modality", "Exact p"],
  TT(["U", "F", "M", "K"].map(g => [
    { U: "Classical", F: "Scratch-FUSeg", M: "Modern", K: "Scratch-Kvasir" }[g], `{n.${g}}`, `{dice.${g}.control|v}`,
    ...LV.map(l => `{dice.${g}.${l}|v}`), `{dice.${g}.modality_diff|v} ({dice.${g}.modality_diff|ci})`, `{dice.${g}.p}`])),
  [1500, 450, 900, 1000, 1100, 1100, 2150, 1160], { leftCols: 1, size: 17 }));

add(h2("Mixing method groups manufactures agreement and disagreement"));
P("Pooling groups changed every one of these results (Fig. 2, Table 4). When the 18 classical methods were added to the 11 modern ones, agreement between wound datasets rose from {dice.M.wound_x_wound} to {dice.U+M.wound_x_wound}, and wound datasets became anti-correlated with the other modalities ({dice.U+M.dom:in_out}), although neither group shows that relation on its own ({dice.U.dom:in_out|v} and {dice.M.dom:in_out|v}). Mixing classical with Scratch-FUSeg methods produced the same pattern at a smaller size ({dice.U+F.dom:in_out}). In each case the pooled value lies outside the range of the two within-group values, which is the signature of a composition effect [@n_simpson1951]: the modern methods lead on wound photographs and trail on every other modality (Fig. 1), so any wound dataset ranks modern methods first and any other dataset ranks them last.");
P("The mirror test with colonoscopy-trained methods behaved as this mechanism predicts. Scratch-Kvasir methods led on colonoscopy but stayed close to classical methods on wound photographs, so the two groups only partly swapped places. Agreement between colonoscopy and other datasets was {mirror.U} among classical and {mirror.K} among Scratch-Kvasir methods, and {mirror.UK} when pooled: below both, but not negative. Ranking methods within their own group before pooling removed most of the manufactured structure: the stratified agreement between wound datasets in the classical plus modern pool was {dice.U+M.strat:wound_x_wound|v}, against {dice.U+M.wound_x_wound|v} for the ordinary correlation.");
add(figure("fig2_matrices.png", 620, 330));
add(caption("Figure", 2, "Agreement between clinical datasets under Dice. (a) The 11 modern methods. (b) The same 11 plus the 18 classical methods. Lines separate wound photography, colonoscopy, dermoscopy and ultrasound."));
add(caption("Table", 4, "Agreement by type of pair, per pool, under Dice. The training domain is wound photography for FUSeg-trained groups and colonoscopy for Scratch-Kvasir; for the classical group it is set to wound photography."));
add(table(["Pool", "n", "Within training domain", "Training domain with other modalities", "Other modalities with each other", "Stratified, domain with others"],
  TT([["Classical", "U"], ["Modern", "M"], ["Classical + Modern", "U+M"], ["Scratch-FUSeg", "F"], ["Classical + Scratch-FUSeg", "U+F"], ["Scratch-Kvasir", "K"], ["Classical + Scratch-Kvasir", "U+K"]].map(([nm, g]) => [
    nm, `{n.${g}}`, `{dice.${g}.dom:in_in|v} ({dice.${g}.dom:in_in|ci})`, `{dice.${g}.dom:in_out|v} ({dice.${g}.dom:in_out|ci})`,
    `{dice.${g}.dom:out_across|v}`, g.includes("+") ? `{dice.${g}.strat:in_out|v}` : "not applicable"])),
  [2200, 450, 1750, 1900, 1400, 1660], { leftCols: 1, size: 17 }));

add(h2("The measured validity of a synthetic benchmark follows the pool"));
P("Generator B's agreement with the four wound datasets followed the same pattern (Table 5, Fig. 3). Among modern methods alone it was {dice.M.bench_wound}, and among classical methods alone {dice.U.bench_wound}. Over the mixed pool it rose to {dice.U+M.bench_wound}, a figure that would usually be read as strong validity. That figure reflects the modern methods scoring higher on synthetic and on clinical wound images alike: mean Dice on generator B at amplitude 40 was {md.M.genB40} for the modern group and {md.U.genB40} for the classical group. Within the stratified analysis the mixed-pool agreement fell to {dice.U+M.strat:bench_wound|v}.");
P("Which wound dataset the benchmark is compared with mattered as much as the pool. Among classical methods, agreement ranged from {dice.U.bench:azh|v} with AZH to {dice.U.bench:co2wounds|v} with CO2Wounds-V2; among modern methods from {dice.M.bench:medetec|v} with Medetec to {dice.M.bench:co2wounds|v} with CO2Wounds-V2.");
add(caption("Table", 5, "Agreement of generator B with each wound dataset, per pool, under Dice: mean over nine benchmarks (three texture amplitudes by three seeds). Wound with wound is the mean agreement between the four clinical wound datasets over the same methods."));
add(table(["Pool", "n", "Medetec", "CO2Wounds-V2", "FUSeg", "AZH", "Benchmark, mean", "Wound with wound"],
  TT([["Classical", "U"], ["Scratch-FUSeg", "F"], ["Modern", "M"], ["Scratch-Kvasir", "K"], ["Classical + Scratch-FUSeg", "U+F"], ["Classical + Modern", "U+M"]].map(([nm, g]) => [
    nm, `{n.${g}}`, `{dice.${g}.bench:medetec|v}`, `{dice.${g}.bench:co2wounds|v}`, `{dice.${g}.bench:fuseg|v}`, `{dice.${g}.bench:azh|v}`,
    `{dice.${g}.bench_wound|v} ({dice.${g}.bench_wound|ci})`, `{dice.${g}.wound_x_wound|v} ({dice.${g}.wound_x_wound|ci})`])),
  [2050, 400, 850, 1150, 750, 750, 1700, 1700], { leftCols: 1, size: 17 }));
add(figure("fig3_forest.png", 620, 221));
add(caption("Figure", 3, "Agreement under Dice with 95% method-bootstrap intervals for each pool: (a) between wound datasets, (b) between the training-domain datasets and other modalities, (c) between generator B and the wound datasets. Filled points are mixed pools."));

add(h2("Texture and matched benchmarks (exploratory)"));
P("Over the mixed classical plus Scratch-FUSeg pool, generator B's agreement with the wound datasets rose with texture amplitude ({dice.U+F.bench_amp11|v}, {dice.U+F.bench_amp28|v} and {dice.U+F.bench_amp40|v} at amplitudes 11, 28 and 40). Within groups the trend depended on the dataset and the group: among classical methods {dice.U.bench_amp11|v}, {dice.U.bench_amp28|v} and {dice.U.bench_amp40|v}, and among modern methods {dice.M.bench_amp11|v}, {dice.M.bench_amp28|v} and {dice.M.bench_amp40|v}. Four benchmarks from three generators spanning measured background texture from about 2.1 to 3.3 gave similar agreement within each group once averaged over three seeds (Table 6), so the measured texture statistic did not predict validity in this range. In the mixed classical plus modern pool the same four benchmarks ranged from {matched_mixed_range}, because the modern methods scored near zero on the uniform-drape images of generator A and well on the others; the spread between benchmarks in a mixed pool reflects how each benchmark treats the two groups.");
const MT = F.matched;
add(caption("Table", 6, "Four benchmarks from three generators, mean (standard deviation) over seeds 2024, 7 and 101 of the agreement with the four wound datasets under Dice. Texture is the median local standard deviation of shading-corrected background luminance."));
add(table(["Benchmark", "Texture", "Classical", "Modern", "Classical + Modern", "Classical + Scratch-FUSeg"],
  [["nrA", "Generator A, uniform drape"], ["nrC0", "Generator C, no noise"], ["nrC34", "Generator C, noise 3.4"], ["nrB0", "Generator B, texture removed"]].map(([k, nm]) => [
    nm, `${MT[k].texture} (${MT[k].texture_range})`.replace(/-/g, "−"), MT[k]["U"] || "", MT[k]["M"] || "", MT[k]["U+M"] || "", MT[k]["U+F"] || ""]),
  [2500, 1700, 1150, 1150, 1300, 1560], { leftCols: 1, size: 17 }));

add(h2("Robustness"));
P("Every result above holds under normalised surface Dice and with empty-mask images excluded. Under surface Dice the same-minus-across-modality difference was {nsd.U.modality_diff|v}, {nsd.F.modality_diff|v}, {nsd.M.modality_diff|v} and {nsd.K.modality_diff|v} for the four groups, wound datasets did not cluster within any group ({nsd.M.pw} among modern methods), and mixing classical with modern methods again raised wound-to-wound agreement, from {nsd.M.wound_x_wound|v} to {nsd.U+M.wound_x_wound|v}, and generator B's agreement, from {nsd.M.bench_wound|v} to {nsd.U+M.bench_wound|v}. With empty-mask images excluded the corresponding Dice values were {dice_noempty.M.wound_x_wound|v} and {dice_noempty.U+M.wound_x_wound|v}.");

// -------------------------------------------------------------- discussion
add(h1("Discussion"));
P("A validity figure for a synthetic benchmark is a rank correlation, and a rank correlation over a mixed pool of methods is dominated by the largest differences between methods. In segmentation those differences are between classical and trained methods, and trained methods are tied to the domain they were trained on. The result is that a benchmark can show agreement of {dice.U+M.bench_wound|v} with clinical wound data over a mixed pool while agreeing at only {dice.M.bench_wound|v} over the modern networks a user would actually choose between. The clinical datasets behave the same way: four wound datasets that appear to agree strongly over the mixed pool disagree among modern networks.");
P("This is the same phenomenon reported for language model benchmarks [@n_perlitz2024; @zhang2026] and for in-distribution versus out-of-distribution accuracy [@teney2023; @n_fang2022], and in statistical terms it is Simpson's paradox [@n_simpson1951; @n_kievit2013]. What is specific to segmentation is its size and direction. Mixing method types did not only inflate agreement within wound photography; it created anti-correlation between wound photography and every other modality, and positive agreement between modalities that share nothing. A study that reports one pooled correlation cannot tell these effects from real relationships between datasets.");
P("Two findings matter for practice beyond synthetic data. First, colonoscopy datasets from different centres ranked methods consistently in every group, so a colonoscopy dataset is a reasonable proxy for another. Wound datasets were not, even among strong modern networks, so choosing a wound segmentation model on one public dataset is unlikely to transfer to another clinic, consistent with earlier reports of variable wound rankings across datasets [@lucho2024]. Second, close work on synthetic test sets in CT reported strong transfer of rankings to clinical data [@n_tushar2026itrialspace] and found that transfer depends on the task [@n_tushar2026transfer]. Our results add that it also depends on the composition of the method pool, which should be reported alongside any such figure.");
P("Four practices follow and each is inexpensive. Report agreement within groups of comparable methods, or with the stratified agreement used here, rather than over a mixed pool. Give intervals from resampling methods, which here were often wider than the differences being claimed. Report agreement against several clinical datasets, because the choice of dataset alone moved the result substantially. And report the split-half reliability of each dataset's ranking, so that low agreement can be told apart from noise.");

add(h2("Limitations"));
[
  "Wound photography is represented by four datasets from three sources, two of which share a clinic, and the synthetic benchmarks are wound images only. The colonoscopy result rests on five datasets from three sources.",
  "The modern group has 11 networks, and its intervals are wide. Networks were fine-tuned on about 390 images under one protocol; larger training sets or other protocols may change the spread between them.",
  "SAM and MedSAM were prompted with boxes taken from the ground truth, so their scores are an upper reference and they were not ranked.",
  "Agreement was measured with Spearman correlation of mean Dice. Rankings based on other aggregations, such as per-image ranking or median scores, were not studied.",
  "The synthetic benchmarks contain 45 images each and come from parametric generators written by the same group; learned generators such as diffusion models were not tested.",
  "The texture and matched-benchmark analyses are exploratory, and the measured background texture of a benchmark varied by up to {tex_seed_spread} between seeds.",
  "The within-dataset control uses 103 AZH training images after exclusions, and image matching may miss some crops of FUSeg photographs."
].forEach(t => P(t, { indent: { left: 240 }, after: 90 }));

// ------------------------------------------------------------- conclusions
add(h1("Conclusions"));
P("Rank agreement between clinical datasets, and between a synthetic benchmark and clinical data, depends on which methods are ranked. Mixing classical and modern segmentation methods produced strong apparent agreement between wound datasets and strong apparent validity for a synthetic wound benchmark, both of which largely disappeared among modern networks alone. Validity figures should be reported within groups of comparable methods, against several clinical datasets and with intervals.");

// ---------------------------------------------------------- declarations
add(h1("Statements and Declarations"));
P("**Funding.** No funding was received for this study.");
P("**Competing interests.** The authors have no competing interests to declare.");
P("**Ethics approval.** This study used only publicly available, de-identified datasets released for research use. No new images were collected and no ethics approval was required.");
P("**Consent to participate and to publish.** Not applicable.");
P("**Data availability.** All clinical datasets are available from their original distributors. Per-image scores for every method and dataset, the exclusion lists and the synthetic benchmarks are provided as Online Resources.");
P("**Code availability.** The code for all methods, generators and analyses is provided as an Online Resource and will be released in a public repository on acceptance.");
P("**Use of AI tools.** An AI assistant was used for code development, analysis scripting and language editing under the authors' direction. The authors checked all code, results and text and take full responsibility for them.");

// -------------------------------------------------------------- references
add(new Paragraph({ children: [new PageBreak()] }));
add(h1("References"));
order.forEach((k, i) => {
  const e = REFS[k];
  const m = e.match(/(https?:\/\/\S+)$/);
  const head = m ? e.slice(0, m.index) : e;
  const ch = [new TextRun({ text: `${i + 1}. `, font: L.FONT, size: 19 }), ...runs(head, { size: 19 })];
  if (m) ch.push(new ExternalHyperlink({ link: m[1], children: [new TextRun({ text: m[1], font: L.FONT, size: 19, color: "28598A", underline: { type: "single", color: "28598A" } })] }));
  add(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 70, line: 250 }, indent: { left: 400, hanging: 400 }, children: ch }));
});

const doc = new Document({
  creator: "", title: "Mixing method types inflates rank agreement",
  styles: { default: { document: { run: { font: L.FONT, size: 22, color: L.INK } } } },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: convertInchesToTwip(1), bottom: convertInchesToTwip(1), left: convertInchesToTwip(1), right: convertInchesToTwip(1) } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: L.FONT, size: 18, color: "3F3F3F" })] })] }) },
    children: B
  }]
});
const out = BLIND ? "manuscript_v5_blinded.docx" : "manuscript_v5.docx";
Packer.toBuffer(doc).then(b => { fs.writeFileSync(out, b); console.log("wrote " + out + ", " + order.length + " references"); });
