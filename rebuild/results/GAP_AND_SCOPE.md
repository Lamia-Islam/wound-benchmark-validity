# Where the gap actually is, and what we can claim

Based on nine targeted searches against each of our claims. Every adjacent work
below is real and checked; nothing here is asserted from memory.

---

## 1. The surrounding literature, and what each one does NOT do

### Ranking instability is established. The cause is not localised.

**Maier-Hein et al., Nature Communications 2018** analysed 150 challenges and
showed an algorithm's rank is not robust to the test data used, the ranking
scheme, the annotators, or the aggregation method. **Metrics Reloaded**
(Nature Methods 2024) and its companion pitfalls paper address metric selection
and application. **MedSegBenchmarker** (arXiv:2608.29677, Aug 2026) shows that
reaggregating identical predictions changes the top-ranked architecture in three
of six dataset-resolution settings.

All of this says rankings are fragile and names sources. None of it asks whether
a *synthetic* benchmark preserves ranking, and none identifies a property of the
benchmark data you could measure to find out.

### Image-quality metrics already fail to predict downstream suitability.

**"Metrics that Matter"** (arXiv:2505.07175) is the closest work on metrics, and
after reading it in full it is closer than the abstract suggested. It evaluates
**no-reference image quality metrics** on brain MRI under controlled
perturbations (noise, distribution shift, localised morphological alteration)
and across generative architectures, and finds that widely-used no-reference
metrics "correlate poorly with downstream task suitability and exhibit a
profound insensitivity to localised anatomical details crucial for clinical
validity". It concludes that a multifaceted validation framework is needed and
declines to propose any single statistic as a design target.

**This partially pre-empts the shape of our negative result and we must say so.**
They show a single image-quality number does not predict whether generated data
is *good for* a downstream task. We show a single measured background statistic
does not predict whether a benchmark *orders methods* the way clinical data
does. Different quantity — suitability versus rank fidelity — and a different
construction, since ours is a statistic deliberately chosen as a design target
and matched across three independently built generator families rather than an
off-the-shelf quality score. But the honest framing is *consistent with and
extending to the ranking axis*, not *nobody suspected this*.

What remains ours: the ranking axis specifically, the matched-value construction
that isolates the residual, and the quantification that between-generator spread
at fixed statistic value exceeds the within-generator effect of the statistic.

### The rank-correlation question has been asked, in NLP.

**"Efficacy of Synthetic Data as a Benchmark"** (arXiv:2409.11968) asks whether
LLM-generated data can serve as a benchmark rather than as training data, across
six datasets and three tasks, and finds it works for simple tasks (intent
classification) and fails for complex ones (NER). It introduces a bias factor
for the case where the same model generates and is evaluated.

So the question is legitimate and has precedent. It has not been asked for
medical image segmentation, and that paper identifies task complexity as the
moderator, not a measurable property of the generated data.

### Region-size bias exists, but in the loss, not the benchmark.

**"Do We Really Need Dice?"** (arXiv:2104.08717, Medical Image Analysis 2023)
shows Dice and cross-entropy carry different label-marginal biases, with Dice
preferring small structures. This is about the **training loss**, and its
proposed fix is a regularisation term. It explicitly does not treat dataset
framing, cropping or region-size distribution as a preprocessing factor.

Our lesion-fraction result is at a different level: not which loss to train
with, but that the reference dataset's lesion fraction determines which
*benchmark design* looks realistic.

### Cropping is known to change results, one level below ours.

**A multi-dataset multi-seed preprocessing study** (Diagnostics 2026,
doi 10.3390/diagnostics16172880) compares five preprocessing pipelines including
ROI cropping across REFUGE, ORIGA and Drishti-GS with seeds 42, 15 and 89. Read
in full: it **does** report a reversal — full-resolution CLAHE is the best
configuration on REFUGE, while "ORIGA inverts that", with the ROI pipelines
holding a small advantage there. Its ROI crop is a 512x512 square around the
centroid of the ground-truth disc/cup bounding box, i.e. oracle localisation. It
also reports that multi-seed evaluation overturned single-run conclusions, which
is exactly what happened to us.

So the reversal claim is not new in general — but theirs is a reversal **between
datasets** (which pipeline wins depends on which dataset you use). Ours is a
reversal **within one dataset**: crop FUSeg to Medetec's lesion fraction and the
same clinic, same patients, same annotations flip their contrast preference from
+0.964 to -0.964. That is a strictly stronger form, and it is about the
benchmark-design verdict rather than about model performance. Their paper
supports our framing result and gives us a citable precedent for the seed
lesson; it does not pre-empt the within-dataset inversion.

### Model ranking without labels is an active area, on a different axis.

**Unsupervised source-free ranking of biomedical segmentation models under
distribution shift** (Springer, MICCAI-series) ranks pre-trained models on an
unlabelled target domain using prediction consistency under perturbations. That
is the competence-monitoring axis, and it is why the competence-signal section
of the original manuscript should stay cut: this is a live, better-resourced
line of work and we would be entering it from behind.

### Domain randomization points the opposite way, for a different purpose.

The sim-to-real literature (Tobin et al. 2017 onward, Structured Domain
Randomization 2018) established that for **training** transfer you maximise
variation: randomise textures, lighting and placement far beyond realism. Our
validation result is the opposite instruction, match rather than maximise, and
the contrast is worth stating explicitly because a reader's prior will be
domain randomization.

---

## 2. The gap, stated precisely

Four things nobody appears to have done.

**(a) Rank-preservation of synthetic benchmarks in medical image segmentation.**
Asked in NLP, asked for generative models, not asked here.

**(b) Tested whether a measurable image property can serve as a benchmark design
specification for rank fidelity, and reported that it fails.** Three benchmarks
matched on a measured texture statistic score 0.121, 0.434 and 0.189; the
between-generator spread at fixed statistic value exceeds the within-generator
effect of the statistic itself. arXiv:2505.07175 found the analogous failure for
image-quality metrics against downstream *suitability*, so the shape of this
result has precedent and must be cited as such. What is new is the ranking axis,
the matched-value construction across three independently built generator
families, and the quantified residual.

**(c) The clinical reference's framing convention determines the verdict,
within a single dataset.** Cropping FUSeg to Medetec's lesion fraction reverses
its contrast preference from +0.964 to -0.964. Same clinic, same patients, same
annotations. The preprocessing literature shows crops change performance and
that the winning pipeline differs *between* datasets (Diagnostics 2026); nobody
has shown a crop inverts *one* dataset's answer to a benchmark-design question.
This is now the cleanest genuinely-new result in the paper.

**(d) A per-channel multi-scale background descriptor used as a diagnostic.**
Ours orders three generator families correctly (-0.626, p = 0.020) where a
single statistic cannot (+0.244, p = 0.40), and it located a six-fold chroma
error in our own generator that every prior experiment missed. Fixing that error
moved the benchmark from 94% to 99% of the clinical-to-clinical ceiling and
sharpened the ablation from 2.1x to 4.1x.

---

## 3. What we can and cannot claim

**Claim, strongly:**

- Benchmark predictive validity is not reducible to a single measured image
  statistic. Measured, with the residual quantified, across three independently
  built generator families.
- A clinical dataset's apparent preference over synthetic design properties is
  an artifact of its framing convention, demonstrated by reversal.
- A per-channel multi-scale background descriptor is an effective diagnostic for
  synthetic benchmark defects, demonstrated by finding and fixing one.
- Within a fixed generator family, background texture dominates the other design
  properties tested, by 4.1x, cross-seed confirmed with non-overlapping ranges.
- For validation the objective is matching, not maximising, which is the
  opposite of the domain-randomization instruction for training.

**Claim, with care:**

- The clinical-to-clinical ceiling as the correct reference quantity for
  benchmark quality. Natural and useful, but two datasets is one pairwise
  measurement, and finding (c) shows that pair is not interchangeable with
  another pair at different framings.

**Do not claim:**

- A portable specification or target value. It failed the test we built for it.
- Anything about competence monitoring. Actively worked on elsewhere, and we
  would be behind.
- The rank inversion between synthetic benchmarks from the earlier version.
- Generality beyond chronic diabetic foot ulcer photography.

---

## 4. Scope and venue

The paper is a **validation-methodology paper with a negative core result**.
That is a real genre and it is the honest shape of what we have. It sits
directly downstream of Maier-Hein's line of work and should cite it as the
problem statement rather than compete with it.

Suggested framing: *ranking instability has a known list of causes; we tested
whether one of them, the benchmark data itself, can be controlled by a
measurable property, and it cannot. Here is the measurement, here is the
residual, here is what the property is good for instead, and here is a
second-order effect nobody has reported: the clinical reference's framing
decides the answer.*

**Venue.** JIIM remains reasonable and free via the subscription route. But the
contribution is now methodological rather than applied, so two better fits are
worth checking:

- **MELBA** (Machine Learning for Biomedical Imaging). Strong scope match for
  negative and methodological results. Rejected earlier in this project for
  lacking Scopus indexing; that trade should be re-examined now the paper is
  methodological, because MELBA readers are exactly the audience.
- **Medical Image Analysis** or **IEEE TMI**. Higher bar, longer review, but
  this is the literature those journals host. The negative result plus the
  framing finding would be a reasonable TMI submission if the clinical dataset
  count were higher.

The binding constraint on the higher venues is **two clinical datasets of one
pathology**. Everything else in the paper is now solid. Adding a third and
fourth clinical reference, ideally from a different modality, is the single
change that moves this from a solid JIIM paper to a credible MedIA or TMI one.

---

## 5. Caveats on this analysis

- Nine searches, not a systematic review. The four gap claims in Section 2 should
  each get a proper search before being asserted in a manuscript, particularly
  (a), which is the one most likely to have prior work I did not surface.
- I did not search non-English literature or conference workshop proceedings,
  where benchmark-design work often appears first.
- Both nearest neighbours have now been read in full and Sections 1 and 2 were
  revised accordingly. Net effect: claim (b) weakened from "nobody has reported
  this" to "the shape has precedent for a different quantity; the ranking axis
  and the matched-value construction are ours", and claim (c) strengthened from
  "nobody has shown a crop inverts a verdict" to the sharper within-dataset
  form, since Diagnostics 2026 already has the between-dataset version.
- (c) is therefore now the cleanest new result, not (b). The manuscript should
  lead with it.
