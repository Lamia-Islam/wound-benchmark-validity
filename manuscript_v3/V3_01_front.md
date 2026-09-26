# Manuscript V3, Part 1: front matter, introduction, related work

---

## Title

**Benchmark validity is not a property of the benchmark: rank agreement in
medical image segmentation depends jointly on the data, the reference, and the
method pool**

Short title: What benchmark agreement actually measures

---

## Abstract

**Background.** Algorithm rankings in medical image segmentation are known to be
fragile. Existing work attributes this to the test data, the ranking scheme, the
annotators and the aggregation method. It does not ask whether the fragility can
be localised to a measurable property of the benchmark, nor whether the method
pool being ranked is itself a source of it.

**Methods.** We scored 32 segmentation methods on five clinical reference sets
and on synthetic benchmarks from three independently constructed generator
families. The pool spans 21 classical algorithms, three pixel classifiers, and
eleven convolutional and transformer architectures trained from random
initialisation under one protocol. The clinical references are three chronic
wound photography datasets and two colonoscopy datasets. Agreement is Spearman
rank correlation of per-method Dice, with permutation tests at 20,000
permutations and bootstrap intervals at 5,000 resamples. Every reported
relationship is repeated under a boundary-sensitive metric, normalised surface
Dice at 2 px, and across four generator seeds.

**Results.** Rank agreement between clinical references falls in steps: +0.944
between disjoint halves of one dataset, +0.952 between two curations of one
source, +0.563 on average between datasets of the same modality, and no
detectable agreement across modality within a homogeneous method pool. In a
mixed pool the across-modality correlation appears strongly negative, -0.253,
but this is a composition artifact: it is +0.108 among classical methods alone
and +0.236 among deep methods alone, and arises because methods trained on one
modality lose their advantage on another. Benchmark validity is likewise
pool-dependent. The best synthetic configuration reaches 98% of the
clinical-to-clinical ceiling when validity is assessed over 22 mostly classical
methods, and 49% of it over 32 methods against references none of them were
trained on. Within one modality, a reference's verdict on synthetic design is a
monotonic function of the fraction of the frame its lesions occupy, holding
across four seeds and both metrics, with two held-out clinical datasets placed
to within 5% to 19% of the curve's range. Finally, benchmarks matched to within
0.03 on a measured background-texture statistic differ by 0.115 in validity,
so validity is not reducible to that statistic.

**Conclusions.** Benchmark agreement is a three-way quantity. A validity figure
quoted for a benchmark alone, without naming the clinical reference and the
method pool it was measured over, does not transfer. We give the size of each
dependence and the controls that make them interpretable.

**Keywords.** benchmark validity, rank correlation, synthetic data, evaluation
methodology, medical image segmentation

---

## 1. Introduction

Maier-Hein and colleagues analysed 150 biomedical image analysis challenges and
showed that an algorithm's rank is not robust: it moves with the test cases
used, with the ranking scheme, with the annotators, and with how per-case scores
are aggregated. Metrics Reloaded subsequently addressed metric choice, and more
recent work has shown that reaggregating identical predictions can change which
architecture ranks first.

That line of work establishes that rankings are fragile and names the sources.
It does not answer the next question, which is the practical one: given that a
ranking is fragile, what would you have to measure about your benchmark to know
how much to trust it?

This paper asks that question and reports that the answer has three parts rather
than one.

The first part is the clinical reference. We show that the agreement between two
clinical datasets about which method is better is not a fixed quantity. It falls
from +0.944 between two disjoint halves of one dataset, to +0.563 between
different datasets of the same modality, to nothing at all between modalities.
Since the within-dataset value establishes that the measurement itself is
stable, the decline is real disagreement between references and not sampling
noise. Any statement of the form "our benchmark agrees with clinical data at
rho = x" is therefore incomplete until it names which clinical data.

The second part is the method pool. This is the result we did not expect. The
same synthetic benchmark, scored against the same clinical references, reaches
98% of the clinical-to-clinical ceiling when validity is assessed over a pool of
21 classical algorithms and one small network, and 49% of it when eleven modern
architectures are added. A benchmark that looks nearly perfect for choosing
between thresholding methods is roughly half as good at ordering the methods a
practitioner would actually deploy. We also show that in a pool mixing trained
and untrained methods, cross-domain rank comparisons measure which methods had
domain-matched training rather than anything about the data, and that this can
invert the sign of the reported correlation.

The third part is the benchmark data itself, and here the result is negative in
a way we think is useful. We built a measurable background-texture statistic as
a candidate design specification, calibrated it against clinical images, and
tested it by constructing benchmarks from three independent generator families
matched on it. Benchmarks matched to within 0.03 on that statistic differ by
0.115 in validity. The between-generator spread at fixed statistic value is
comparable to the entire within-generator effect of the statistic. Validity is
not reducible to it.

Within a single modality, one property of the reference does predict the
direction of its bias. The fraction of the frame occupied by the lesion
determines whether a reference rewards high or low lesion-to-surround contrast,
monotonically, and the relationship holds across four generator seeds, under
both an overlap metric and a boundary metric, and over the enlarged method pool.
Two clinical datasets that took no part in constructing the relationship fall on
it. We report the relationship and explicitly decline to report its crossover
point as a portable constant, because that point moved when the method pool
changed.

### 1.1 Contributions

1. A measured hierarchy of rank agreement between clinical references, from
   within-dataset to across-modality, anchored by a within-dataset control that
   makes the decline interpretable.
2. Evidence that benchmark validity depends on the method pool it is assessed
   over, with the size of the effect: 98% of ceiling over a classical pool
   against 49% over a pool including modern architectures.
3. A demonstration that cross-domain rank comparison in a mixed pool measures
   training-domain match, with a worked case in which this inverts the sign of
   the reported correlation.
4. A tested and failed reduction of benchmark validity to a measured image
   statistic, with the residual quantified across three generator families.
5. A within-modality relationship between a reference's lesion fraction and the
   direction of its design preference, confirmed across seeds, metrics and pool
   composition, with two held-out datasets placed on it.

### 1.2 What this paper does not claim

We do not propose a benchmark design specification, because we tested the most
obvious candidate and it failed. We do not claim the framing relationship
transfers across modality, because our own cross-modality result shows the
underlying quantity is not interpretable there. We do not report a threshold
value for others to apply. We do not claim generality beyond the two modalities
studied.

---

## 2. Related work

### 2.1 Ranking instability

Maier-Hein et al. (Nature Communications, 2018) is the problem statement for
this paper. Across 150 challenges they found rankings sensitive to test-case
selection, ranking scheme, annotator and aggregation. Metrics Reloaded (Nature
Methods, 2024) and its companion pitfalls paper give a framework for choosing
metrics appropriate to a problem. Recent reaggregation work shows that identical
predictions can yield different top-ranked architectures depending on how
per-case scores are pooled.

Our contribution sits one step downstream. That literature tells a practitioner
that their ranking may not be robust. We measure how far agreement falls as the
reference moves away from the one at hand, and we add a source that literature
does not treat: the composition of the method pool.

### 2.2 Metrics for synthetic medical images

Work on no-reference image quality metrics for generated medical images finds
that widely used metrics correlate poorly with downstream task suitability and
are insensitive to localised anatomical detail, and concludes that a
multifaceted validation framework is needed rather than any single score.

That finding is the closest neighbour to our non-reducibility result and we
state the relationship plainly: the shape of our negative result has precedent.
They show a single quality number does not predict whether generated data is
good for a task. We show a single measured background statistic does not predict
whether a benchmark orders methods the way clinical data does. The quantity
differs, suitability against rank fidelity, and our construction differs in that
the statistic was chosen as a design target and matched across three
independently built generator families, but we extend their conclusion to the
ranking axis rather than discovering it.

### 2.3 Synthetic data as a benchmark

Outside medical imaging, work in natural language processing has asked directly
whether model-generated data can serve as a benchmark rather than as training
data, finding that it works for simple tasks and fails for complex ones, and
identifying task complexity as the moderator. The question is therefore
established as legitimate. It has not been asked for medical image segmentation,
and that work points to task complexity rather than to any measurable property
of the generated data.

### 2.4 Preprocessing and framing

A multi-dataset, multi-seed study of preprocessing pipelines for optic disc and
cup segmentation reports that which configuration ranks best is dataset
dependent, with full-resolution contrast enhancement winning on one dataset and
region-of-interest cropping winning on another. It also reports that multi-seed
evaluation overturned conclusions that single-run experiments had supported.

This supports our framing result and supplies a precedent for the seed lesson,
which we encountered independently. The difference is the level: they report a
reversal between datasets, meaning the winning pipeline depends on which dataset
you use. We report a relationship within one dataset, in which cropping the same
images with the same annotations to different lesion fractions moves the
reference's design verdict monotonically through zero.

### 2.5 Domain randomization

The sim-to-real literature established that for training transfer one should
maximise variation, randomising textures, lighting and placement well beyond
realism. Our result concerns validation rather than training and points the
opposite way: for a benchmark intended to order methods as clinical data does,
the objective is to match the reference rather than to maximise variation. We
state the contrast explicitly because a reader's prior will be domain
randomization.
