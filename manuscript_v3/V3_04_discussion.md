# Manuscript V3, Part 4: discussion, limitations, declarations

---

## 8. Discussion

### 8.1 What a validity figure means

The common form of a synthetic-benchmark validity claim is a single number: our
benchmark agrees with clinical data at rho = x. This paper shows that number is
underspecified in three ways at once, and gives the size of each.

It depends on which clinical reference. Two datasets of the same modality agree
with each other at 0.46 to 0.66, against 0.94 for two halves of one dataset.
Quoting agreement against one reference tells the reader roughly half of what
they need.

It depends on the method pool. The same generator against the same references
reaches 98% of ceiling over a mostly classical pool and 49% over a pool
including modern architectures. This is the largest of the three dependences and
the least discussed in the literature.

It does not depend on any single measured property of the benchmark images in a
way that could be turned into a design specification, at least not the most
natural candidate. Benchmarks matched to 0.03 on a calibrated texture statistic
differ by 0.115 in validity.

The practical recommendation is narrow and we think defensible: report benchmark
agreement as a matrix over references rather than as a scalar, report the method
pool it was computed over, and include a within-dataset control so a reader can
tell how much of any decline is measurement noise. The control costs one extra
scoring run on a disjoint slice and is what makes every other number in this
paper interpretable.

### 8.2 Why the method-pool dependence is not obvious

It would be reasonable to expect a benchmark that orders twenty-two methods
correctly to order thirty-two correctly, since the added methods are simply more
points on the same scale. That is not what happens, for a reason visible in
Section 7.2: the added methods are not more of the same. They are trained, and
the untrained methods are not, and a benchmark's ability to reproduce the
ordering within each group is not the same as its ability to reproduce the
ordering between them.

This generalises beyond our setting. Any evaluation that mixes trained and
untrained methods, or methods trained on different corpora, is partly measuring
training-domain match. Where the comparison crosses domains, Section 7.2 shows
this can dominate to the point of flipping the sign of the reported correlation.
Pretrained foundation models make this more pressing, not less, since their
training corpora are large, opaque and unequal across methods.

### 8.3 The framing result and what it is for

Within a modality, one measurable property of a reference predicts the direction
of its design preference. A reference whose lesions fill little of the frame
rewards high lesion-to-surround contrast; one whose lesions fill a lot rewards
low contrast; the crossover is somewhere above lesion fraction 0.11 for this
pool.

The use is diagnostic rather than prescriptive. A researcher who concludes from
one clinical reference that a particular synthetic design property is realistic
can check whether that conclusion would survive a reference framed differently,
and the direction of the change is predictable from a quantity anyone can
compute in one line. That is a check on a conclusion, not a recipe for building
a benchmark.

We deliberately do not report the crossover as a number to apply. It moved by a
factor of two when the method pool changed, which is precisely the
pool-dependence the rest of the paper is about.

### 8.4 Matching rather than maximising

The domain randomization literature established that for training transfer one
should maximise variation. Our validation results point the other way. Within a
generator family, agreement rises monotonically with background texture towards
the clinical range, and the design property that dominates the others is the one
that brings the benchmark closer to clinical images rather than further from
them.

The two are not in conflict; they are different objectives. A benchmark exists
to order methods as the deployment distribution would, so it should resemble
that distribution. A training set exists to produce a method robust to
distributions it has not seen, so it should exceed it. Section 7.2 gives an
unusually clean illustration: models trained on one narrow domain reach 0.58
mean Dice on it and 0.02 to 0.06 on another, which is the failure domain
randomization exists to prevent.

---

## 9. Limitations

**Data.** Five reference sets from three independent sources, and one of those
sources supplies both endoscopy references. Every cross-modality conclusion
rests on a single endoscopy collection. A second, independent endoscopy source
would materially strengthen Sections 7.1 and 7.2 and is the most valuable single
addition to this study.

**Modalities.** Two, both of them colour photography of surface tissue under
illumination. Nothing here establishes behaviour for volumetric modalities such
as CT or MRI, where the notions of background texture and lesion fraction would
need to be redefined.

**Method pool realism.** The eleven trained methods are trained from random
initialisation on 393 images at 128 x 128. They are real architectures under a
real and identical protocol, which is what the paper requires, but they are not
the fully trained, augmented, pretrained models a practitioner would deploy.
Neither nnU-Net nor a segmentation foundation model is included, both of which
would need either a training budget or pretrained weights beyond what was
available. We claim the pool spans modern architectural variation, not that it
reproduces state-of-the-art performance.

**Sample size for correlations.** Every agreement value is a Spearman over 32
points. The detection limit is |rho| of about 0.35, so reported absences of
agreement are absences of detectable agreement. The comparison of same-modality
against across-modality pairs is a Mann-Whitney over 3 against 6 values, for
which p = 0.0238 is the smallest attainable value; the individual pair
significances in Section 7.1 carry that result, not the group test.

**Oracle cropping.** The eight framing references are cropped using the
ground-truth mask centroid and area. That is oracle localisation and no
deployable preprocessing pipeline could reproduce it. The crops are instruments
for varying one property in isolation, not proposed pipelines. The three real
clinical datasets, which were curated without any oracle, land in the predicted
places.

**Crop series overstate the framing effect.** Every real clinical dataset placed
on a crop-built curve lands above it, four of four placements across two
independently built curves. Cropping to a target lesion fraction discards
surrounding context that a natively framed dataset retains, so the two are not
equivalent objects at the same fraction. The crop series establishes direction
and ordering; it is not a calibrated predictor, and held-out residuals of 27% to
41% of the curve range on the uncontaminated curve reflect that.

**Generator independence.** The three generator families were written by the
same authors. Generator C was built specifically to break shared implementation
assumptions and shares no code path with generator B, but shared conceptual
assumptions cannot be excluded.

**FUSeg as a reference.** Eleven of 32 methods were trained on FUSeg training
images. Although the evaluation images are disjoint, the distribution is not,
and Section 7.2 shows the resulting inflation is large. We report headline
figures against Medetec and AZH for this reason, and the primary framing curve
is built from AZH crops. The FUSeg-built curve is reported alongside it as a
replication and carries this caveat.

**Coverage of the primary framing curve.** AZH is distributed as 224 px patches
at a median lesion fraction of 0.029, and cropping can only raise a lesion
fraction, so the AZH-built curve starts at 0.035. The low-fraction end, from
0.008 to 0.035, exists only on the contaminated FUSeg curve. Only one held-out
dataset, Medetec, falls inside the AZH curve's range.

**Contrast sweep resolution.** The verdict slope is fitted over seven contrast
levels.

---

## 10. Conclusions

Benchmark validity in medical image segmentation is a three-way property of the
benchmark, the clinical reference, and the method pool. We measured all three
dependences.

Rank agreement between clinical references falls from +0.944 within a dataset to
+0.563 between datasets of one modality and to no detectable agreement across
modalities, with a within-dataset control establishing that the decline is real.
A synthetic benchmark's measured validity halves, from 98% to 49% of the
clinical ceiling, when modern architectures enter the method pool. And the
natural reduction of validity to a single measured image statistic fails, with
benchmarks matched to 0.03 on that statistic differing by 0.115.

Within one modality, the fraction of the frame a reference's lesions occupy
predicts the direction of its bias over synthetic design, monotonically, across
four seeds, under both an overlap and a boundary metric, with two held-out
clinical datasets placed on the relationship.

The recommendation that follows is that a benchmark validity figure should be
reported as a matrix over references, with the method pool named and a
within-dataset control included, and that cross-domain rank comparisons over
pools mixing trained and untrained methods should not be interpreted as
statements about the data.

---

## 11. Declarations

**Data availability.** All five clinical reference sets are publicly available.
Medetec, FUSeg and AZH are distributed in the uwm-bigdata/wound-segmentation
repository; AZH is described in Wang et al., Scientific Reports 10:21897, 2020.
Kvasir-SEG and its sessile subset are distributed by Simula. No dataset is
redistributed with this work. Sources and exact handling, including the
differing mask-binarisation conventions, are documented in the repository.

**Code availability.** The full implementation, including all 32 methods, the
three generator families, every experiment script and the trained model
checkpoints, is available at the repository given in the manuscript header. Each
experiment script is resumable and writes its intermediate cells to disk.

**Funding.** None declared.

**Competing interests.** None declared.

**Ethics.** This study uses only publicly available, de-identified datasets and
required no ethical approval.

---

## 12. Figures

**Figure 1. Rank agreement falls in steps.** Forest plot of the eleven pairwise
clinical agreements from Table 7.1, ordered by relationship, with bootstrap
confidence intervals. Grouping bands mark same-dataset, same-source,
same-modality and across-modality. Source: `hierarchy32.json`.

**Figure 2. The cross-modality inversion is a composition artifact.** Two
panels. Left, the pooled across-modality correlation against the same quantity
within each subgroup. Right, mean Dice of the deep and non-deep subsets on each
of the five references, showing the two groups exchanging places between
modalities. Source: `kvasir_clinical.json`, `clinical_ranking_cache32.json`.

**Figure 3. Benchmark validity depends on the method pool.** Generator B
agreement against the neutral references as a function of texture amplitude,
plotted against the clinical ceiling, with the 22-method result shown for
comparison. Source: `benchmark_quality32.json`.

**Figure 4. The framing curve, from two references, under two metrics.** Verdict
slope against reference lesion fraction on a log axis. Each panel shows the
AZH-built curve, seven crop levels, and the FUSeg-built replication, eight crop
levels, both over four seeds, with Medetec overlaid as open predicted and filled
measured markers joined by a residual stem. Two panels, Dice and normalised
surface Dice. File: `fig4_framing_both.png`.

**Figure 5. Matched texture, different validity.** The four benchmarks of
Table 7.5 plotted as validity against measured texture, with the two matched
pairs bracketed. Source: `nonreduce32.json`.

---

## 13. Pre-submission checklist

1. Fill the literature citations in Section 2 with final bibliographic entries.
   The works are identified but several are cited descriptively.
2. Push the repository so the Data Availability and Code Availability statements
   in Section 11 are true. They are not yet.
3. All five figures exist.
4. Decide the target venue. The contribution is evaluation methodology rather
   than a wound-assessment application, so the framing of the title and abstract
   should follow the venue rather than the project's history.
5. Obtain a second, independent endoscopy source if at all possible before
   submission. It is the one addition that changes the strength of Sections 7.1
   and 7.2.
6. Confirm that no figure or table in the final manuscript quotes a number from
   the 22-method pool without labelling it as such. Sections 7.3 and 7.6 do this
   deliberately; nothing else should.
