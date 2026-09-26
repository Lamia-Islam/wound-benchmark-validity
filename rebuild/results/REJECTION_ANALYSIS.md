# Every reason a reviewer rejects this as useless

Written adversarially. Each item is the reviewer's argument stated as strongly
as it can honestly be made, followed by whether it is survivable and what would
answer it. Ordered by how likely it is to kill the paper.

---

## Tier 1: structural, and one of these is probably fatal as the paper stands

### 1. The method pool is not the method pool anyone cares about

21 of the 22 methods are classical computer vision: Otsu, adaptive thresholds,
k-means in Lab, mean shift, watershed, GrabCut, Chan-Vese, Canny fill, flood
fill, colour ratios. The 22nd is a three-level U-Net with 117,681 parameters.

A 2026 reviewer in medical image segmentation compares nnU-Net, SAM and MedSAM,
transformer hybrids, and foundation-model finetunes. The question "does a
synthetic benchmark rank methods the way clinical data does" is only interesting
if the methods are the ones people would choose between. Ranking Otsu against
k-means correctly says nothing about whether the benchmark can separate two deep
architectures, because the thing that separates deep architectures is data
diversity and training dynamics, not colour thresholds.

Worse, the failure mode is specific. Classical methods are ranked almost
entirely by gross colour and intensity behaviour. The generators' main knobs are
gross colour and intensity. So "the benchmark ranks these methods like clinical
data does" may be close to a tautology about colour statistics rather than a
statement about benchmark validity.

**Survivable?** Not by argument. This needs at least three or four modern
segmentation models in the pool, trained identically and ranked alongside the
classical ones. If the rank agreement holds when the pool includes real
architectures, the paper is transformed. If it does not hold, that is also a
finding and a more interesting one.

### 2. The circularity charge

The generators were built by looking at Medetec and FUSeg. Their parameters were
calibrated against statistics measured on Medetec and FUSeg. The benchmark is
then validated by correlating against Medetec and FUSeg.

The chroma fix makes this concrete and a sharp reviewer will find it: a defect
was diagnosed by comparing generator output to clinical images, corrected, and
the corrected generator then scored better against those same clinical images.
That is a fitting loop, whatever the intention.

**Survivable?** Partly. Generator C was built independently and the profile
descriptor was never tuned against the agreement values. Those defences need to
be stated explicitly and early, not buried. The stronger answer is a held-out
clinical reference never used in any design decision, which AZH now is, so the
headline agreement should be reported against AZH specifically.

### 3. Every correlation has n = 22 and the confidence intervals swallow the
### claims

The headline clinical ceiling is 0.567 with a bootstrap CI of [+0.132, +0.887].
The three wound pairs are 0.567, 0.513 and 0.451. Those are indistinguishable.
The claimed three-level hierarchy of 0.964, 0.864, 0.511 and 0.068 has, at each
level, either one pair or three, with overlapping intervals between the top two
levels.

At n = 22, a Spearman needs roughly |rho| >= 0.43 to reach p < 0.05. So the
statement "across modality there is no agreement" actually means "we cannot
detect agreement below about 0.43". That is evidence of absence being claimed
from an underpowered null.

The within-versus-across comparison is Mann-Whitney U = 18, p = 0.0238, on 3
pairs against 6. With that design, 0.0238 is the smallest p obtainable. The
result is maximally significant precisely because it is maximally underpowered,
and a statistically literate reviewer will say so in one line.

**Survivable?** Only by enlarging the method pool, which also fixes item 1. More
methods means a larger n for every correlation in the paper. This is the single
change with the widest effect.

### 4. The framing curve is a step function wearing a trend

The verdict statistic takes values +0.964, +0.964, +0.821, +0.821, +0.360,
-0.893, -0.929, -0.964 across the eight crop levels. It is a Spearman over seven
points, so it is bounded, saturating, and takes few distinct values. Reporting
Spearman -0.988 with p = 0.0001 on eight points of a saturating statistic
overstates what was measured. The honest description is a sharp transition
somewhere between lesion fraction 0.055 and 0.080, located to within one
interval, not a smooth monotonic law.

**Survivable?** Yes, by describing it accurately. Report the transition and its
bracket. Drop the word "law" and any language implying a fitted function. The
held-out placement of Medetec still works as a check on the sign and region.

### 5. Every framing result rests on one seed

The seven contrast benchmarks underlying every point on the curve, and every
verdict computed for every clinical reference, were generated with seed 2024.
Seed count has already overturned two conclusions in this project. The seed
repeat was written and never run.

**Survivable?** Only by running it. If the curve holds across seeds it becomes a
real result. If it does not, the paper loses its best finding and needs to be
rebuilt around the cross-modality result instead.

---

## Tier 2: serious, answerable in revision

### 6. Dice only

Metrics Reloaded is the current standard reference and it exists precisely to
stop single-metric evaluation. Boundary-sensitive metrics such as NSD and HD95
can rank methods differently from overlap metrics, especially for small lesions,
which is exactly the regime the framing result lives in. A reviewer will ask
whether the entire framing effect is an artifact of Dice's known small-structure
bias.

This is the most dangerous Tier 2 item, because there is a plausible mechanism.
Dice is biased by region size. The framing experiment varies region size. The
two are confounded by construction.

**Answer:** repeat the framing curve with at least one boundary metric. If the
transition persists, the confound is excluded and the result is much stronger.
If it does not, the finding is about Dice, which is still publishable but is a
different and smaller paper.

### 7. The crop uses the ground-truth mask

Crops are centred on the centroid of the ground-truth lesion and sized from the
true lesion area. That is oracle localisation. No real preprocessing pipeline
can do it, so the eight cropped references are not pipelines anyone could run.
The Diagnostics 2026 preprocessing study flags this same limitation in itself.

**Answer:** state it plainly as a construction choice rather than a proposed
method. The crops are instruments for varying one property, not preprocessing
recommendations. Also worth showing the three real datasets land in the right
places, which they do, since they were curated without any oracle.

### 8. Five clinical references are really about three independent sources

AZH and FUSeg come from the same clinic. Kvasir-SEG and its sessile subset come
from the same collection. So the count is Medetec, the AZH family, and the
Kvasir family. The entire cross-modality conclusion rests on one endoscopy
source.

**Answer:** report it as three sources, not five. Adding CVC-ClinicDB as a
second, independent endoscopy source would materially strengthen the
cross-modality claim, and it is a small download.

### 9. The three generator families are not independent

A, B and C were written by the same author with the same mental model of what a
wound photograph looks like. Calling them "independently constructed" invites
the objection that they share every unstated assumption, which is exactly the
kind of shared bias that would produce correlated failures.

**Answer:** soften the language to "constructed on different principles" and
state what each does not share. Generator C's spectral-noise construction is a
genuine difference and should be described in enough detail that a reader can
see it.

### 10. No comparison with generative models

In 2026 the obvious way to make synthetic medical images is a diffusion model.
The paper hand-builds parametric generators and never says why, nor compares
against a generated-image benchmark.

**Answer:** a short, honest paragraph. Parametric generators expose the design
properties as explicit knobs, which is what the ablation and the dose-response
require, and a diffusion model does not. That is a real methodological reason
and it reads well if stated up front rather than defensively.

---

## Tier 3: the "useless" charge itself

This is the review that actually ends it, and it is worth writing out in the
reviewer's voice:

> The authors establish that a synthetic benchmark of their own construction
> ranks twenty-one classical thresholding methods roughly the way two small
> wound datasets rank them; that this agreement depends on how the reference
> datasets were cropped; that it cannot be predicted from a single image
> statistic; and that none of it transfers to another imaging modality. I cannot
> identify a decision any researcher would make differently after reading this.

Everything in Tiers 1 and 2 is a technical objection with a technical answer.
This one is not. It says the paper has no consumer.

The honest position is that the paper currently has three candidate consumers
and serves none of them fully:

- Someone building a synthetic benchmark. They want a recipe. The paper tells
  them no single statistic works and the descriptor is not an objective. That is
  a warning, not a recipe.
- Someone validating a benchmark. The framing result is genuinely for them, but
  it is stated for one modality and one metric.
- Someone reading a benchmark paper. The cross-modality result is for them and
  is the most transferable thing here, but it is currently a secondary section.

**The strongest available fix is to make the cross-modality result the paper.**
Rank agreement between clinical references falls away with source and vanishes
across modality, measured against a within-dataset control of 0.964. That is a
general statement about what benchmark evaluation can and cannot tell you, it
does not depend on the synthetic generators at all, and it applies to every
reader of every benchmark paper. The synthetic-benchmark work then becomes the
case study that motivates and demonstrates it.

That reframing needs items 1 and 3 fixed to survive, because a hierarchy
measured over 22 classical methods is not a general statement about anything.

---

## What actually has to happen, in order

1. Enlarge the method pool with three or four modern segmentation models. This
   fixes items 1 and 3 together and is the highest-value change available.
2. Run the seed repeat on the framing curve. It is written and takes under an
   hour. Until it runs, the best result in the paper is provisional.
3. Repeat the framing curve with a boundary metric to exclude the Dice
   region-size confound. This is the objection most likely to be raised by a
   reviewer who knows the literature.
4. Add CVC-ClinicDB so the cross-modality claim rests on two endoscopy sources.
5. Restate the framing result as a bracketed transition rather than a fitted
   law, and report headline agreement against AZH as the reference no design
   decision touched.
6. Rewrite the manuscript around the cross-modality hierarchy, with the
   synthetic benchmark as the case study.

Items 2, 4 and 5 are cheap. Item 3 is a day. Item 1 is the real work and it is
the one that decides whether this is a JIIM paper or a MedIA paper.

---

## What survives a hostile read unchanged

Worth stating, because the list above is one-sided by design.

- The within-dataset control at 0.964 over disjoint slices. Nobody argues with
  it and it is what makes every other agreement number interpretable.
- The non-reducibility result. Three benchmarks matched on a measured statistic
  scoring 0.121, 0.434 and 0.189 is a clean negative with the residual
  quantified.
- The correction discipline. The chroma overclaim, the resolution-dependent
  statistic, the single-seed ceiling, the cross-modality noise floor were all
  caught internally. A reviewer who reads the limitations sections will trust
  the rest of the paper more, not less.
