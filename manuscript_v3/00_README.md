# Manuscript V3 — reading order

| # | file | covers |
|---|---|---|
| 1 | `V3_01_front.md` | Title, abstract, introduction, contributions, related work |
| 2 | `V3_02_methods.md` | The 32-method pool, clinical references, generators, statistical protocol |
| 3 | `V3_03_results.md` | Sections 7.1 to 7.7 |
| 4 | `V3_04_discussion.md` | Discussion, limitations, conclusions, declarations, figures, checklist |

Figures: `fig1_hierarchy.png`, `fig2_composition.png`, `fig3_pool_dependence.png`,
`fig_framing_dice.png`, `fig_framing_nsd.png`, `fig5_nonreducibility.png`.

## What changed from V2

V2 led with a texture statistic presented as a benchmark design specification,
and with a clinical-to-clinical ceiling quoted as a single number. Both were
measured over 22 methods, 21 of them classical, on one seed, under one metric.

V3 leads with a different claim, because enlarging the method pool changed the
answers:

**Benchmark validity is a three-way property of the benchmark, the clinical
reference, and the method pool.** Every result is an instance of that.

1. Rank agreement between clinical references falls from +0.944 within a dataset
   to +0.563 within a modality to no detectable agreement across modalities.
2. The same generator reaches 98% of ceiling over the 22-method pool and 49%
   over the 32-method pool against references no method trained on.
3. Cross-domain rank comparison in a mixed pool measures training-domain match,
   and can invert the sign of the reported correlation.
4. Validity is not reducible to the measured texture statistic: benchmarks
   matched to 0.03 differ by 0.115.
5. Within a modality, a reference's lesion fraction predicts the direction of
   its design preference, across four seeds and two metrics.

## Numbers that are settled

- pool: 32 methods, eleven trained, no pretrained weights anywhere
- clinical ceiling, Medetec vs AZH, n=32: +0.568 (p 0.0012)
- within-dataset control, AZH test vs AZH train: +0.944
- same-source control, Kvasir vs sessile: +0.952
- same modality mean +0.563, across modality mean -0.253 pooled,
  +0.108 classical only, +0.236 deep only
- best generator B vs neutral references: +0.277 = 49% of ceiling
- framing curve: monotonic in 8 of 8 seed-metric combinations at p < 0.01,
  seed sd never above 0.008, sign change between 0.110 and 0.160
- non-reducibility: C at 3.20 and B at 3.23 differ by 0.115

## Withdrawn from earlier versions

- the rank inversion between synthetic benchmarks, rho -0.526: not reproduced
- the chroma correction as an improvement: seed-matched it is -0.001 overall
- the 99% and 98% of ceiling figures as general statements: they are
  pool-specific and reference-specific
- the framing crossover as a portable value: it moved by a factor of two
- any use of the saturating rank-based verdict statistic

## Before submission

See the checklist at the end of `V3_04_discussion.md`. The two that gate
everything: push the repository so the availability statements become true, and
get a second independent endoscopy source if at all possible.
