# Manuscript V2 — reading order

| # | file | covers |
|---|---|---|
| 1 | `V2_01_front.md` | Title, structured abstract, Introduction, contributions, scope |
| 2 | `V2_02_methods.md` | Related work, the 22 methods, the texture statistic, data, protocol |
| 3 | `V2_03_results.md` | Results, Sections 7.1 to 7.7 |
| 4 | `V2_04_discussion.md` | Discussion, what did not reproduce, limitations, declarations, figure specs, checklist |

Figures included: `fig_main.png` (Figures 1 and 2), `fig3_feature_ablation.png`,
`fig4_examples.png`.

## What changed from V1

V1 led with "two synthetic benchmarks rank methods in significantly opposite
orders" (rho -0.526). That did not survive reconstruction and is now Section 9,
stated as superseded. V2 leads with five results that did survive or are new:

1. A texture statistic measurable on any image set, with a clinical target range
2. A threshold relationship between that statistic and predictive validity,
   confirmed across five seeds with non-overlapping ranges
3. A null control ruling out target visibility as the mechanism
4. Framing convention reversing a clinical dataset's apparent realism preference
5. A single classifier feature demonstrating the figure-ground mechanism

## Numbers that are settled

- clinical texture range 2.69 to 3.97, generator A 2.10 (canonical 512 px, 9 px window)
- clinical-to-clinical ceiling rho 0.567
- cross-seed: 0.431 at texture 2.14, 0.536 at 4.40, ranges non-overlapping, p 0.0040
- contrast null Spearman 0.143, p 0.780
- framing reversal +0.964 to -0.964
- a_localsd drop: 0.936 to 0.122 on a uniform-background benchmark
- U-Net 117,681 parameters, exact

## Before submission

See the checklist at the end of `V2_04_discussion.md`. The two that gate
everything: fill the [SEARCH] citations in Section 2, and push the repository so
the Data Availability statement in Section 11 becomes true. It is currently
false.
