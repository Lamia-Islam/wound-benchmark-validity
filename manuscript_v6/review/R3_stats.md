# R3: Statistical review of manuscript v6 (JIIM)

Reviewer role: statistics. Scope: Methods and Results of `manuscript_v6_text.txt`, checked against `rebuild/results/v7/{STATS,BENCH,PRACTICAL,ROBUST,ROBUST_POOLED}_V7.md` and the scripts that produced them (`stats_v7.py`, `bench_v7.py`, `practical_v7.py`, `robust_v7.py`, `robust_pooled_v7.py`, `perimage_v7.py`). No existing file was edited. Scratch scripts for the new numbers below are in `/tmp/claude-0/-home-claude/edc235aa-bf33-56a4-af46-8d8d7e42d15c/scratchpad/r3/` (`ident.py`, `seedcheck.py`, `selcheck.py`, `benchdiff.py`, `misscheck.py`, `synpool.py`).

## Verdict

Major revision. No critical issues.

- The stratification algebra is correct. The code does what the Methods say for the method bootstrap, the hierarchical bootstrap, the exact relabeling test, split-half reliability and the seed analyses. The numbers I spot-checked match the result files.
- The problems are in the inference drawn from the numbers:
  - the "confirmatory" label does not match the tests that were run;
  - the reading rule fixed in advance is described as fixed earlier than the files show, and the description leaves out one of its two branches;
  - the claim that the dataset-versus-seed gap "comes from" the nine original architectures is a subgroup contrast whose own interval includes zero;
  - the gap mixes dataset differences with the effect of the two seeds sharing the same test images;
  - the selection-loss comparison has no uncertainty and uses seed-averaged scores, which no deployable model produces;
  - some sentences turn a wide interval around zero into "as good as and no better", or into a "ceiling" that is not a ceiling.

All of these can be fixed in the text, most with numbers that already exist or that I computed below.

## What was verified

| Item | Result |
|---|---|
| Stratification identity, algebra | Correct line by line (details in m1). |
| Identity on real data, U+M19 and U+F, Dice and NSD | Exact for every wound pair and for the wound-pair mean (differences below 1e-15). Over all 55 pairs the largest single-pair difference between Pearson-on-percentiles and the (n_g - 1/n_g) weighted average is 0.0017 (U+M19, Dice), 0.0054 (U+F, Dice), 0.0009 (U+M19, NSD), 0.0052 (U+F, NSD). All of these differences involve TN3K (Dice: 5 tied untrained and 5 tied from-scratch means) or PH2 (NSD: 2 tied from-scratch means). |
| Method bootstrap | Methods resampled with replacement within each group, B 2000, default_rng(0) per set (`v5_method_takes`), percentile intervals. Matches the text. |
| Hierarchical bootstrap | Units resampled independently per dataset, photographs for Medetec (38), AZH (118) and control (89), then methods within groups, B 1000 in `stats_v7.py`. The benchmark hierarchical bootstrap in `bench_v7.py` uses B 2000 (see m3). |
| Exact relabeling test | 13,860 multiset permutations of the 11 labels, one-sided, observed assignment included, two equivalent assignments, so min p = 2/13,860. The 52-pair version drops the three within-source pairs by identity. Matches the text. |
| Split-half reliability | 500 unit splits, Spearman between halves, Spearman-Brown, median. Matches. |
| Seed analyses | Same-dataset r-bar over 3 seed pairs; cross-dataset cross-seed over 6 ordered seed pairs x 6 wound pairs = 36 correlations; gap = difference; architecture bootstrap B 2000; hierarchical B 1000. Matches. The ICC is computed but its formula is not in the Methods (m8). |
| Selection loss, miss rates | Definitions match `practical_v7.py`. Miss intervals resample units with models fixed, B 2000. Matches. |

## Major issues

### M1. "Confirmatory" analyses do not match the tests that were run

Quote (Methods, Confirmatory and exploratory analyses): "The composition effect of pooling and the modality structure within groups were the confirmatory analyses of the study design and are tested with the exact relabeling test."

**Problem**

- The exact relabeling test is applied only to the modality difference.
- The composition effect is judged with a paired method-bootstrap interval for "pooled minus the higher within-group agreement" (`robust_pooled_v7.py`, criterion C1). It is never tested with the relabeling test.
- "Confirmatory" implies a dated protocol that fixed the hypotheses, the method sets and the test before the data were seen. The manuscript cites none. The modern group used in the confirmatory table also changed between analysis versions, from M (11 methods) to M19. The V7 brief fixed M19 as primary before the V7 analysis, but after the earlier M results were known.
- Seven method sets are tested in Table 3 with no multiplicity statement, although all seven survive a Holm correction (largest Holm-adjusted p is 0.009, for F).

**Replacement**

> "Two analyses were planned as primary. The composition effect was assessed with the paired method-bootstrap interval of pooled agreement minus the higher within-group agreement. The modality structure within groups was assessed with the exact relabeling test. The modern group was redefined from 11 to 19 methods when the additional trainings became available, before the analyses reported here were run. With seven method sets tested, all modality differences in Table 3 remain below 0.05 after Holm correction. All other analyses (training seeds, benchmarks, selection and robustness) are exploratory. They are reported with intervals and are not corrected for multiple comparisons."

If no dated protocol exists, replace "confirmatory" with "primary" throughout, including Table 3 and the Limitations.

### M2. The reading rule fixed in advance: timing and the missing second branch

**Quotes**

- Methods: "One reading rule was fixed before the new trainings were scored".
- Results: "The rule fixed before the seed experiment required same-dataset cross-seed agreement above 0.7."

**Problem 1: timing.** The two sentences give different times, and neither matches the files.

- The rule is in `results/v7_morning/BRIEF_V7.md` (file time 2026-10-11 10:20).
- That brief already lists the three diverged trainings and refers to existing score files in `results/kaggle2/scores` (file times 2026-10-10).
- So the rule was written after training and scoring, though apparently before the seed analysis. File times may reflect copying, but the brief's content alone shows that training had finished.

**Problem 2: the second branch.** The brief has two branches. The manuscript reports only the first. The second reads: "If cross-seed agreement is as low as cross-dataset agreement, say so plainly: then modern models are statistically indistinguishable and the framing must change."

**What was observed.** The result fell between the two branches:

- same-dataset cross-seed 0.45;
- cross-dataset cross-seed 0.10.

**Problem 3: the replacement criterion.** The Discussion then adopts a criterion that was not fixed in advance (gap above zero, in a subgroup).

**Replacements**

Methods:

> "One reading rule was written down after the new models had been trained and scored, but before any seed analysis was run. The statement that modern models disagree across wound datasets because the datasets differ, not because of training noise, would be supported if same-dataset cross-seed agreement exceeded 0.7 while cross-dataset agreement stayed near 0.1 to 0.3. If cross-seed agreement were as low as cross-dataset agreement, the modern models would be treated as not distinguishable."

Results:

> "The rule written down before the seed analysis required same-dataset cross-seed agreement above 0.7. It was not met (0.45). Cross-seed agreement was also not as low as cross-dataset agreement (0.10), so neither branch of the rule applies. The comparison of the two that follows was not specified in advance."

### M3. "The gap comes from the nine original architectures" is not supported as a subgroup contrast

**Quotes**

- Results: "The gap comes from the nine original architectures (0.49); among the seven new ones it is 0.05, so for them disagreement across datasets is no larger than disagreement between seeds."
- Discussion: "for the original architectures, two wound datasets disagree more than two seeds do, by a margin whose interval excludes zero. ... It did not appear among the new architectures".

**Problem: the difference between the subgroups is not established.** With 9 and 7 architectures, each subgroup gap is very imprecise:

- 9 original architectures: 0.49 (0.09 to 0.85);
- 7 new architectures: 0.05 (-0.12 to 0.26).

I resampled the architectures of the two subgroups independently (B 2000). The difference in gap is 0.44 (-0.01 to 0.84), so the data do not establish that the subgroups differ.

**Problem: the two subgroups differ in seed agreement, not in cross-dataset agreement.** The 7 new architectures have the higher cross-dataset agreement of the two subgroups:

| Subgroup | Same-dataset cross-seed | Cross-dataset cross-seed |
|---|---|---|
| 9 original | 0.56 | 0.07 |
| 7 new | 0.26 (Medetec alone: -0.19) | 0.20 |

So the small gap among the new architectures comes from their poor seed-to-seed reproducibility. It does not show that their datasets agree as well as their seeds do.

**Problem: confounding with the session.** For the 9 original architectures, seed 0 was trained in a different session from seeds 1 and 2, so session and seed effects cannot be separated (the Limitations already say so). Among them, the same-session pair r12 is higher than r01 and r02 on all 11 datasets (0.54 against 0.35).

**Replacements**

Results:

> "The gap was 0.49 (0.09 to 0.85) for the nine original architectures and 0.05 (-0.12 to 0.26) for the seven new ones. The difference between these two gaps, 0.44 (-0.01 to 0.84), is too uncertain to attribute the gap to one subgroup. The new architectures also agreed less across seeds on the same dataset (0.26 against 0.56)."

Discussion:

> "Second, the datasets may differ. Over the 16 architectures, two wound datasets disagree more than two seeds do (gap 0.35, 0.11 to 0.56). The gap was larger for the original architectures than for the new ones, but the two subgroups cannot be told apart at these sizes."

Table 5: add a row "Gap, original minus new: 0.44 (-0.01 to 0.84)", or drop the two subgroup rows.

### M4. The gap includes the effect of shared test images, and its hierarchical interval is not conservative

**Quotes**

- Methods: "The gap between the two measures how much more two datasets disagree than two trainings do."
- Discussion: "That part of the disagreement is a property of the data rather than of training."
- Methods: "hierarchical intervals therefore sit partly below the estimate and are conservative for claims that agreement exceeds zero."

**Problem: shared images.** Same-dataset cross-seed agreement compares two seeds scored on the same images. Cross-dataset agreement changes both the dataset and the image sample, so the gap contains two things:

- real differences between the datasets;
- the agreement that two seeds get only because they are scored on the same finite set of images.

I tested this with random half splits of units (300 per dataset, 16 architectures):

| Architectures | Shared halves | Independent halves |
|---|---|---|
| All 16 | 0.46 | 0.36 |
| 9 original | 0.55 | 0.49 |
| 7 new | 0.29 | 0.18 |

At full size the inflation should be smaller. Even so, it is a non-trivial part of a gap of 0.35, so the gap is an upper bound on the part due to datasets.

**Problem: direction of the hierarchical bias.** Within each hierarchical draw, both seeds share one unit resample, while different datasets are resampled independently. Resampling therefore lowers cross-dataset agreement more than same-dataset agreement, and the gap draws are centred above the estimate. In a units-only bootstrap with architectures fixed, the mean gap was:

| Architectures | Mean of gap draws | Estimate |
|---|---|---|
| All 16 | 0.365 | 0.346 |
| 7 new | 0.129 | 0.055 |

The reported hierarchical gap interval (0.12 to 0.61) is therefore shifted up, not conservative.

**Replacements**

Methods:

> "The gap between the two measures how much more two datasets disagree than two trainings scored on the same images do. Because the two seeds share their test images, the gap also contains image-sampling noise and is an upper bound on the part of the disagreement due to the datasets. Hierarchical intervals pull single agreement values towards zero. For the gap they do the reverse, because seeds on the same dataset share their resampled images."

Discussion:

> "That part of the disagreement reflects the datasets and their finite image samples rather than training. With independent halves of the same dataset, cross-seed agreement fell from 0.46 to 0.36, so a portion of the gap is due to image sampling."

### M5. The selection loss uses non-deployable seed averages and has no uncertainty

**Quotes**

- Abstract: "A modern model chosen on one wound dataset lost 0.047 Dice on average on another, against 0.063 for a random choice".
- Discussion: "a model chosen on one wound dataset did only modestly better on another than a random choice."
- Limitations: "The selection losses are point estimates".

**Problem: the candidates are not models.** The candidates are seed-averaged per-image scores. The manuscript itself says, for misses, that "a seed-averaged score is not the output of any model". With the deployable single-seed candidate sets (`practical_v7.json`, M19_s0, M19_s1, M19_s2), the chooser does about as well as a random pick:

| Candidate set | Chooser loss | Random loss |
|---|---|---|
| Seed 0 models | 0.058 | 0.060 |
| Seed 1 models | 0.076 | 0.084 |
| Seed 2 models | 0.053 | 0.059 |

**Problem: no uncertainty.** I ran a unit bootstrap (B 2000, same unit definitions) on the seed-averaged set:

| Quantity | Estimate | 95% interval |
|---|---|---|
| Chooser loss | 0.047 | 0.033 to 0.073 |
| Random loss | 0.063 | 0.055 to 0.081 |
| Paired difference | -0.016 | -0.032 to 0.003 |

Resampling the 19 candidates instead gives a difference interval of -0.032 to 0.002. Both losses are inflated, because the "best" model on the deployment dataset is the maximum of noisy means. The paired difference is less affected.

**A useful reference.** Choosing on one random half of a wound dataset and deploying on the other half loses only 0.002 to 0.024. The cross-dataset loss of 0.047 is larger than image noise alone produces.

**Replacements**

Abstract:

> "An architecture chosen on one wound dataset lost 0.047 Dice on average on another, against 0.063 for a random choice (difference -0.016, 95% interval -0.032 to 0.003). With single trained models the advantage over a random choice was 0.002 to 0.008."

Results: add the three single-seed chooser losses next to the single-seed synthetic picks already in Table 5, and give the bootstrap intervals above.

Limitations: replace "The selection losses are point estimates" with:

> "Intervals for selection losses resample images with the candidates fixed. Both losses are measured against the best observed model, which is biased upwards."

### M6. Equivalence and "ceiling" statements that the data do not support

**Quotes**

- Discussion: "Ordering modern models among themselves is the question a vendor, a hospital or a regulator usually faces, and its ceiling is lower than it seems. Wound datasets agree with each other at 0.21 among modern models, so a benchmark that agrees with them at 0.21 is as good a proxy as another clinical dataset and no better."
- Conclusions: "A synthetic benchmark can then be validated no better than one clinical dataset predicts another."

**Problem 1: equivalence.** The difference is 0.00 with interval -0.41 to 0.30 (hierarchical -0.38 to 0.27). An interval this wide cannot support "as good as and no better".

**Problem 2: the ceiling.** Inter-dataset agreement is not a ceiling on benchmark agreement. Suppose each dataset ranking is a shared signal plus independent noise. A benchmark that captured the shared signal would agree with each dataset at about the square root of their mutual agreement: about 0.46 when that agreement is 0.21 (exact for Pearson, approximate for Spearman). Such a benchmark would agree with each dataset more than the datasets agree with each other. Low inter-dataset agreement limits what can be shown, not what a benchmark can achieve.

**Replacements**

Discussion:

> "Wound datasets agree with each other at 0.21 among modern models, and the benchmark agrees with them at 0.21. The difference is 0.00, but its interval (-0.41 to 0.30) is too wide to say whether the benchmark is a better or a worse proxy than another clinical dataset. When clinical datasets agree this weakly, a validation study against them has little power to show that a benchmark is good."

Conclusions:

> "When clinical datasets agree weakly with each other, a validation against them can say little about how well a synthetic benchmark orders the models."

### M7. "Oracle threshold lowers agreement" is not supported

**Quotes**

- Results: "With a per-dataset oracle threshold for every model, which removes calibration differences, agreement falls to 0.12 (−0.13 to 0.40)."
- Discussion: "an oracle threshold that removes calibration differences lowers agreement."

**Problem 1: the change is within noise.** The paired change from the main protocol is -0.089 (-0.279 to 0.083) (ROBUST_V7, oracle row). The Robustness section itself lists the oracle among the changes whose interval includes zero.

**Problem 2: the oracle is limited by the grid.** On Medetec and CO2Wounds, 92% and 96% of the models have their oracle threshold at the grid minimum of 0.05. The oracle therefore does not remove calibration differences on those datasets.

**Replacements**

Results:

> "With a per-dataset oracle threshold for every model, agreement is 0.12 (-0.13 to 0.40). This is not distinguishable from the main protocol (change -0.09, -0.28 to 0.08). On Medetec and CO2Wounds most oracle thresholds sit at the lowest grid value, so this setting removes calibration differences only in part."

Discussion:

> "an oracle threshold, which reduces calibration differences, did not raise agreement."

## Minor issues

### m1. Stratification identity: state exactly when it holds, and match the code

Quote: "With ties the variances differ slightly between datasets and the identity holds approximately. ... In our data the Pearson value matched the identity to rounding error".

**The algebra, line by line**

1. With average ranks, the mean rank in a group of n_g is (n_g + 1)/2, so the mean percentile (r - 0.5)/n_g is 1/2 in every group, with or without ties.
2. Both datasets therefore have all group means equal to the pooled mean. The between-group term of the pooled covariance is zero, and the pooled covariance is (1/N) sum n_g c_g, using population covariances.
3. Each pooled variance is (1/N) sum n_g v_gA on dataset A, and likewise for B.
4. Ranks 1 to n have population variance (n^2 - 1)/12, so without ties v_g = (1 - 1/n_g^2)/12.
5. Pearson on within-group percentiles equals Pearson on average ranks, which is the tie-corrected Spearman r_g. This holds with ties as well.
6. Hence, in general, the result is

   sum n_g r_g sqrt(v_gA v_gB) / sqrt(sum n_g v_gA x sum n_g v_gB).

   Without ties this reduces to sum w_g r_g / sum w_g, with w_g = n_g v_g x 12 = n_g - 1/n_g.
7. Means over pairs follow by linearity only when the weights are the same for every pair, that is, without ties.

**What needs fixing**

- In the data, ties occur only on TN3K (Dice: 5 untrained and 5 from-scratch methods with identical means) and PH2 (NSD: 2 from-scratch). Single-pair deviations are therefore up to 0.005 (U+F), not "rounding error". Wound pairs are exact.
- `stats_v7.py` and `robust_pooled_v7.py` call "size-weighted average" the average with weights n_g, not n_g - 1/n_g. The two agree to within 0.0003 here, but the paper should name one.
- The identity is the classical pooled within-group correlation. Saying so would help readers.

**Replacement**

> "The identity holds exactly when no two methods in a group have the same mean on either dataset. With ties, each group is weighted by n_g times the square root of its two percentile variances, and the weights then depend on the pair. In our data ties occur only on TN3K and PH2. There the Pearson value differs from the weighted average by at most 0.006 for a single pair and by less than 0.001 for means over pairs. For the wound pairs it is exact. This is the pooled within-group correlation, applied to within-group ranks."

### m2. Split-half reliability over 65 methods is inflated by the composition effect, and does not cover retraining

Quote: "Each dataset ranks the methods reproducibly: split-half reliability over all 65 methods is 0.97 to 0.99 on the 11 datasets, and among the 19 modern models it is 0.82 to 0.93 on the four wound datasets."

**Problems**

- A pooled reliability shows the same between-group inflation that the paper describes for agreement.
- Within groups the values are 0.94 to 0.99 (U), 0.85 to 0.99 (F) and 0.84 to 0.99 (K) on the 11 reference datasets.
- For the modern models, the reliability holds the trained models fixed. It says nothing about retraining, whose same-dataset agreement is 0.45.

**Replacement**

> "With the trained models fixed, each dataset ranks the methods reproducibly across its images. Split-half reliability within groups is 0.84 to 0.99 on the 11 datasets, and 0.82 to 0.93 for the 19 modern models on the wound datasets. This reliability does not include retraining."

### m3. Number of hierarchical draws

Quote: "The hierarchical bootstrap also resamples the units of every dataset independently (1,000 draws)".

**Problem.** `bench_v7.py` uses 2,000 hierarchical draws (default_rng(1)) for the synthetic-benchmark intervals, for example the hierarchical interval -0.38 to 0.27.

**Replacement**

> "(1,000 draws for the clinical and seed analyses, 2,000 for the synthetic-benchmark analyses)".

### m4. What the method bootstrap means, and ties from duplicated methods

**Problems**

- The Methods do not say what the methods are a sample of.
- The Methods do not say that ties are handled with average ranks.
- A resample of n methods with replacement contains on average about 63% distinct methods: about 4 to 6 for 7 to 9 architectures. Spearman on so few distinct items moves in coarse steps.
- Percentile intervals for correlations at n 7 to 19 have uncertain coverage.

**Replacement** (after "(2,000 draws)")

> "These intervals show how much agreement depends on which methods of each kind are included, treating the methods as a sample of comparable methods. Spearman correlations use average ranks for ties, which occur when a method is drawn more than once. With 7 to 19 methods, a draw contains few distinct methods, so these intervals are approximate."

### m5. Scope of "conservative" for hierarchical intervals

Quote: "hierarchical intervals therefore sit partly below the estimate and are conservative for claims that agreement exceeds zero."

**Problems**

- The statement is right for single agreement values.
- It does not hold for the seed gap (M4).
- It also makes hierarchical intervals unsuitable as evidence that agreement is low. Examples: "the four wound datasets agree at 0.21 on average (hierarchical interval −0.10 to 0.49)" and Table 5. A downward pull also lowers the upper bound.

**Replacement**

> "...conservative for claims that a single agreement value exceeds zero, but not for claims that agreement is low, and not for differences in which one term shares images across seeds."

Report the method-bootstrap interval next to the hierarchical one wherever the claim is that agreement is low.

### m6. p values for exploratory results

Quote: "0.02 with the seed-1 models (p = 0.38) and 0.18 with the seed-2 models (nine architectures; p = 0.022), and the eight new architectures show none (0.05, p = 0.33)".

**Problems**

- The Methods say that exploratory analyses are reported with intervals.
- Seed 0 has no p value (it is 0.00072).
- "Not distinguishable from zero" for the U+F excess is fine, but it is clearer to give the interval.

**Replacement**

> "...0.46 with the seed-0 models, 0.02 with the seed-1 models and 0.18 with the seed-2 models (nine architectures), and 0.05 for the eight new architectures (exploratory; exact p 0.0007, 0.38, 0.022 and 0.33, uncorrected)."

For U+F: "the excess over the untrained group, 0.16, has an interval that includes zero (-0.07 to 0.35)."

### m7. "Validity per reference varies widely" needs paired intervals

Quote: "Among untrained methods the benchmark agrees with CO2Wounds-V2 at 0.80 and with AZH at 0.31 ... Reported against a single dataset, the same benchmark and pool could be described as valid or invalid."

**Problem.** Per-reference intervals are wide: U with AZH 0.31 (-0.23 to 0.73). I computed paired method-bootstrap differences (B 2000):

| Set | Comparison | Difference | 95% interval |
|---|---|---|---|
| U | CO2Wounds minus AZH | 0.49 | -0.04 to 1.05 |
| K | AZH minus FUSeg | 0.90 | 0.16 to 1.48 |
| M19 | CO2Wounds minus FUSeg | 0.43 | -0.11 to 0.95 |

**Replacement**

> "Validity per reference varies, although single per-reference values are imprecise. Among Kvasir-trained methods the benchmark agrees with AZH at 0.56 and with FUSeg at -0.34 (difference 0.90, 0.16 to 1.48). Among untrained methods it agrees with CO2Wounds-V2 at 0.80 and with AZH at 0.31 (difference 0.49, -0.04 to 1.05)."

### m8. The ICC is not defined in the Methods and is not a ranking reliability

Quote: "A single-seed ranking on one wound dataset has a reliability (intraclass correlation) of 0.27 to 0.77, and a three-seed average one of 0.52 to 0.91".

**Problems**

- The code (`stats_v7.py`, section 8d) computes a one-way random-effects ICC on mean Dice values, not on ranks:
  - T = Var(3-seed means) - mean seed variance / 3;
  - ICC(1) = T / (T + W);
  - ICC(3-seed) = T / Var(3-seed means).
- It has no interval.
- It can be negative (CVC-ColonDB).
- For 9 of the 16 architectures it mixes the session effect into W.

**Replacement** (Methods)

> "The reliability of an architecture's mean Dice was summarised by a one-way random-effects intraclass correlation over the 16 architectures. Between-architecture variance T is the variance of the three-seed means minus the mean seed variance divided by 3. The ICC is T/(T + W) for one seed and T divided by the variance of the three-seed means for a three-seed average, where W is the mean seed variance."

Results: "The mean Dice of a single-seed model has a reliability (intraclass correlation) of 0.27 to 0.77..." Add architecture-bootstrap intervals, or say there are none.

### m9. "The new architectures miss less than the original ten"

Quote: "The new architectures miss less than the original ten (24.9% against 30.3%)".

**Problem.** The paired unit-bootstrap difference is 5.4 points (3.2 to 7.7) with the architectures fixed. With the architectures also resampled it is -0.8 to 11.8. The nnU-Net interval (0.0% to 3.5%) excludes training noise, although pretrained architectures varied by up to 13.6 points between seeds.

**Replacement**

> "The new architectures missed fewer AZH wounds than the original ten (24.9% against 30.3%; difference 5.4 points, 3.2 to 7.7 with these architectures fixed, -0.8 to 11.8 when architectures are also resampled). nnU-Net was trained once, so its interval does not include training noise."

### m10. Padding correlations: clustered images, and comparing two dependent correlations

Quote: "The share of the 51 pretrained models that miss an image correlates with its black-pixel share (Spearman 0.73) more than with its lesion fraction (−0.46)."

**Problems**

- The 140 non-empty AZH images come from fewer photographs, so images are not independent.
- "More than" compares two correlations that share a variable.
- Black share and lesion fraction are themselves related.

**Replacement**

> "...correlates with its black-pixel share (Spearman 0.73) and with its lesion fraction (-0.46). The tercile analysis below separates the two."

Add photograph-bootstrap intervals for both correlations, or drop "more than".

### m11. "About what chance would give"

Quote: "only 1 of the 16 per-reference intervals excludes zero, about what chance would give."

**Replacement**

> "only 1 of the 16 per-reference intervals excludes zero. If there were no effect, about 0.8 would be expected, although the 16 intervals are not independent."

### m12. "Depends on the training seed" from three values

Quote: "Benchmark agreement among modern models also depends on the training seed. With single-seed versions of the ten original architectures it ranges from 0.26 to 0.40".

**Replacement**

> "Benchmark agreement among modern models also varied with the training seed..."

Use "varied with", not "depends on", wherever the evidence is three seeds with no interval for the differences. The same applies to the modality difference ("but it depends on the training seed" becomes "but it varied with the training seed").

### m13. The pooled benchmark is called "a worse proxy" from a methods-only interval

Quote: "Over the pooled set the benchmark is a worse proxy than another wound dataset: it agrees with each wound dataset less than the other wound datasets do, by −0.17 on average (−0.38 to −0.02)."

**Replacement**

> "...by -0.17 on average (-0.38 to -0.02; methods resampled only, because untrained methods have no per-image synthetic scores)."

### m14. Robustness wording: no excluded zero is not evidence of no change

Quote: "and none of these changes has an interval that excludes zero".

**Replacement**

> "...no change has an interval that excludes zero. The largest change was [value], with intervals up to [half-width] wide, so changes of that size cannot be ruled out."

Take the values from ROBUST_V7, where the largest M19 Dice change is the oracle, -0.089 (-0.279 to 0.083).

### m15. The exact test: what it conditions on, and what the pooled-set p values mean

**Problems**

- The relabeling test treats the observed correlation matrix as fixed. It asks whether the modality labels are special among the 11 datasets for this method set. It does not include method or image sampling.
- For pooled sets, Table 3's p = 0.00014 tests modality structure that the paper attributes to pooling.

**Replacement** (Methods)

> "The test treats the method set and images as fixed. For pooled sets it shows that pooling produces apparent modality structure, not evidence that such structure exists."

### m16. The wound-pair mean includes the within-source pair

**Problem.** The mean over the six wound pairs includes FUSeg~AZH (one clinic). Without it, the different-source mean changes several values:

| Set | Six wound pairs | Five different-source pairs |
|---|---|---|
| U | 0.37 | 0.41 |
| K | 0.36 | 0.44 |
| U+K | 0.28 | 0.37 |

M19 and U+M19 are unchanged.

**Replacement.** Add the five-pair mean in Table 3, or state in the Methods: "Results with the five different-source wound pairs only are in Online Resource 1."

### m17. Missing check worth adding: synthetic benchmark reliability

**Point.** The three synthetic seeds of generator B at amplitude 40 rank the 19 modern models alike: mean Spearman between synthetic seeds 0.94. Agreement with the wound datasets is 0.22 if the 135 images are pooled, against 0.21 as the mean over seeds.

**Replacement.** Add one sentence:

> "The benchmark itself ranks the modern models reproducibly (agreement between its three synthetic seeds 0.94), so its low agreement with the wound datasets is not due to its 45-image size."

### m18. Table 5 heading

**Problem.** "Ranking reliability across training seeds" mixes hierarchical and architecture-bootstrap rows.

**Replacement.** Label every row's interval type in the interval column, as is already done for the two subgroup rows, or give all rows the same type.

## Numbers computed for this review (not in the result files)

| Quantity | Value | Script |
|---|---|---|
| Pearson-on-percentiles vs (n_g - 1/n_g) identity, max single pair, Dice | U+M19 0.0017, U+F 0.0054 (TN3K ties); wound pairs exact | ident.py |
| Gap, 9 original minus 7 new | 0.44 (-0.01 to 0.84), 2.7% of draws at or below 0 | seedcheck.py |
| Same-dataset cross-seed, half-size, shared vs independent images | 16 architectures 0.46 vs 0.36; 9 original 0.55 vs 0.49; 7 new 0.29 vs 0.18 | seedcheck.py |
| Units-only bootstrap mean of the gap vs the estimate | 16 architectures 0.365 vs 0.346; 7 new 0.129 vs 0.055 | seedcheck.py |
| Cross-seed AZH vs AZH control (different images, same source), 16 architectures | 0.66 (same-image AZH r-bar 0.60) | seedcheck.py |
| Selection loss, M19 seed averages, unit bootstrap | chooser 0.047 (0.033 to 0.073); random 0.063 (0.055 to 0.081); difference -0.016 (-0.032 to 0.003) | selcheck.py |
| Selection loss, choose on one half and deploy on the other half of the same dataset | Medetec 0.024, CO2Wounds 0.024, FUSeg 0.007, AZH 0.002 | selcheck.py |
| Per-reference benchmark differences | U CO2 minus AZH 0.49 (-0.04 to 1.05); K AZH minus FUSeg 0.90 (0.16 to 1.48); M19 CO2 minus FUSeg 0.43 (-0.11 to 0.95) | benchdiff.py |
| AZH miss rate, original minus new architectures | 5.4 points (3.2 to 7.7) with architectures fixed; -0.8 to 11.8 with architectures resampled | misscheck.py |
| Agreement between synthetic seeds (M19, amplitude 40) | 0.94; pooled 135 images vs wound datasets 0.22 | synpool.py |

All computations were run with nice -n 10. Bootstrap seeds are fixed in the scripts.
