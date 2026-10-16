# Figure values, manuscript V6 (Figures 1 to 3)

Written by `scripts/v7/fig_v6.py` while drawing; every value below is the exact number plotted, read from the JSON path given (paths relative to `rebuild/results/v7/`). Values are shown to 6 decimals; the figures themselves print no numbers except the two reference values in the Figure 3b legend (2 decimals).

## Definitions

- Agreement: Spearman correlation of per-method mean scores (Dice) between two image sets over a method set.
- Figure 1: `comp|<domain>|<category>` = mean agreement over the dataset pairs of that category (domain with domain; domain with other; other datasets, same modality; other datasets, across modality). Interval: method-only bootstrap, stats_v5 design, B 2000, seed 0 (`method_only_repro_B2000`), the design of V5 Figure 1.
- Figure 2: benchmark = `mean|bench` (generator B, texture amplitude 40, mean over synthetic seeds and over the 4 wound references); wound with each other = `pairs6` (mean of the 6 wound pairs; equals `mean|other`). Interval: method bootstrap B 2000, default_rng(0).
- Figure 3a (C16, 16 architectures with 3 seeds): per wound dataset r-bar = mean of the 3 seed-pair Spearman values on that dataset (interval `rbar_method_boot`); `rbar_wound` = mean over the 4 wound datasets; `single|wound pairs` = wound-pair mean at one seed, averaged over seeds 0, 1, 2; `crossseed|wound pairs` = seed s on one dataset vs seed t on the other (s not t), 36 values; `rbar_control` = r-bar on the AZH train slice images; `single|same dataset` = AZH vs AZH train slice at one seed, averaged over seeds. Per-pair values (`observed_single_mean`) have no stored interval. Intervals: architecture bootstrap B 2000, seed 7070 (`method_boot`).
- Figure 3b: M19 wound-pair agreement with one common grid threshold for all 19 models (P0 pipeline); interval `ci` = method bootstrap B 2000, seed 0. Filled marker where the paired change from the own-threshold value has a 95% interval with lower bound above 0 (`delta_vs_default`).

## Checks run before drawing

- method-set size U equal in stats_v7 and bench_v7: passed (n 18)
- method-set size F equal in stats_v7 and bench_v7: passed (n 14)
- method-set size K equal in stats_v7 and bench_v7: passed (n 14)
- method-set size M equal in stats_v7 and bench_v7: passed (n 11)
- method-set size U+F equal in stats_v7 and bench_v7: passed (n 32)
- method-set size U+K equal in stats_v7 and bench_v7: passed (n 32)
- method-set size U+M equal in stats_v7 and bench_v7: passed (n 29)
- method-set size M11s equal in stats_v7 and bench_v7: passed (n 11)
- method-set size M19 equal in stats_v7 and bench_v7: passed (n 19)
- method-set size N8s equal in stats_v7 and bench_v7: passed (n 8)
- method-set size U+M11s equal in stats_v7 and bench_v7: passed (n 29)
- method-set size U+M19 equal in stats_v7 and bench_v7: passed (n 37)
- Fig1 U|wound|in x in equals stats_v5 comp: passed (max abs diff 5.6e-17)
- Fig1 U|wound|in x out equals stats_v5 comp: passed (max abs diff 5.6e-17)
- Fig1 U|wound|out, same modality equals stats_v5 comp: passed (max abs diff 0.0e+00)
- Fig1 U|wound|out, across modality equals stats_v5 comp: passed (max abs diff 6.9e-18)
- Fig1 U|wound|in x in equals stats_v5 comp: passed (max abs diff 5.6e-17)
- Fig1 U|wound|in x out equals stats_v5 comp: passed (max abs diff 5.6e-17)
- Fig1 U|wound|out, same modality equals stats_v5 comp: passed (max abs diff 0.0e+00)
- Fig1 U|wound|out, across modality equals stats_v5 comp: passed (max abs diff 6.9e-18)
- Fig1 F|wound|in x in equals stats_v5 comp: passed (max abs diff 2.8e-17)
- Fig1 F|wound|in x out equals stats_v5 comp: passed (max abs diff 5.6e-17)
- Fig1 F|wound|out, same modality equals stats_v5 comp: passed (max abs diff 0.0e+00)
- Fig1 F|wound|out, across modality equals stats_v5 comp: passed (max abs diff 1.7e-16)
- Fig1 U+F|wound|in x in equals stats_v5 comp: passed (max abs diff 2.2e-16)
- Fig1 U+F|wound|in x out equals stats_v5 comp: passed (max abs diff 1.1e-16)
- Fig1 U+F|wound|out, same modality equals stats_v5 comp: passed (max abs diff 0.0e+00)
- Fig1 U+F|wound|out, across modality equals stats_v5 comp: passed (max abs diff 2.8e-17)
- Fig1 U|colonoscopy|in x in equals stats_v5 comp: passed (max abs diff 0.0e+00)
- Fig1 U|colonoscopy|in x out equals stats_v5 comp: passed (max abs diff 5.6e-17)
- Fig1 U|colonoscopy|out, same modality equals stats_v5 comp: passed (max abs diff 5.6e-17)
- Fig1 U|colonoscopy|out, across modality equals stats_v5 comp: passed (max abs diff 5.6e-17)
- Fig1 K|colonoscopy|in x in equals stats_v5 comp: passed (max abs diff 2.2e-16)
- Fig1 K|colonoscopy|in x out equals stats_v5 comp: passed (max abs diff 5.6e-17)
- Fig1 K|colonoscopy|out, same modality equals stats_v5 comp: passed (max abs diff 5.6e-17)
- Fig1 K|colonoscopy|out, across modality equals stats_v5 comp: passed (max abs diff 6.9e-18)
- Fig1 U+K|colonoscopy|in x in equals stats_v5 comp: passed (max abs diff 2.2e-16)
- Fig1 U+K|colonoscopy|in x out equals stats_v5 comp: passed (max abs diff 4.2e-17)
- Fig1 U+K|colonoscopy|out, same modality equals stats_v5 comp: passed (max abs diff 5.6e-17)
- Fig1 U+K|colonoscopy|out, across modality equals stats_v5 comp: passed (max abs diff 1.4e-17)
- Fig1 number of V5-comparable cells: passed (28 cell checks: U wound (drawn in panels a and b, checked twice), F, U+F, U colonoscopy, K, U+K; max abs diff 2.2e-16)
- Fig2 U pairs6 equals mean|other: passed (5.6e-17)
- Fig2 U wound-to-wound equals stats_v7 method-only B2000: passed (2.8e-17)
- Fig2 U equals stats_v5 bench40: passed (1.1e-16)
- Fig2 F pairs6 equals mean|other: passed (1.1e-16)
- Fig2 F wound-to-wound equals stats_v7 method-only B2000: passed (2.8e-17)
- Fig2 F equals stats_v5 bench40: passed (0.0e+00)
- Fig2 K pairs6 equals mean|other: passed (1.1e-16)
- Fig2 K wound-to-wound equals stats_v7 method-only B2000: passed (5.6e-17)
- Fig2 K equals stats_v5 bench40: passed (5.6e-17)
- Fig2 M19 pairs6 equals mean|other: passed (0.0e+00)
- Fig2 M19 wound-to-wound equals stats_v7 method-only B2000: passed (2.8e-17)
- Fig2 U+F pairs6 equals mean|other: passed (1.1e-16)
- Fig2 U+F wound-to-wound equals stats_v7 method-only B2000: passed (3.3e-16)
- Fig2 U+F equals stats_v5 bench40: passed (1.1e-16)
- Fig2 U+K pairs6 equals mean|other: passed (2.8e-17)
- Fig2 U+K wound-to-wound equals stats_v7 method-only B2000: passed (8.3e-17)
- Fig2 U+K equals stats_v5 bench40: passed (5.6e-17)
- Fig2 U+M19 pairs6 equals mean|other: passed (1.1e-16)
- Fig2 U+M19 wound-to-wound equals stats_v7 method-only B2000: passed (2.2e-16)
- Fig3a C16 has 16 architectures: passed
- Fig3a mean of 4 per-dataset r-bar equals rbar_wound: passed
- Fig3a per_dataset azh_train_slice r-bar equals rbar_control: passed
- Fig3a 6 wound pairs: passed
- Fig3a mean of per-pair single-seed values equals single|wound pairs: passed
- Fig3b grid is 18 thresholds 0.05 to 0.90: passed
- Fig3b filled markers equal robust_v7 common_threshold_wound_pairs_delta_ci_above0: passed ([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
- Fig3b default equals stats_v7 M19 wound pairs: passed

## Figure 1

| panel | row / series | quantity | value | JSON path |
|---|---|---|---|---|
| a | U (18), domain with domain | estimate | 0.366357 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x in']['est']` |
| a | U (18), domain with domain | 95% lower | 0.138916 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x in']['method_only_repro_B2000'][0]` |
| a | U (18), domain with domain | 95% upper | 0.607847 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x in']['method_only_repro_B2000'][1]` |
| a | U (18), domain with other | estimate | 0.091603 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x out']['est']` |
| a | U (18), domain with other | 95% lower | -0.152398 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x out']['method_only_repro_B2000'][0]` |
| a | U (18), domain with other | 95% upper | 0.302308 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x out']['method_only_repro_B2000'][1]` |
| a | U (18), other, same modality | estimate | 0.859236 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, same modality']['est']` |
| a | U (18), other, same modality | 95% lower | 0.675889 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, same modality']['method_only_repro_B2000'][0]` |
| a | U (18), other, same modality | 95% upper | 0.938358 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, same modality']['method_only_repro_B2000'][1]` |
| a | U (18), other, across modality | estimate | 0.035392 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, across modality']['est']` |
| a | U (18), other, across modality | 95% lower | -0.201283 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, across modality']['method_only_repro_B2000'][0]` |
| a | U (18), other, across modality | 95% upper | 0.287150 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, across modality']['method_only_repro_B2000'][1]` |
| a | M19 (19), domain with domain | estimate | 0.211404 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|in x in']['est']` |
| a | M19 (19), domain with domain | 95% lower | -0.069542 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|in x in']['method_only_repro_B2000'][0]` |
| a | M19 (19), domain with domain | 95% upper | 0.501638 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|in x in']['method_only_repro_B2000'][1]` |
| a | M19 (19), domain with other | estimate | 0.144674 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|in x out']['est']` |
| a | M19 (19), domain with other | 95% lower | -0.055228 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|in x out']['method_only_repro_B2000'][0]` |
| a | M19 (19), domain with other | 95% upper | 0.377556 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|in x out']['method_only_repro_B2000'][1]` |
| a | M19 (19), other, same modality | estimate | 0.402807 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|out, same modality']['est']` |
| a | M19 (19), other, same modality | 95% lower | 0.153555 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|out, same modality']['method_only_repro_B2000'][0]` |
| a | M19 (19), other, same modality | 95% upper | 0.629898 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|out, same modality']['method_only_repro_B2000'][1]` |
| a | M19 (19), other, across modality | estimate | 0.006539 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|out, across modality']['est']` |
| a | M19 (19), other, across modality | 95% lower | -0.255980 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|out, across modality']['method_only_repro_B2000'][0]` |
| a | M19 (19), other, across modality | 95% upper | 0.308801 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|out, across modality']['method_only_repro_B2000'][1]` |
| a | U+M19 (37), domain with domain | estimate | 0.820650 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|in x in']['est']` |
| a | U+M19 (37), domain with domain | 95% lower | 0.774512 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|in x in']['method_only_repro_B2000'][0]` |
| a | U+M19 (37), domain with domain | 95% upper | 0.869590 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|in x in']['method_only_repro_B2000'][1]` |
| a | U+M19 (37), domain with other | estimate | -0.559943 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|in x out']['est']` |
| a | U+M19 (37), domain with other | 95% lower | -0.663887 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|in x out']['method_only_repro_B2000'][0]` |
| a | U+M19 (37), domain with other | 95% upper | -0.442821 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|in x out']['method_only_repro_B2000'][1]` |
| a | U+M19 (37), other, same modality | estimate | 0.820175 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|out, same modality']['est']` |
| a | U+M19 (37), other, same modality | 95% lower | 0.713666 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|out, same modality']['method_only_repro_B2000'][0]` |
| a | U+M19 (37), other, same modality | 95% upper | 0.890419 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|out, same modality']['method_only_repro_B2000'][1]` |
| a | U+M19 (37), other, across modality | estimate | 0.431975 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|out, across modality']['est']` |
| a | U+M19 (37), other, across modality | 95% lower | 0.230255 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|out, across modality']['method_only_repro_B2000'][0]` |
| a | U+M19 (37), other, across modality | 95% upper | 0.624468 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|out, across modality']['method_only_repro_B2000'][1]` |
| b | U (18), domain with domain | estimate | 0.366357 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x in']['est']` |
| b | U (18), domain with domain | 95% lower | 0.138916 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x in']['method_only_repro_B2000'][0]` |
| b | U (18), domain with domain | 95% upper | 0.607847 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x in']['method_only_repro_B2000'][1]` |
| b | U (18), domain with other | estimate | 0.091603 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x out']['est']` |
| b | U (18), domain with other | 95% lower | -0.152398 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x out']['method_only_repro_B2000'][0]` |
| b | U (18), domain with other | 95% upper | 0.302308 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x out']['method_only_repro_B2000'][1]` |
| b | U (18), other, same modality | estimate | 0.859236 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, same modality']['est']` |
| b | U (18), other, same modality | 95% lower | 0.675889 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, same modality']['method_only_repro_B2000'][0]` |
| b | U (18), other, same modality | 95% upper | 0.938358 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, same modality']['method_only_repro_B2000'][1]` |
| b | U (18), other, across modality | estimate | 0.035392 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, across modality']['est']` |
| b | U (18), other, across modality | 95% lower | -0.201283 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, across modality']['method_only_repro_B2000'][0]` |
| b | U (18), other, across modality | 95% upper | 0.287150 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, across modality']['method_only_repro_B2000'][1]` |
| b | F (14), domain with domain | estimate | 0.162637 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|in x in']['est']` |
| b | F (14), domain with domain | 95% lower | -0.154309 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|in x in']['method_only_repro_B2000'][0]` |
| b | F (14), domain with domain | 95% upper | 0.530710 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|in x in']['method_only_repro_B2000'][1]` |
| b | F (14), domain with other | estimate | 0.142061 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|in x out']['est']` |
| b | F (14), domain with other | 95% lower | -0.093849 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|in x out']['method_only_repro_B2000'][0]` |
| b | F (14), domain with other | 95% upper | 0.361360 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|in x out']['method_only_repro_B2000'][1]` |
| b | F (14), other, same modality | estimate | 0.879121 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|out, same modality']['est']` |
| b | F (14), other, same modality | 95% lower | 0.627442 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|out, same modality']['method_only_repro_B2000'][0]` |
| b | F (14), other, same modality | 95% upper | 0.964477 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|out, same modality']['method_only_repro_B2000'][1]` |
| b | F (14), other, across modality | estimate | 0.141728 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|out, across modality']['est']` |
| b | F (14), other, across modality | 95% lower | -0.053821 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|out, across modality']['method_only_repro_B2000'][0]` |
| b | F (14), other, across modality | 95% upper | 0.413323 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|out, across modality']['method_only_repro_B2000'][1]` |
| b | U+F (32), domain with domain | estimate | 0.528226 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|in x in']['est']` |
| b | U+F (32), domain with domain | 95% lower | 0.301048 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|in x in']['method_only_repro_B2000'][0]` |
| b | U+F (32), domain with domain | 95% upper | 0.698415 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|in x in']['method_only_repro_B2000'][1]` |
| b | U+F (32), domain with other | estimate | -0.263566 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|in x out']['est']` |
| b | U+F (32), domain with other | 95% lower | -0.441408 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|in x out']['method_only_repro_B2000'][0]` |
| b | U+F (32), domain with other | 95% upper | -0.092997 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|in x out']['method_only_repro_B2000'][1]` |
| b | U+F (32), other, same modality | estimate | 0.899304 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|out, same modality']['est']` |
| b | U+F (32), other, same modality | 95% lower | 0.814255 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|out, same modality']['method_only_repro_B2000'][0]` |
| b | U+F (32), other, same modality | 95% upper | 0.935338 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|out, same modality']['method_only_repro_B2000'][1]` |
| b | U+F (32), other, across modality | estimate | 0.406322 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|out, across modality']['est']` |
| b | U+F (32), other, across modality | 95% lower | 0.188642 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|out, across modality']['method_only_repro_B2000'][0]` |
| b | U+F (32), other, across modality | 95% upper | 0.599707 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|out, across modality']['method_only_repro_B2000'][1]` |
| c | U (18), domain with domain | estimate | 0.859236 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|in x in']['est']` |
| c | U (18), domain with domain | 95% lower | 0.675889 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|in x in']['method_only_repro_B2000'][0]` |
| c | U (18), domain with domain | 95% upper | 0.938358 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|in x in']['method_only_repro_B2000'][1]` |
| c | U (18), domain with other | estimate | 0.129718 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|in x out']['est']` |
| c | U (18), domain with other | 95% lower | -0.105914 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|in x out']['method_only_repro_B2000'][0]` |
| c | U (18), domain with other | 95% upper | 0.314775 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|in x out']['method_only_repro_B2000'][1]` |
| c | U (18), other, same modality | estimate | 0.366357 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|out, same modality']['est']` |
| c | U (18), other, same modality | 95% lower | 0.138916 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|out, same modality']['method_only_repro_B2000'][0]` |
| c | U (18), other, same modality | 95% upper | 0.607847 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|out, same modality']['method_only_repro_B2000'][1]` |
| c | U (18), other, across modality | estimate | -0.104149 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|out, across modality']['est']` |
| c | U (18), other, across modality | 95% lower | -0.250549 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|out, across modality']['method_only_repro_B2000'][0]` |
| c | U (18), other, across modality | 95% upper | 0.092853 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|out, across modality']['method_only_repro_B2000'][1]` |
| c | K (14), domain with domain | estimate | 0.829011 | `stats_v7.json['bootstrap']['dice|K']['comp|colonoscopy|in x in']['est']` |
| c | K (14), domain with domain | 95% lower | 0.596298 | `stats_v7.json['bootstrap']['dice|K']['comp|colonoscopy|in x in']['method_only_repro_B2000'][0]` |
| c | K (14), domain with domain | 95% upper | 0.921973 | `stats_v7.json['bootstrap']['dice|K']['comp|colonoscopy|in x in']['method_only_repro_B2000'][1]` |
| c | K (14), domain with other | estimate | 0.196923 | `stats_v7.json['bootstrap']['dice|K']['comp|colonoscopy|in x out']['est']` |
| c | K (14), domain with other | 95% lower | -0.127909 | `stats_v7.json['bootstrap']['dice|K']['comp|colonoscopy|in x out']['method_only_repro_B2000'][0]` |
| c | K (14), domain with other | 95% upper | 0.488059 | `stats_v7.json['bootstrap']['dice|K']['comp|colonoscopy|in x out']['method_only_repro_B2000'][1]` |
| c | K (14), other, same modality | estimate | 0.362637 | `stats_v7.json['bootstrap']['dice|K']['comp|colonoscopy|out, same modality']['est']` |
| c | K (14), other, same modality | 95% lower | 0.047689 | `stats_v7.json['bootstrap']['dice|K']['comp|colonoscopy|out, same modality']['method_only_repro_B2000'][0]` |
| c | K (14), other, same modality | 95% upper | 0.635609 | `stats_v7.json['bootstrap']['dice|K']['comp|colonoscopy|out, same modality']['method_only_repro_B2000'][1]` |
| c | K (14), other, across modality | estimate | 0.021245 | `stats_v7.json['bootstrap']['dice|K']['comp|colonoscopy|out, across modality']['est']` |
| c | K (14), other, across modality | 95% lower | -0.228934 | `stats_v7.json['bootstrap']['dice|K']['comp|colonoscopy|out, across modality']['method_only_repro_B2000'][0]` |
| c | K (14), other, across modality | 95% upper | 0.332879 | `stats_v7.json['bootstrap']['dice|K']['comp|colonoscopy|out, across modality']['method_only_repro_B2000'][1]` |
| c | U+K (32), domain with domain | estimate | 0.922104 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|in x in']['est']` |
| c | U+K (32), domain with domain | 95% lower | 0.834794 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|in x in']['method_only_repro_B2000'][0]` |
| c | U+K (32), domain with domain | 95% upper | 0.957526 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|in x in']['method_only_repro_B2000'][1]` |
| c | U+K (32), domain with other | estimate | 0.024851 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|in x out']['est']` |
| c | U+K (32), domain with other | 95% lower | -0.075556 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|in x out']['method_only_repro_B2000'][0]` |
| c | U+K (32), domain with other | 95% upper | 0.118569 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|in x out']['method_only_repro_B2000'][1]` |
| c | U+K (32), other, same modality | estimate | 0.275599 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|out, same modality']['est']` |
| c | U+K (32), other, same modality | 95% lower | 0.107815 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|out, same modality']['method_only_repro_B2000'][0]` |
| c | U+K (32), other, same modality | 95% upper | 0.436245 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|out, same modality']['method_only_repro_B2000'][1]` |
| c | U+K (32), other, across modality | estimate | -0.078776 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|out, across modality']['est']` |
| c | U+K (32), other, across modality | 95% lower | -0.232968 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|out, across modality']['method_only_repro_B2000'][0]` |
| c | U+K (32), other, across modality | 95% upper | 0.102078 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|out, across modality']['method_only_repro_B2000'][1]` |

## Figure 2

| panel | row / series | quantity | value | JSON path |
|---|---|---|---|---|
| - | 18 untrained [U]: benchmark with wound data | estimate | 0.525112 | `bench_v7.json['rel']['U']['40']['mean|bench']['est']` |
| - | 18 untrained [U]: benchmark with wound data | 95% lower | 0.204990 | `bench_v7.json['rel']['U']['40']['mean|bench']['lo']` |
| - | 18 untrained [U]: benchmark with wound data | 95% upper | 0.742757 | `bench_v7.json['rel']['U']['40']['mean|bench']['hi']` |
| - | 18 untrained [U]: wound data with each other | estimate | 0.366357 | `bench_v7.json['rel']['U']['40']['pairs6']['est']` |
| - | 18 untrained [U]: wound data with each other | 95% lower | 0.138916 | `bench_v7.json['rel']['U']['40']['pairs6']['lo']` |
| - | 18 untrained [U]: wound data with each other | 95% upper | 0.607847 | `bench_v7.json['rel']['U']['40']['pairs6']['hi']` |
| - | 14 from scratch [F]: benchmark with wound data | estimate | 0.222344 | `bench_v7.json['rel']['F']['40']['mean|bench']['est']` |
| - | 14 from scratch [F]: benchmark with wound data | 95% lower | -0.038362 | `bench_v7.json['rel']['F']['40']['mean|bench']['lo']` |
| - | 14 from scratch [F]: benchmark with wound data | 95% upper | 0.445343 | `bench_v7.json['rel']['F']['40']['mean|bench']['hi']` |
| - | 14 from scratch [F]: wound data with each other | estimate | 0.162637 | `bench_v7.json['rel']['F']['40']['pairs6']['est']` |
| - | 14 from scratch [F]: wound data with each other | 95% lower | -0.154309 | `bench_v7.json['rel']['F']['40']['pairs6']['lo']` |
| - | 14 from scratch [F]: wound data with each other | 95% upper | 0.530710 | `bench_v7.json['rel']['F']['40']['pairs6']['hi']` |
| - | 14 Kvasir-trained [K]: benchmark with wound data | estimate | 0.112454 | `bench_v7.json['rel']['K']['40']['mean|bench']['est']` |
| - | 14 Kvasir-trained [K]: benchmark with wound data | 95% lower | -0.350375 | `bench_v7.json['rel']['K']['40']['mean|bench']['lo']` |
| - | 14 Kvasir-trained [K]: benchmark with wound data | 95% upper | 0.536076 | `bench_v7.json['rel']['K']['40']['mean|bench']['hi']` |
| - | 14 Kvasir-trained [K]: wound data with each other | estimate | 0.362637 | `bench_v7.json['rel']['K']['40']['pairs6']['est']` |
| - | 14 Kvasir-trained [K]: wound data with each other | 95% lower | 0.047689 | `bench_v7.json['rel']['K']['40']['pairs6']['lo']` |
| - | 14 Kvasir-trained [K]: wound data with each other | 95% upper | 0.635609 | `bench_v7.json['rel']['K']['40']['pairs6']['hi']` |
| - | 19 modern [M19]: benchmark with wound data | estimate | 0.207749 | `bench_v7.json['rel']['M19']['40']['mean|bench']['est']` |
| - | 19 modern [M19]: benchmark with wound data | 95% lower | -0.167274 | `bench_v7.json['rel']['M19']['40']['mean|bench']['lo']` |
| - | 19 modern [M19]: benchmark with wound data | 95% upper | 0.491452 | `bench_v7.json['rel']['M19']['40']['mean|bench']['hi']` |
| - | 19 modern [M19]: wound data with each other | estimate | 0.211404 | `bench_v7.json['rel']['M19']['40']['pairs6']['est']` |
| - | 19 modern [M19]: wound data with each other | 95% lower | -0.069542 | `bench_v7.json['rel']['M19']['40']['pairs6']['lo']` |
| - | 19 modern [M19]: wound data with each other | 95% upper | 0.501638 | `bench_v7.json['rel']['M19']['40']['pairs6']['hi']` |
| - | 32 pooled: untrained + from scratch [U+F]: benchmark with wound data | estimate | 0.343811 | `bench_v7.json['rel']['U+F']['40']['mean|bench']['est']` |
| - | 32 pooled: untrained + from scratch [U+F]: benchmark with wound data | 95% lower | 0.069316 | `bench_v7.json['rel']['U+F']['40']['mean|bench']['lo']` |
| - | 32 pooled: untrained + from scratch [U+F]: benchmark with wound data | 95% upper | 0.570994 | `bench_v7.json['rel']['U+F']['40']['mean|bench']['hi']` |
| - | 32 pooled: untrained + from scratch [U+F]: wound data with each other | estimate | 0.528226 | `bench_v7.json['rel']['U+F']['40']['pairs6']['est']` |
| - | 32 pooled: untrained + from scratch [U+F]: wound data with each other | 95% lower | 0.301048 | `bench_v7.json['rel']['U+F']['40']['pairs6']['lo']` |
| - | 32 pooled: untrained + from scratch [U+F]: wound data with each other | 95% upper | 0.698415 | `bench_v7.json['rel']['U+F']['40']['pairs6']['hi']` |
| - | 32 pooled: untrained + Kvasir-trained [U+K]: benchmark with wound data | estimate | 0.319495 | `bench_v7.json['rel']['U+K']['40']['mean|bench']['est']` |
| - | 32 pooled: untrained + Kvasir-trained [U+K]: benchmark with wound data | 95% lower | 0.061527 | `bench_v7.json['rel']['U+K']['40']['mean|bench']['lo']` |
| - | 32 pooled: untrained + Kvasir-trained [U+K]: benchmark with wound data | 95% upper | 0.540497 | `bench_v7.json['rel']['U+K']['40']['mean|bench']['hi']` |
| - | 32 pooled: untrained + Kvasir-trained [U+K]: wound data with each other | estimate | 0.275599 | `bench_v7.json['rel']['U+K']['40']['pairs6']['est']` |
| - | 32 pooled: untrained + Kvasir-trained [U+K]: wound data with each other | 95% lower | 0.107815 | `bench_v7.json['rel']['U+K']['40']['pairs6']['lo']` |
| - | 32 pooled: untrained + Kvasir-trained [U+K]: wound data with each other | 95% upper | 0.436245 | `bench_v7.json['rel']['U+K']['40']['pairs6']['hi']` |
| - | 37 pooled: untrained + modern [U+M19]: benchmark with wound data | estimate | 0.646792 | `bench_v7.json['rel']['U+M19']['40']['mean|bench']['est']` |
| - | 37 pooled: untrained + modern [U+M19]: benchmark with wound data | 95% lower | 0.438089 | `bench_v7.json['rel']['U+M19']['40']['mean|bench']['lo']` |
| - | 37 pooled: untrained + modern [U+M19]: benchmark with wound data | 95% upper | 0.805436 | `bench_v7.json['rel']['U+M19']['40']['mean|bench']['hi']` |
| - | 37 pooled: untrained + modern [U+M19]: wound data with each other | estimate | 0.820650 | `bench_v7.json['rel']['U+M19']['40']['pairs6']['est']` |
| - | 37 pooled: untrained + modern [U+M19]: wound data with each other | 95% lower | 0.774512 | `bench_v7.json['rel']['U+M19']['40']['pairs6']['lo']` |
| - | 37 pooled: untrained + modern [U+M19]: wound data with each other | 95% upper | 0.869590 | `bench_v7.json['rel']['U+M19']['40']['pairs6']['hi']` |

## Figure 3

| panel | row / series | quantity | value | JSON path |
|---|---|---|---|---|
| a | same dataset, different seed: Medetec | estimate | 0.333333 | `stats_v7.json['seeds']['dice|C16']['per_dataset']['medetec']['rbar']` |
| a | same dataset, different seed: Medetec | 95% lower | -0.081358 | `stats_v7.json['seeds']['dice|C16']['per_dataset']['medetec']['rbar_method_boot'][0]` |
| a | same dataset, different seed: Medetec | 95% upper | 0.684321 | `stats_v7.json['seeds']['dice|C16']['per_dataset']['medetec']['rbar_method_boot'][1]` |
| a | same dataset, different seed: CO2Wounds-V2 | estimate | 0.263725 | `stats_v7.json['seeds']['dice|C16']['per_dataset']['co2wounds']['rbar']` |
| a | same dataset, different seed: CO2Wounds-V2 | 95% lower | -0.110813 | `stats_v7.json['seeds']['dice|C16']['per_dataset']['co2wounds']['rbar_method_boot'][0]` |
| a | same dataset, different seed: CO2Wounds-V2 | 95% upper | 0.570315 | `stats_v7.json['seeds']['dice|C16']['per_dataset']['co2wounds']['rbar_method_boot'][1]` |
| a | same dataset, different seed: FUSeg | estimate | 0.589216 | `stats_v7.json['seeds']['dice|C16']['per_dataset']['fuseg']['rbar']` |
| a | same dataset, different seed: FUSeg | 95% lower | 0.187898 | `stats_v7.json['seeds']['dice|C16']['per_dataset']['fuseg']['rbar_method_boot'][0]` |
| a | same dataset, different seed: FUSeg | 95% upper | 0.873661 | `stats_v7.json['seeds']['dice|C16']['per_dataset']['fuseg']['rbar_method_boot'][1]` |
| a | same dataset, different seed: AZH | estimate | 0.597059 | `stats_v7.json['seeds']['dice|C16']['per_dataset']['azh']['rbar']` |
| a | same dataset, different seed: AZH | 95% lower | 0.230847 | `stats_v7.json['seeds']['dice|C16']['per_dataset']['azh']['rbar_method_boot'][0]` |
| a | same dataset, different seed: AZH | 95% upper | 0.835368 | `stats_v7.json['seeds']['dice|C16']['per_dataset']['azh']['rbar_method_boot'][1]` |
| a | same dataset, different seed: mean of 4 datasets | estimate | 0.445833 | `stats_v7.json['seeds']['dice|C16']['summary']['rbar_wound']['est']` |
| a | same dataset, different seed: mean of 4 datasets | 95% lower | 0.177407 | `stats_v7.json['seeds']['dice|C16']['summary']['rbar_wound']['method_boot'][0]` |
| a | same dataset, different seed: mean of 4 datasets | 95% upper | 0.652786 | `stats_v7.json['seeds']['dice|C16']['summary']['rbar_wound']['method_boot'][1]` |
| a | different dataset, same seed: Medetec vs CO2Wounds-V2 (mean over 3 seeds, no interval) | estimate | 0.521569 | `stats_v7.json['seeds']['disattenuation']['dice|C16']['wound_pairs']['medetec~co2wounds']['observed_single_mean']` |
| a | different dataset, same seed: Medetec vs FUSeg (mean over 3 seeds, no interval) | estimate | 0.000000 | `stats_v7.json['seeds']['disattenuation']['dice|C16']['wound_pairs']['medetec~fuseg']['observed_single_mean']` |
| a | different dataset, same seed: Medetec vs AZH (mean over 3 seeds, no interval) | estimate | 0.210784 | `stats_v7.json['seeds']['disattenuation']['dice|C16']['wound_pairs']['medetec~azh']['observed_single_mean']` |
| a | different dataset, same seed: CO2Wounds-V2 vs FUSeg (mean over 3 seeds, no interval) | estimate | -0.037255 | `stats_v7.json['seeds']['disattenuation']['dice|C16']['wound_pairs']['co2wounds~fuseg']['observed_single_mean']` |
| a | different dataset, same seed: CO2Wounds-V2 vs AZH (mean over 3 seeds, no interval) | estimate | 0.281373 | `stats_v7.json['seeds']['disattenuation']['dice|C16']['wound_pairs']['co2wounds~azh']['observed_single_mean']` |
| a | different dataset, same seed: FUSeg vs AZH (mean over 3 seeds, no interval) | estimate | 0.084314 | `stats_v7.json['seeds']['disattenuation']['dice|C16']['wound_pairs']['fuseg~azh']['observed_single_mean']` |
| a | different dataset, same seed: mean of 6 pairs | estimate | 0.176797 | `stats_v7.json['seeds']['dice|C16']['summary']['single|wound pairs']['est']` |
| a | different dataset, same seed: mean of 6 pairs | 95% lower | -0.048215 | `stats_v7.json['seeds']['dice|C16']['summary']['single|wound pairs']['method_boot'][0]` |
| a | different dataset, same seed: mean of 6 pairs | 95% upper | 0.406877 | `stats_v7.json['seeds']['dice|C16']['summary']['single|wound pairs']['method_boot'][1]` |
| a | different dataset, different seed: mean of 6 pairs | estimate | 0.100163 | `stats_v7.json['seeds']['dice|C16']['summary']['crossseed|wound pairs']['est']` |
| a | different dataset, different seed: mean of 6 pairs | 95% lower | -0.119385 | `stats_v7.json['seeds']['dice|C16']['summary']['crossseed|wound pairs']['method_boot'][0]` |
| a | different dataset, different seed: mean of 6 pairs | 95% upper | 0.339586 | `stats_v7.json['seeds']['dice|C16']['summary']['crossseed|wound pairs']['method_boot'][1]` |
| a | control: Control images, different seed | estimate | 0.825490 | `stats_v7.json['seeds']['dice|C16']['summary']['rbar_control']['est']` |
| a | control: Control images, different seed | 95% lower | 0.551139 | `stats_v7.json['seeds']['dice|C16']['summary']['rbar_control']['method_boot'][0]` |
| a | control: Control images, different seed | 95% upper | 0.936321 | `stats_v7.json['seeds']['dice|C16']['summary']['rbar_control']['method_boot'][1]` |
| b | common threshold 0.05 | estimate | 0.370175 | `robust_v7.json['threshold']['common']['M19|0.05']['wound pairs']['est']` |
| b | common threshold 0.05 | 95% lower | 0.058213 | `robust_v7.json['threshold']['common']['M19|0.05']['wound pairs']['ci'][0]` |
| b | common threshold 0.05 | 95% upper | 0.637373 | `robust_v7.json['threshold']['common']['M19|0.05']['wound pairs']['ci'][1]` |
| b | common threshold 0.05: change from own threshold, 95% lower (filled marker if above 0) | value | -0.169747 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.05|wound pairs']['ci'][0]` |
| b | common threshold 0.10 | estimate | 0.219298 | `robust_v7.json['threshold']['common']['M19|0.10']['wound pairs']['est']` |
| b | common threshold 0.10 | 95% lower | -0.069365 | `robust_v7.json['threshold']['common']['M19|0.10']['wound pairs']['ci'][0]` |
| b | common threshold 0.10 | 95% upper | 0.510799 | `robust_v7.json['threshold']['common']['M19|0.10']['wound pairs']['ci'][1]` |
| b | common threshold 0.10: change from own threshold, 95% lower (filled marker if above 0) | value | -0.244469 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.10|wound pairs']['ci'][0]` |
| b | common threshold 0.15 | estimate | 0.212573 | `robust_v7.json['threshold']['common']['M19|0.15']['wound pairs']['est']` |
| b | common threshold 0.15 | 95% lower | -0.060475 | `robust_v7.json['threshold']['common']['M19|0.15']['wound pairs']['ci'][0]` |
| b | common threshold 0.15 | 95% upper | 0.511665 | `robust_v7.json['threshold']['common']['M19|0.15']['wound pairs']['ci'][1]` |
| b | common threshold 0.15: change from own threshold, 95% lower (filled marker if above 0) | value | -0.210873 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.15|wound pairs']['ci'][0]` |
| b | common threshold 0.20 | estimate | 0.256433 | `robust_v7.json['threshold']['common']['M19|0.20']['wound pairs']['est']` |
| b | common threshold 0.20 | 95% lower | -0.039082 | `robust_v7.json['threshold']['common']['M19|0.20']['wound pairs']['ci'][0]` |
| b | common threshold 0.20 | 95% upper | 0.542676 | `robust_v7.json['threshold']['common']['M19|0.20']['wound pairs']['ci'][1]` |
| b | common threshold 0.20: change from own threshold, 95% lower (filled marker if above 0) | value | -0.144427 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.20|wound pairs']['ci'][0]` |
| b | common threshold 0.25 | estimate | 0.271637 | `robust_v7.json['threshold']['common']['M19|0.25']['wound pairs']['est']` |
| b | common threshold 0.25 | 95% lower | -0.011012 | `robust_v7.json['threshold']['common']['M19|0.25']['wound pairs']['ci'][0]` |
| b | common threshold 0.25 | 95% upper | 0.559735 | `robust_v7.json['threshold']['common']['M19|0.25']['wound pairs']['ci'][1]` |
| b | common threshold 0.25: change from own threshold, 95% lower (filled marker if above 0) | value | -0.117707 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.25|wound pairs']['ci'][0]` |
| b | common threshold 0.30 | estimate | 0.276608 | `robust_v7.json['threshold']['common']['M19|0.30']['wound pairs']['est']` |
| b | common threshold 0.30 | 95% lower | -0.000015 | `robust_v7.json['threshold']['common']['M19|0.30']['wound pairs']['ci'][0]` |
| b | common threshold 0.30 | 95% upper | 0.563648 | `robust_v7.json['threshold']['common']['M19|0.30']['wound pairs']['ci'][1]` |
| b | common threshold 0.30: change from own threshold, 95% lower (filled marker if above 0) | value | -0.102648 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.30|wound pairs']['ci'][0]` |
| b | common threshold 0.35 | estimate | 0.267836 | `robust_v7.json['threshold']['common']['M19|0.35']['wound pairs']['est']` |
| b | common threshold 0.35 | 95% lower | -0.012731 | `robust_v7.json['threshold']['common']['M19|0.35']['wound pairs']['ci'][0]` |
| b | common threshold 0.35 | 95% upper | 0.541006 | `robust_v7.json['threshold']['common']['M19|0.35']['wound pairs']['ci'][1]` |
| b | common threshold 0.35: change from own threshold, 95% lower (filled marker if above 0) | value | -0.111640 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.35|wound pairs']['ci'][0]` |
| b | common threshold 0.40 | estimate | 0.315497 | `robust_v7.json['threshold']['common']['M19|0.40']['wound pairs']['est']` |
| b | common threshold 0.40 | 95% lower | 0.038889 | `robust_v7.json['threshold']['common']['M19|0.40']['wound pairs']['ci'][0]` |
| b | common threshold 0.40 | 95% upper | 0.578259 | `robust_v7.json['threshold']['common']['M19|0.40']['wound pairs']['ci'][1]` |
| b | common threshold 0.40: change from own threshold, 95% lower (filled marker if above 0) | value | -0.066450 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.40|wound pairs']['ci'][0]` |
| b | common threshold 0.45 | estimate | 0.374269 | `robust_v7.json['threshold']['common']['M19|0.45']['wound pairs']['est']` |
| b | common threshold 0.45 | 95% lower | 0.100998 | `robust_v7.json['threshold']['common']['M19|0.45']['wound pairs']['ci'][0]` |
| b | common threshold 0.45 | 95% upper | 0.611803 | `robust_v7.json['threshold']['common']['M19|0.45']['wound pairs']['ci'][1]` |
| b | common threshold 0.45: change from own threshold, 95% lower (filled marker if above 0) | value | 0.001438 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.45|wound pairs']['ci'][0]` |
| b | common threshold 0.50 | estimate | 0.411696 | `robust_v7.json['threshold']['common']['M19|0.50']['wound pairs']['est']` |
| b | common threshold 0.50 | 95% lower | 0.141159 | `robust_v7.json['threshold']['common']['M19|0.50']['wound pairs']['ci'][0]` |
| b | common threshold 0.50 | 95% upper | 0.646194 | `robust_v7.json['threshold']['common']['M19|0.50']['wound pairs']['ci'][1]` |
| b | common threshold 0.50: change from own threshold, 95% lower (filled marker if above 0) | value | 0.034729 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.50|wound pairs']['ci'][0]` |
| b | common threshold 0.55 | estimate | 0.442105 | `robust_v7.json['threshold']['common']['M19|0.55']['wound pairs']['est']` |
| b | common threshold 0.55 | 95% lower | 0.161333 | `robust_v7.json['threshold']['common']['M19|0.55']['wound pairs']['ci'][0]` |
| b | common threshold 0.55 | 95% upper | 0.666991 | `robust_v7.json['threshold']['common']['M19|0.55']['wound pairs']['ci'][1]` |
| b | common threshold 0.55: change from own threshold, 95% lower (filled marker if above 0) | value | 0.076036 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.55|wound pairs']['ci'][0]` |
| b | common threshold 0.60 | estimate | 0.472222 | `robust_v7.json['threshold']['common']['M19|0.60']['wound pairs']['est']` |
| b | common threshold 0.60 | 95% lower | 0.184615 | `robust_v7.json['threshold']['common']['M19|0.60']['wound pairs']['ci'][0]` |
| b | common threshold 0.60 | 95% upper | 0.694847 | `robust_v7.json['threshold']['common']['M19|0.60']['wound pairs']['ci'][1]` |
| b | common threshold 0.60: change from own threshold, 95% lower (filled marker if above 0) | value | 0.087819 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.60|wound pairs']['ci'][0]` |
| b | common threshold 0.65 | estimate | 0.495029 | `robust_v7.json['threshold']['common']['M19|0.65']['wound pairs']['est']` |
| b | common threshold 0.65 | 95% lower | 0.203820 | `robust_v7.json['threshold']['common']['M19|0.65']['wound pairs']['ci'][0]` |
| b | common threshold 0.65 | 95% upper | 0.707277 | `robust_v7.json['threshold']['common']['M19|0.65']['wound pairs']['ci'][1]` |
| b | common threshold 0.65: change from own threshold, 95% lower (filled marker if above 0) | value | 0.109440 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.65|wound pairs']['ci'][0]` |
| b | common threshold 0.70 | estimate | 0.528363 | `robust_v7.json['threshold']['common']['M19|0.70']['wound pairs']['est']` |
| b | common threshold 0.70 | 95% lower | 0.232180 | `robust_v7.json['threshold']['common']['M19|0.70']['wound pairs']['ci'][0]` |
| b | common threshold 0.70 | 95% upper | 0.741985 | `robust_v7.json['threshold']['common']['M19|0.70']['wound pairs']['ci'][1]` |
| b | common threshold 0.70: change from own threshold, 95% lower (filled marker if above 0) | value | 0.112746 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.70|wound pairs']['ci'][0]` |
| b | common threshold 0.75 | estimate | 0.566667 | `robust_v7.json['threshold']['common']['M19|0.75']['wound pairs']['est']` |
| b | common threshold 0.75 | 95% lower | 0.275836 | `robust_v7.json['threshold']['common']['M19|0.75']['wound pairs']['ci'][0]` |
| b | common threshold 0.75 | 95% upper | 0.767390 | `robust_v7.json['threshold']['common']['M19|0.75']['wound pairs']['ci'][1]` |
| b | common threshold 0.75: change from own threshold, 95% lower (filled marker if above 0) | value | 0.137739 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.75|wound pairs']['ci'][0]` |
| b | common threshold 0.80 | estimate | 0.569298 | `robust_v7.json['threshold']['common']['M19|0.80']['wound pairs']['est']` |
| b | common threshold 0.80 | 95% lower | 0.279162 | `robust_v7.json['threshold']['common']['M19|0.80']['wound pairs']['ci'][0]` |
| b | common threshold 0.80 | 95% upper | 0.762254 | `robust_v7.json['threshold']['common']['M19|0.80']['wound pairs']['ci'][1]` |
| b | common threshold 0.80: change from own threshold, 95% lower (filled marker if above 0) | value | 0.138851 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.80|wound pairs']['ci'][0]` |
| b | common threshold 0.85 | estimate | 0.620175 | `robust_v7.json['threshold']['common']['M19|0.85']['wound pairs']['est']` |
| b | common threshold 0.85 | 95% lower | 0.332928 | `robust_v7.json['threshold']['common']['M19|0.85']['wound pairs']['ci'][0]` |
| b | common threshold 0.85 | 95% upper | 0.809257 | `robust_v7.json['threshold']['common']['M19|0.85']['wound pairs']['ci'][1]` |
| b | common threshold 0.85: change from own threshold, 95% lower (filled marker if above 0) | value | 0.182375 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.85|wound pairs']['ci'][0]` |
| b | common threshold 0.90 | estimate | 0.621053 | `robust_v7.json['threshold']['common']['M19|0.90']['wound pairs']['est']` |
| b | common threshold 0.90 | 95% lower | 0.331560 | `robust_v7.json['threshold']['common']['M19|0.90']['wound pairs']['ci'][0]` |
| b | common threshold 0.90 | 95% upper | 0.811933 | `robust_v7.json['threshold']['common']['M19|0.90']['wound pairs']['ci'][1]` |
| b | common threshold 0.90: change from own threshold, 95% lower (filled marker if above 0) | value | 0.166774 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.90|wound pairs']['ci'][0]` |
| b | own threshold per model (default, line) | estimate | 0.211404 | `robust_v7.json['threshold']['default']['M19']['wound pairs']['est']` |
| b | own threshold per model (default, line) | 95% lower | -0.069542 | `robust_v7.json['threshold']['default']['M19']['wound pairs']['ci'][0]` |
| b | own threshold per model (default, line) | 95% upper | 0.501638 | `robust_v7.json['threshold']['default']['M19']['wound pairs']['ci'][1]` |
| b | per-dataset oracle threshold (line) | estimate | 0.122222 | `robust_v7.json['threshold']['oracle']['M19']['wound pairs']['est']` |
| b | per-dataset oracle threshold (line) | 95% lower | -0.129991 | `robust_v7.json['threshold']['oracle']['M19']['wound pairs']['ci'][0]` |
| b | per-dataset oracle threshold (line) | 95% upper | 0.395027 | `robust_v7.json['threshold']['oracle']['M19']['wound pairs']['ci'][1]` |

Total plotted values: 258.
