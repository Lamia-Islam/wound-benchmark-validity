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
- Fig1 values vs stats_v5: not applicable (exclusions_v6.json); the same extraction passed under exclusions_v5.json
- Fig1 number of V5-comparable cells: passed (0 cell checks: U wound (drawn in panels a and b, checked twice), F, U+F, U colonoscopy, K, U+K; max abs diff 0.0e+00)
- Fig2 U pairs6 equals mean|other: passed (5.6e-17)
- Fig2 U wound-to-wound equals stats_v7 method-only B2000: passed (2.8e-17)
- Fig2 F pairs6 equals mean|other: passed (0.0e+00)
- Fig2 F wound-to-wound equals stats_v7 method-only B2000: passed (1.1e-16)
- Fig2 K pairs6 equals mean|other: passed (1.1e-16)
- Fig2 K wound-to-wound equals stats_v7 method-only B2000: passed (5.6e-17)
- Fig2 M19 pairs6 equals mean|other: passed (2.8e-17)
- Fig2 M19 wound-to-wound equals stats_v7 method-only B2000: passed (5.6e-17)
- Fig2 U+F pairs6 equals mean|other: passed (1.1e-16)
- Fig2 U+F wound-to-wound equals stats_v7 method-only B2000: passed (1.1e-16)
- Fig2 U+K pairs6 equals mean|other: passed (5.6e-17)
- Fig2 U+K wound-to-wound equals stats_v7 method-only B2000: passed (1.1e-16)
- Fig2 U+M19 pairs6 equals mean|other: passed (1.1e-16)
- Fig2 U+M19 wound-to-wound equals stats_v7 method-only B2000: passed (1.1e-16)
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
| a | U (18), domain with domain | estimate | 0.350533 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x in']['est']` |
| a | U (18), domain with domain | 95% lower | 0.121948 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x in']['method_only_repro_B2000'][0]` |
| a | U (18), domain with domain | 95% upper | 0.593595 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x in']['method_only_repro_B2000'][1]` |
| a | U (18), domain with other | estimate | 0.086888 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x out']['est']` |
| a | U (18), domain with other | 95% lower | -0.155042 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x out']['method_only_repro_B2000'][0]` |
| a | U (18), domain with other | 95% upper | 0.294907 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x out']['method_only_repro_B2000'][1]` |
| a | U (18), other, same modality | estimate | 0.859236 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, same modality']['est']` |
| a | U (18), other, same modality | 95% lower | 0.675889 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, same modality']['method_only_repro_B2000'][0]` |
| a | U (18), other, same modality | 95% upper | 0.938358 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, same modality']['method_only_repro_B2000'][1]` |
| a | U (18), other, across modality | estimate | 0.035392 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, across modality']['est']` |
| a | U (18), other, across modality | 95% lower | -0.201283 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, across modality']['method_only_repro_B2000'][0]` |
| a | U (18), other, across modality | 95% upper | 0.287150 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, across modality']['method_only_repro_B2000'][1]` |
| a | M19 (19), domain with domain | estimate | 0.205848 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|in x in']['est']` |
| a | M19 (19), domain with domain | 95% lower | -0.070450 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|in x in']['method_only_repro_B2000'][0]` |
| a | M19 (19), domain with domain | 95% upper | 0.491296 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|in x in']['method_only_repro_B2000'][1]` |
| a | M19 (19), domain with other | estimate | 0.142669 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|in x out']['est']` |
| a | M19 (19), domain with other | 95% lower | -0.055228 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|in x out']['method_only_repro_B2000'][0]` |
| a | M19 (19), domain with other | 95% upper | 0.373366 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|in x out']['method_only_repro_B2000'][1]` |
| a | M19 (19), other, same modality | estimate | 0.402807 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|out, same modality']['est']` |
| a | M19 (19), other, same modality | 95% lower | 0.153555 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|out, same modality']['method_only_repro_B2000'][0]` |
| a | M19 (19), other, same modality | 95% upper | 0.629898 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|out, same modality']['method_only_repro_B2000'][1]` |
| a | M19 (19), other, across modality | estimate | 0.006539 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|out, across modality']['est']` |
| a | M19 (19), other, across modality | 95% lower | -0.255980 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|out, across modality']['method_only_repro_B2000'][0]` |
| a | M19 (19), other, across modality | 95% upper | 0.308801 | `stats_v7.json['bootstrap']['dice|M19']['comp|wound|out, across modality']['method_only_repro_B2000'][1]` |
| a | U+M19 (37), domain with domain | estimate | 0.818081 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|in x in']['est']` |
| a | U+M19 (37), domain with domain | 95% lower | 0.772800 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|in x in']['method_only_repro_B2000'][0]` |
| a | U+M19 (37), domain with domain | 95% upper | 0.866115 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|in x in']['method_only_repro_B2000'][1]` |
| a | U+M19 (37), domain with other | estimate | -0.561323 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|in x out']['est']` |
| a | U+M19 (37), domain with other | 95% lower | -0.664011 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|in x out']['method_only_repro_B2000'][0]` |
| a | U+M19 (37), domain with other | 95% upper | -0.446461 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|in x out']['method_only_repro_B2000'][1]` |
| a | U+M19 (37), other, same modality | estimate | 0.820175 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|out, same modality']['est']` |
| a | U+M19 (37), other, same modality | 95% lower | 0.713666 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|out, same modality']['method_only_repro_B2000'][0]` |
| a | U+M19 (37), other, same modality | 95% upper | 0.890419 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|out, same modality']['method_only_repro_B2000'][1]` |
| a | U+M19 (37), other, across modality | estimate | 0.431975 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|out, across modality']['est']` |
| a | U+M19 (37), other, across modality | 95% lower | 0.230255 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|out, across modality']['method_only_repro_B2000'][0]` |
| a | U+M19 (37), other, across modality | 95% upper | 0.624468 | `stats_v7.json['bootstrap']['dice|U+M19']['comp|wound|out, across modality']['method_only_repro_B2000'][1]` |
| b | U (18), domain with domain | estimate | 0.350533 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x in']['est']` |
| b | U (18), domain with domain | 95% lower | 0.121948 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x in']['method_only_repro_B2000'][0]` |
| b | U (18), domain with domain | 95% upper | 0.593595 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x in']['method_only_repro_B2000'][1]` |
| b | U (18), domain with other | estimate | 0.086888 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x out']['est']` |
| b | U (18), domain with other | 95% lower | -0.155042 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x out']['method_only_repro_B2000'][0]` |
| b | U (18), domain with other | 95% upper | 0.294907 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|in x out']['method_only_repro_B2000'][1]` |
| b | U (18), other, same modality | estimate | 0.859236 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, same modality']['est']` |
| b | U (18), other, same modality | 95% lower | 0.675889 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, same modality']['method_only_repro_B2000'][0]` |
| b | U (18), other, same modality | 95% upper | 0.938358 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, same modality']['method_only_repro_B2000'][1]` |
| b | U (18), other, across modality | estimate | 0.035392 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, across modality']['est']` |
| b | U (18), other, across modality | 95% lower | -0.201283 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, across modality']['method_only_repro_B2000'][0]` |
| b | U (18), other, across modality | 95% upper | 0.287150 | `stats_v7.json['bootstrap']['dice|U']['comp|wound|out, across modality']['method_only_repro_B2000'][1]` |
| b | F (14), domain with domain | estimate | 0.164103 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|in x in']['est']` |
| b | F (14), domain with domain | 95% lower | -0.152070 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|in x in']['method_only_repro_B2000'][0]` |
| b | F (14), domain with domain | 95% upper | 0.531181 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|in x in']['method_only_repro_B2000'][1]` |
| b | F (14), domain with other | estimate | 0.144874 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|in x out']['est']` |
| b | F (14), domain with other | 95% lower | -0.093824 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|in x out']['method_only_repro_B2000'][0]` |
| b | F (14), domain with other | 95% upper | 0.365805 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|in x out']['method_only_repro_B2000'][1]` |
| b | F (14), other, same modality | estimate | 0.879121 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|out, same modality']['est']` |
| b | F (14), other, same modality | 95% lower | 0.627442 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|out, same modality']['method_only_repro_B2000'][0]` |
| b | F (14), other, same modality | 95% upper | 0.964477 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|out, same modality']['method_only_repro_B2000'][1]` |
| b | F (14), other, across modality | estimate | 0.141728 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|out, across modality']['est']` |
| b | F (14), other, across modality | 95% lower | -0.053821 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|out, across modality']['method_only_repro_B2000'][0]` |
| b | F (14), other, across modality | 95% upper | 0.413323 | `stats_v7.json['bootstrap']['dice|F']['comp|wound|out, across modality']['method_only_repro_B2000'][1]` |
| b | U+F (32), domain with domain | estimate | 0.523338 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|in x in']['est']` |
| b | U+F (32), domain with domain | 95% lower | 0.294292 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|in x in']['method_only_repro_B2000'][0]` |
| b | U+F (32), domain with domain | 95% upper | 0.694042 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|in x in']['method_only_repro_B2000'][1]` |
| b | U+F (32), domain with other | estimate | -0.264560 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|in x out']['est']` |
| b | U+F (32), domain with other | 95% lower | -0.441400 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|in x out']['method_only_repro_B2000'][0]` |
| b | U+F (32), domain with other | 95% upper | -0.093024 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|in x out']['method_only_repro_B2000'][1]` |
| b | U+F (32), other, same modality | estimate | 0.899304 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|out, same modality']['est']` |
| b | U+F (32), other, same modality | 95% lower | 0.814255 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|out, same modality']['method_only_repro_B2000'][0]` |
| b | U+F (32), other, same modality | 95% upper | 0.935338 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|out, same modality']['method_only_repro_B2000'][1]` |
| b | U+F (32), other, across modality | estimate | 0.406322 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|out, across modality']['est']` |
| b | U+F (32), other, across modality | 95% lower | 0.188642 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|out, across modality']['method_only_repro_B2000'][0]` |
| b | U+F (32), other, across modality | 95% upper | 0.599707 | `stats_v7.json['bootstrap']['dice|U+F']['comp|wound|out, across modality']['method_only_repro_B2000'][1]` |
| c | U (18), domain with domain | estimate | 0.859236 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|in x in']['est']` |
| c | U (18), domain with domain | 95% lower | 0.675889 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|in x in']['method_only_repro_B2000'][0]` |
| c | U (18), domain with domain | 95% upper | 0.938358 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|in x in']['method_only_repro_B2000'][1]` |
| c | U (18), domain with other | estimate | 0.125315 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|in x out']['est']` |
| c | U (18), domain with other | 95% lower | -0.109834 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|in x out']['method_only_repro_B2000'][0]` |
| c | U (18), domain with other | 95% upper | 0.311395 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|in x out']['method_only_repro_B2000'][1]` |
| c | U (18), other, same modality | estimate | 0.350533 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|out, same modality']['est']` |
| c | U (18), other, same modality | 95% lower | 0.121948 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|out, same modality']['method_only_repro_B2000'][0]` |
| c | U (18), other, same modality | 95% upper | 0.593595 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|out, same modality']['method_only_repro_B2000'][1]` |
| c | U (18), other, across modality | estimate | -0.104141 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|out, across modality']['est']` |
| c | U (18), other, across modality | 95% lower | -0.250411 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|out, across modality']['method_only_repro_B2000'][0]` |
| c | U (18), other, across modality | 95% upper | 0.092821 | `stats_v7.json['bootstrap']['dice|U']['comp|colonoscopy|out, across modality']['method_only_repro_B2000'][1]` |
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
| c | U+K (32), domain with other | estimate | 0.024045 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|in x out']['est']` |
| c | U+K (32), domain with other | 95% lower | -0.076645 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|in x out']['method_only_repro_B2000'][0]` |
| c | U+K (32), domain with other | 95% upper | 0.118030 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|in x out']['method_only_repro_B2000'][1]` |
| c | U+K (32), other, same modality | estimate | 0.270161 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|out, same modality']['est']` |
| c | U+K (32), other, same modality | 95% lower | 0.104664 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|out, same modality']['method_only_repro_B2000'][0]` |
| c | U+K (32), other, same modality | 95% upper | 0.431861 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|out, same modality']['method_only_repro_B2000'][1]` |
| c | U+K (32), other, across modality | estimate | -0.078653 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|out, across modality']['est']` |
| c | U+K (32), other, across modality | 95% lower | -0.233044 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|out, across modality']['method_only_repro_B2000'][0]` |
| c | U+K (32), other, across modality | 95% upper | 0.102078 | `stats_v7.json['bootstrap']['dice|U+K']['comp|colonoscopy|out, across modality']['method_only_repro_B2000'][1]` |

## Figure 2

| panel | row / series | quantity | value | JSON path |
|---|---|---|---|---|
| - | 18 untrained [U]: benchmark with wound data | estimate | 0.525800 | `bench_v7.json['rel']['U']['40']['mean|bench']['est']` |
| - | 18 untrained [U]: benchmark with wound data | 95% lower | 0.216524 | `bench_v7.json['rel']['U']['40']['mean|bench']['lo']` |
| - | 18 untrained [U]: benchmark with wound data | 95% upper | 0.740470 | `bench_v7.json['rel']['U']['40']['mean|bench']['hi']` |
| - | 18 untrained [U]: wound data with each other | estimate | 0.350533 | `bench_v7.json['rel']['U']['40']['pairs6']['est']` |
| - | 18 untrained [U]: wound data with each other | 95% lower | 0.121948 | `bench_v7.json['rel']['U']['40']['pairs6']['lo']` |
| - | 18 untrained [U]: wound data with each other | 95% upper | 0.593595 | `bench_v7.json['rel']['U']['40']['pairs6']['hi']` |
| - | 14 from scratch [F]: benchmark with wound data | estimate | 0.228938 | `bench_v7.json['rel']['F']['40']['mean|bench']['est']` |
| - | 14 from scratch [F]: benchmark with wound data | 95% lower | -0.038119 | `bench_v7.json['rel']['F']['40']['mean|bench']['lo']` |
| - | 14 from scratch [F]: benchmark with wound data | 95% upper | 0.449783 | `bench_v7.json['rel']['F']['40']['mean|bench']['hi']` |
| - | 14 from scratch [F]: wound data with each other | estimate | 0.164103 | `bench_v7.json['rel']['F']['40']['pairs6']['est']` |
| - | 14 from scratch [F]: wound data with each other | 95% lower | -0.152070 | `bench_v7.json['rel']['F']['40']['pairs6']['lo']` |
| - | 14 from scratch [F]: wound data with each other | 95% upper | 0.531181 | `bench_v7.json['rel']['F']['40']['pairs6']['hi']` |
| - | 14 Kvasir-trained [K]: benchmark with wound data | estimate | 0.112454 | `bench_v7.json['rel']['K']['40']['mean|bench']['est']` |
| - | 14 Kvasir-trained [K]: benchmark with wound data | 95% lower | -0.350375 | `bench_v7.json['rel']['K']['40']['mean|bench']['lo']` |
| - | 14 Kvasir-trained [K]: benchmark with wound data | 95% upper | 0.536076 | `bench_v7.json['rel']['K']['40']['mean|bench']['hi']` |
| - | 14 Kvasir-trained [K]: wound data with each other | estimate | 0.362637 | `bench_v7.json['rel']['K']['40']['pairs6']['est']` |
| - | 14 Kvasir-trained [K]: wound data with each other | 95% lower | 0.047689 | `bench_v7.json['rel']['K']['40']['pairs6']['lo']` |
| - | 14 Kvasir-trained [K]: wound data with each other | 95% upper | 0.635609 | `bench_v7.json['rel']['K']['40']['pairs6']['hi']` |
| - | 19 modern [M19]: benchmark with wound data | estimate | 0.205263 | `bench_v7.json['rel']['M19']['40']['mean|bench']['est']` |
| - | 19 modern [M19]: benchmark with wound data | 95% lower | -0.167274 | `bench_v7.json['rel']['M19']['40']['mean|bench']['lo']` |
| - | 19 modern [M19]: benchmark with wound data | 95% upper | 0.484983 | `bench_v7.json['rel']['M19']['40']['mean|bench']['hi']` |
| - | 19 modern [M19]: wound data with each other | estimate | 0.205848 | `bench_v7.json['rel']['M19']['40']['pairs6']['est']` |
| - | 19 modern [M19]: wound data with each other | 95% lower | -0.070450 | `bench_v7.json['rel']['M19']['40']['pairs6']['lo']` |
| - | 19 modern [M19]: wound data with each other | 95% upper | 0.491296 | `bench_v7.json['rel']['M19']['40']['pairs6']['hi']` |
| - | 32 pooled: untrained + from scratch [U+F]: benchmark with wound data | estimate | 0.345674 | `bench_v7.json['rel']['U+F']['40']['mean|bench']['est']` |
| - | 32 pooled: untrained + from scratch [U+F]: benchmark with wound data | 95% lower | 0.071399 | `bench_v7.json['rel']['U+F']['40']['mean|bench']['lo']` |
| - | 32 pooled: untrained + from scratch [U+F]: benchmark with wound data | 95% upper | 0.572817 | `bench_v7.json['rel']['U+F']['40']['mean|bench']['hi']` |
| - | 32 pooled: untrained + from scratch [U+F]: wound data with each other | estimate | 0.523338 | `bench_v7.json['rel']['U+F']['40']['pairs6']['est']` |
| - | 32 pooled: untrained + from scratch [U+F]: wound data with each other | 95% lower | 0.294292 | `bench_v7.json['rel']['U+F']['40']['pairs6']['lo']` |
| - | 32 pooled: untrained + from scratch [U+F]: wound data with each other | 95% upper | 0.694042 | `bench_v7.json['rel']['U+F']['40']['pairs6']['hi']` |
| - | 32 pooled: untrained + Kvasir-trained [U+K]: benchmark with wound data | estimate | 0.319343 | `bench_v7.json['rel']['U+K']['40']['mean|bench']['est']` |
| - | 32 pooled: untrained + Kvasir-trained [U+K]: benchmark with wound data | 95% lower | 0.062502 | `bench_v7.json['rel']['U+K']['40']['mean|bench']['lo']` |
| - | 32 pooled: untrained + Kvasir-trained [U+K]: benchmark with wound data | 95% upper | 0.540275 | `bench_v7.json['rel']['U+K']['40']['mean|bench']['hi']` |
| - | 32 pooled: untrained + Kvasir-trained [U+K]: wound data with each other | estimate | 0.270161 | `bench_v7.json['rel']['U+K']['40']['pairs6']['est']` |
| - | 32 pooled: untrained + Kvasir-trained [U+K]: wound data with each other | 95% lower | 0.104664 | `bench_v7.json['rel']['U+K']['40']['pairs6']['lo']` |
| - | 32 pooled: untrained + Kvasir-trained [U+K]: wound data with each other | 95% upper | 0.431861 | `bench_v7.json['rel']['U+K']['40']['pairs6']['hi']` |
| - | 37 pooled: untrained + modern [U+M19]: benchmark with wound data | estimate | 0.646357 | `bench_v7.json['rel']['U+M19']['40']['mean|bench']['est']` |
| - | 37 pooled: untrained + modern [U+M19]: benchmark with wound data | 95% lower | 0.438047 | `bench_v7.json['rel']['U+M19']['40']['mean|bench']['lo']` |
| - | 37 pooled: untrained + modern [U+M19]: benchmark with wound data | 95% upper | 0.804443 | `bench_v7.json['rel']['U+M19']['40']['mean|bench']['hi']` |
| - | 37 pooled: untrained + modern [U+M19]: wound data with each other | estimate | 0.818081 | `bench_v7.json['rel']['U+M19']['40']['pairs6']['est']` |
| - | 37 pooled: untrained + modern [U+M19]: wound data with each other | 95% lower | 0.772800 | `bench_v7.json['rel']['U+M19']['40']['pairs6']['lo']` |
| - | 37 pooled: untrained + modern [U+M19]: wound data with each other | 95% upper | 0.866115 | `bench_v7.json['rel']['U+M19']['40']['pairs6']['hi']` |

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
| a | same dataset, different seed: AZH | estimate | 0.574510 | `stats_v7.json['seeds']['dice|C16']['per_dataset']['azh']['rbar']` |
| a | same dataset, different seed: AZH | 95% lower | 0.185725 | `stats_v7.json['seeds']['dice|C16']['per_dataset']['azh']['rbar_method_boot'][0]` |
| a | same dataset, different seed: AZH | 95% upper | 0.832603 | `stats_v7.json['seeds']['dice|C16']['per_dataset']['azh']['rbar_method_boot'][1]` |
| a | same dataset, different seed: mean of 4 datasets | estimate | 0.440196 | `stats_v7.json['seeds']['dice|C16']['summary']['rbar_wound']['est']` |
| a | same dataset, different seed: mean of 4 datasets | 95% lower | 0.173597 | `stats_v7.json['seeds']['dice|C16']['summary']['rbar_wound']['method_boot'][0]` |
| a | same dataset, different seed: mean of 4 datasets | 95% upper | 0.642858 | `stats_v7.json['seeds']['dice|C16']['summary']['rbar_wound']['method_boot'][1]` |
| a | different dataset, same seed: Medetec vs CO2Wounds-V2 (mean over 3 seeds, no interval) | estimate | 0.521569 | `stats_v7.json['seeds']['disattenuation']['dice|C16']['wound_pairs']['medetec~co2wounds']['observed_single_mean']` |
| a | different dataset, same seed: Medetec vs FUSeg (mean over 3 seeds, no interval) | estimate | 0.000000 | `stats_v7.json['seeds']['disattenuation']['dice|C16']['wound_pairs']['medetec~fuseg']['observed_single_mean']` |
| a | different dataset, same seed: Medetec vs AZH (mean over 3 seeds, no interval) | estimate | 0.201961 | `stats_v7.json['seeds']['disattenuation']['dice|C16']['wound_pairs']['medetec~azh']['observed_single_mean']` |
| a | different dataset, same seed: CO2Wounds-V2 vs FUSeg (mean over 3 seeds, no interval) | estimate | -0.037255 | `stats_v7.json['seeds']['disattenuation']['dice|C16']['wound_pairs']['co2wounds~fuseg']['observed_single_mean']` |
| a | different dataset, same seed: CO2Wounds-V2 vs AZH (mean over 3 seeds, no interval) | estimate | 0.287255 | `stats_v7.json['seeds']['disattenuation']['dice|C16']['wound_pairs']['co2wounds~azh']['observed_single_mean']` |
| a | different dataset, same seed: FUSeg vs AZH (mean over 3 seeds, no interval) | estimate | 0.058824 | `stats_v7.json['seeds']['disattenuation']['dice|C16']['wound_pairs']['fuseg~azh']['observed_single_mean']` |
| a | different dataset, same seed: mean of 6 pairs | estimate | 0.172059 | `stats_v7.json['seeds']['dice|C16']['summary']['single|wound pairs']['est']` |
| a | different dataset, same seed: mean of 6 pairs | 95% lower | -0.048844 | `stats_v7.json['seeds']['dice|C16']['summary']['single|wound pairs']['method_boot'][0]` |
| a | different dataset, same seed: mean of 6 pairs | 95% upper | 0.396620 | `stats_v7.json['seeds']['dice|C16']['summary']['single|wound pairs']['method_boot'][1]` |
| a | different dataset, different seed: mean of 6 pairs | estimate | 0.089706 | `stats_v7.json['seeds']['dice|C16']['summary']['crossseed|wound pairs']['est']` |
| a | different dataset, different seed: mean of 6 pairs | 95% lower | -0.122449 | `stats_v7.json['seeds']['dice|C16']['summary']['crossseed|wound pairs']['method_boot'][0]` |
| a | different dataset, different seed: mean of 6 pairs | 95% upper | 0.324722 | `stats_v7.json['seeds']['dice|C16']['summary']['crossseed|wound pairs']['method_boot'][1]` |
| a | control: Control images, different seed | estimate | 0.839216 | `stats_v7.json['seeds']['dice|C16']['summary']['rbar_control']['est']` |
| a | control: Control images, different seed | 95% lower | 0.560155 | `stats_v7.json['seeds']['dice|C16']['summary']['rbar_control']['method_boot'][0]` |
| a | control: Control images, different seed | 95% upper | 0.943003 | `stats_v7.json['seeds']['dice|C16']['summary']['rbar_control']['method_boot'][1]` |
| b | common threshold 0.05 | estimate | 0.360819 | `robust_v7.json['threshold']['common']['M19|0.05']['wound pairs']['est']` |
| b | common threshold 0.05 | 95% lower | 0.039633 | `robust_v7.json['threshold']['common']['M19|0.05']['wound pairs']['ci'][0]` |
| b | common threshold 0.05 | 95% upper | 0.635767 | `robust_v7.json['threshold']['common']['M19|0.05']['wound pairs']['ci'][1]` |
| b | common threshold 0.05: change from own threshold, 95% lower (filled marker if above 0) | value | -0.177569 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.05|wound pairs']['ci'][0]` |
| b | common threshold 0.10 | estimate | 0.204386 | `robust_v7.json['threshold']['common']['M19|0.10']['wound pairs']['est']` |
| b | common threshold 0.10 | 95% lower | -0.077553 | `robust_v7.json['threshold']['common']['M19|0.10']['wound pairs']['ci'][0]` |
| b | common threshold 0.10 | 95% upper | 0.496166 | `robust_v7.json['threshold']['common']['M19|0.10']['wound pairs']['ci'][1]` |
| b | common threshold 0.10: change from own threshold, 95% lower (filled marker if above 0) | value | -0.260434 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.10|wound pairs']['ci'][0]` |
| b | common threshold 0.15 | estimate | 0.212573 | `robust_v7.json['threshold']['common']['M19|0.15']['wound pairs']['est']` |
| b | common threshold 0.15 | 95% lower | -0.060475 | `robust_v7.json['threshold']['common']['M19|0.15']['wound pairs']['ci'][0]` |
| b | common threshold 0.15 | 95% upper | 0.511665 | `robust_v7.json['threshold']['common']['M19|0.15']['wound pairs']['ci'][1]` |
| b | common threshold 0.15: change from own threshold, 95% lower (filled marker if above 0) | value | -0.208594 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.15|wound pairs']['ci'][0]` |
| b | common threshold 0.20 | estimate | 0.251170 | `robust_v7.json['threshold']['common']['M19|0.20']['wound pairs']['est']` |
| b | common threshold 0.20 | 95% lower | -0.046737 | `robust_v7.json['threshold']['common']['M19|0.20']['wound pairs']['ci'][0]` |
| b | common threshold 0.20 | 95% upper | 0.542614 | `robust_v7.json['threshold']['common']['M19|0.20']['wound pairs']['ci'][1]` |
| b | common threshold 0.20: change from own threshold, 95% lower (filled marker if above 0) | value | -0.147974 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.20|wound pairs']['ci'][0]` |
| b | common threshold 0.25 | estimate | 0.272515 | `robust_v7.json['threshold']['common']['M19|0.25']['wound pairs']['est']` |
| b | common threshold 0.25 | 95% lower | -0.017214 | `robust_v7.json['threshold']['common']['M19|0.25']['wound pairs']['ci'][0]` |
| b | common threshold 0.25 | 95% upper | 0.565177 | `robust_v7.json['threshold']['common']['M19|0.25']['wound pairs']['ci'][1]` |
| b | common threshold 0.25: change from own threshold, 95% lower (filled marker if above 0) | value | -0.117289 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.25|wound pairs']['ci'][0]` |
| b | common threshold 0.30 | estimate | 0.281871 | `robust_v7.json['threshold']['common']['M19|0.30']['wound pairs']['est']` |
| b | common threshold 0.30 | 95% lower | -0.002674 | `robust_v7.json['threshold']['common']['M19|0.30']['wound pairs']['ci'][0]` |
| b | common threshold 0.30 | 95% upper | 0.563703 | `robust_v7.json['threshold']['common']['M19|0.30']['wound pairs']['ci'][1]` |
| b | common threshold 0.30: change from own threshold, 95% lower (filled marker if above 0) | value | -0.098238 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.30|wound pairs']['ci'][0]` |
| b | common threshold 0.35 | estimate | 0.267836 | `robust_v7.json['threshold']['common']['M19|0.35']['wound pairs']['est']` |
| b | common threshold 0.35 | 95% lower | -0.012731 | `robust_v7.json['threshold']['common']['M19|0.35']['wound pairs']['ci'][0]` |
| b | common threshold 0.35 | 95% upper | 0.541006 | `robust_v7.json['threshold']['common']['M19|0.35']['wound pairs']['ci'][1]` |
| b | common threshold 0.35: change from own threshold, 95% lower (filled marker if above 0) | value | -0.111603 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.35|wound pairs']['ci'][0]` |
| b | common threshold 0.40 | estimate | 0.312281 | `robust_v7.json['threshold']['common']['M19|0.40']['wound pairs']['est']` |
| b | common threshold 0.40 | 95% lower | 0.034733 | `robust_v7.json['threshold']['common']['M19|0.40']['wound pairs']['ci'][0]` |
| b | common threshold 0.40 | 95% upper | 0.575295 | `robust_v7.json['threshold']['common']['M19|0.40']['wound pairs']['ci'][1]` |
| b | common threshold 0.40: change from own threshold, 95% lower (filled marker if above 0) | value | -0.065419 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.40|wound pairs']['ci'][0]` |
| b | common threshold 0.45 | estimate | 0.369298 | `robust_v7.json['threshold']['common']['M19|0.45']['wound pairs']['est']` |
| b | common threshold 0.45 | 95% lower | 0.099196 | `robust_v7.json['threshold']['common']['M19|0.45']['wound pairs']['ci'][0]` |
| b | common threshold 0.45 | 95% upper | 0.607178 | `robust_v7.json['threshold']['common']['M19|0.45']['wound pairs']['ci'][1]` |
| b | common threshold 0.45: change from own threshold, 95% lower (filled marker if above 0) | value | 0.000280 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.45|wound pairs']['ci'][0]` |
| b | common threshold 0.50 | estimate | 0.404094 | `robust_v7.json['threshold']['common']['M19|0.50']['wound pairs']['est']` |
| b | common threshold 0.50 | 95% lower | 0.139086 | `robust_v7.json['threshold']['common']['M19|0.50']['wound pairs']['ci'][0]` |
| b | common threshold 0.50 | 95% upper | 0.626726 | `robust_v7.json['threshold']['common']['M19|0.50']['wound pairs']['ci'][1]` |
| b | common threshold 0.50: change from own threshold, 95% lower (filled marker if above 0) | value | 0.033002 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.50|wound pairs']['ci'][0]` |
| b | common threshold 0.55 | estimate | 0.438596 | `robust_v7.json['threshold']['common']['M19|0.55']['wound pairs']['est']` |
| b | common threshold 0.55 | 95% lower | 0.156447 | `robust_v7.json['threshold']['common']['M19|0.55']['wound pairs']['ci'][0]` |
| b | common threshold 0.55 | 95% upper | 0.666667 | `robust_v7.json['threshold']['common']['M19|0.55']['wound pairs']['ci'][1]` |
| b | common threshold 0.55: change from own threshold, 95% lower (filled marker if above 0) | value | 0.078981 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.55|wound pairs']['ci'][0]` |
| b | common threshold 0.60 | estimate | 0.463158 | `robust_v7.json['threshold']['common']['M19|0.60']['wound pairs']['est']` |
| b | common threshold 0.60 | 95% lower | 0.183529 | `robust_v7.json['threshold']['common']['M19|0.60']['wound pairs']['ci'][0]` |
| b | common threshold 0.60 | 95% upper | 0.687316 | `robust_v7.json['threshold']['common']['M19|0.60']['wound pairs']['ci'][1]` |
| b | common threshold 0.60: change from own threshold, 95% lower (filled marker if above 0) | value | 0.087103 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.60|wound pairs']['ci'][0]` |
| b | common threshold 0.65 | estimate | 0.495029 | `robust_v7.json['threshold']['common']['M19|0.65']['wound pairs']['est']` |
| b | common threshold 0.65 | 95% lower | 0.203820 | `robust_v7.json['threshold']['common']['M19|0.65']['wound pairs']['ci'][0]` |
| b | common threshold 0.65 | 95% upper | 0.707277 | `robust_v7.json['threshold']['common']['M19|0.65']['wound pairs']['ci'][1]` |
| b | common threshold 0.65: change from own threshold, 95% lower (filled marker if above 0) | value | 0.112568 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.65|wound pairs']['ci'][0]` |
| b | common threshold 0.70 | estimate | 0.521053 | `robust_v7.json['threshold']['common']['M19|0.70']['wound pairs']['est']` |
| b | common threshold 0.70 | 95% lower | 0.228042 | `robust_v7.json['threshold']['common']['M19|0.70']['wound pairs']['ci'][0]` |
| b | common threshold 0.70 | 95% upper | 0.730056 | `robust_v7.json['threshold']['common']['M19|0.70']['wound pairs']['ci'][1]` |
| b | common threshold 0.70: change from own threshold, 95% lower (filled marker if above 0) | value | 0.116301 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.70|wound pairs']['ci'][0]` |
| b | common threshold 0.75 | estimate | 0.565497 | `robust_v7.json['threshold']['common']['M19|0.75']['wound pairs']['est']` |
| b | common threshold 0.75 | 95% lower | 0.265871 | `robust_v7.json['threshold']['common']['M19|0.75']['wound pairs']['ci'][0]` |
| b | common threshold 0.75 | 95% upper | 0.761361 | `robust_v7.json['threshold']['common']['M19|0.75']['wound pairs']['ci'][1]` |
| b | common threshold 0.75: change from own threshold, 95% lower (filled marker if above 0) | value | 0.141296 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.75|wound pairs']['ci'][0]` |
| b | common threshold 0.80 | estimate | 0.569298 | `robust_v7.json['threshold']['common']['M19|0.80']['wound pairs']['est']` |
| b | common threshold 0.80 | 95% lower | 0.279162 | `robust_v7.json['threshold']['common']['M19|0.80']['wound pairs']['ci'][0]` |
| b | common threshold 0.80 | 95% upper | 0.762254 | `robust_v7.json['threshold']['common']['M19|0.80']['wound pairs']['ci'][1]` |
| b | common threshold 0.80: change from own threshold, 95% lower (filled marker if above 0) | value | 0.148906 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.80|wound pairs']['ci'][0]` |
| b | common threshold 0.85 | estimate | 0.613743 | `robust_v7.json['threshold']['common']['M19|0.85']['wound pairs']['est']` |
| b | common threshold 0.85 | 95% lower | 0.328272 | `robust_v7.json['threshold']['common']['M19|0.85']['wound pairs']['ci'][0]` |
| b | common threshold 0.85 | 95% upper | 0.805139 | `robust_v7.json['threshold']['common']['M19|0.85']['wound pairs']['ci'][1]` |
| b | common threshold 0.85: change from own threshold, 95% lower (filled marker if above 0) | value | 0.188098 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.85|wound pairs']['ci'][0]` |
| b | common threshold 0.90 | estimate | 0.624854 | `robust_v7.json['threshold']['common']['M19|0.90']['wound pairs']['est']` |
| b | common threshold 0.90 | 95% lower | 0.332677 | `robust_v7.json['threshold']['common']['M19|0.90']['wound pairs']['ci'][0]` |
| b | common threshold 0.90 | 95% upper | 0.815025 | `robust_v7.json['threshold']['common']['M19|0.90']['wound pairs']['ci'][1]` |
| b | common threshold 0.90: change from own threshold, 95% lower (filled marker if above 0) | value | 0.169909 | `robust_v7.json['threshold']['delta_vs_default']['M19|0.90|wound pairs']['ci'][0]` |
| b | own threshold per model (default, line) | estimate | 0.205848 | `robust_v7.json['threshold']['default']['M19']['wound pairs']['est']` |
| b | own threshold per model (default, line) | 95% lower | -0.070450 | `robust_v7.json['threshold']['default']['M19']['wound pairs']['ci'][0]` |
| b | own threshold per model (default, line) | 95% upper | 0.491296 | `robust_v7.json['threshold']['default']['M19']['wound pairs']['ci'][1]` |
| b | per-dataset oracle threshold (line) | estimate | 0.115497 | `robust_v7.json['threshold']['oracle']['M19']['wound pairs']['est']` |
| b | per-dataset oracle threshold (line) | 95% lower | -0.136408 | `robust_v7.json['threshold']['oracle']['M19']['wound pairs']['ci'][0]` |
| b | per-dataset oracle threshold (line) | 95% upper | 0.391328 | `robust_v7.json['threshold']['oracle']['M19']['wound pairs']['ci'][1]` |

Total plotted values: 258.
