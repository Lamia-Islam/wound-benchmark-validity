# Synthetic benchmark reanalysis (V6)

Script: `scripts/v6/bench_v6.py`. Numbers: `results/v6/bench_v6.json`.

## Definitions

- Agreement: Spearman correlation (average ranks) of per-method mean Dice between two sets, over a method set.
- Method sets: U (18), F (14), K (14), M (11), U+F (32), U+K (32), U+M (29). S (SAM, MedSAM) is never pooled.
- bench(R): agreement of generator B at the given amplitude with wound reference R, averaged over seeds 2024, 7, 101.
- other(R): mean agreement of the other three wound datasets with R. Averaged over the 4 references it equals the mean of the 6 wound pairs.
- diff(R) = bench(R) - other(R). The 'mean' row averages over Medetec, CO2Wounds-V2, FUSeg, AZH.
- Method bootstrap: methods resampled with replacement within groups, 2,000 draws, percentile 95% intervals, RNG as in stats_v5 (draw for draw).
- Hierarchical bootstrap (M and K only): methods within group plus, independently per dataset, clinical units (Medetec photograph, AZH photograph, otherwise image) and synthetic images; 2,000 draws.
- Data-only bootstrap: units and images resampled, methods fixed (context only).

## 1. Relative agreement, generator B amplitude 40

Method bootstrap intervals. Each cell: estimate [95% interval].

| Set | Ref | bench | other | diff |
|---|---|---|---|---|
| U | medetec | +0.591 [+0.057, +0.924] | +0.586 [+0.407, +0.743] | +0.006 [-0.452, +0.319] |
| U | co2wounds | +0.805 [+0.557, +0.908] | +0.280 [-0.175, +0.646] | +0.524 [+0.125, +0.921] |
| U | fuseg | +0.390 [-0.139, +0.805] | +0.367 [+0.003, +0.627] | +0.023 [-0.431, +0.404] |
| U | azh | +0.315 [-0.227, +0.734] | +0.232 [-0.183, +0.601] | +0.083 [-0.257, +0.355] |
| U | mean | +0.525 [+0.205, +0.743] | +0.366 [+0.139, +0.608] | +0.159 [-0.104, +0.314] |
| F | medetec | +0.046 [-0.485, +0.575] | +0.245 [-0.148, +0.631] | -0.199 [-0.701, +0.358] |
| F | co2wounds | +0.468 [-0.021, +0.818] | +0.168 [-0.250, +0.576] | +0.300 [-0.399, +0.946] |
| F | fuseg | +0.097 [-0.639, +0.753] | -0.045 [-0.513, +0.449] | +0.142 [-0.536, +0.767] |
| F | azh | +0.278 [-0.264, +0.676] | +0.282 [+0.000, +0.574] | -0.004 [-0.604, +0.485] |
| F | mean | +0.222 [-0.038, +0.445] | +0.163 [-0.154, +0.531] | +0.060 [-0.422, +0.409] |
| K | medetec | +0.165 [-0.433, +0.645] | +0.473 [+0.157, +0.699] | -0.308 [-0.776, +0.137] |
| K | co2wounds | +0.064 [-0.641, +0.660] | +0.519 [+0.211, +0.741] | -0.456 [-1.037, +0.120] |
| K | fuseg | -0.338 [-0.835, +0.272] | +0.396 [+0.003, +0.688] | -0.734 [-1.141, -0.282] |
| K | azh | +0.559 [-0.028, +0.882] | +0.062 [-0.487, +0.528] | +0.497 [-0.371, +1.212] |
| K | mean | +0.112 [-0.350, +0.536] | +0.363 [+0.048, +0.636] | -0.250 [-0.649, +0.082] |
| M | medetec | +0.297 [-0.450, +0.846] | +0.218 [-0.267, +0.598] | +0.079 [-0.562, +0.533] |
| M | co2wounds | +0.445 [-0.197, +0.759] | +0.088 [-0.333, +0.485] | +0.358 [-0.178, +0.796] |
| M | fuseg | +0.279 [-0.404, +0.775] | +0.048 [-0.441, +0.441] | +0.230 [-0.319, +0.670] |
| M | azh | +0.391 [-0.383, +0.859] | +0.148 [-0.222, +0.478] | +0.242 [-0.560, +0.756] |
| M | mean | +0.353 [-0.068, +0.631] | +0.126 [-0.116, +0.414] | +0.227 [-0.186, +0.417] |
| U+F | medetec | +0.343 [-0.040, +0.656] | +0.509 [+0.245, +0.710] | -0.166 [-0.566, +0.237] |
| U+F | co2wounds | +0.625 [+0.386, +0.780] | +0.496 [+0.225, +0.708] | +0.129 [-0.186, +0.414] |
| U+F | fuseg | +0.206 [-0.146, +0.515] | +0.547 [+0.312, +0.721] | -0.341 [-0.688, -0.034] |
| U+F | azh | +0.200 [-0.213, +0.547] | +0.561 [+0.330, +0.742] | -0.361 [-0.666, -0.098] |
| U+F | mean | +0.344 [+0.069, +0.571] | +0.528 [+0.301, +0.698] | -0.184 [-0.435, +0.036] |
| U+K | medetec | +0.444 [+0.061, +0.724] | +0.501 [+0.362, +0.611] | -0.057 [-0.352, +0.214] |
| U+K | co2wounds | +0.164 [-0.234, +0.492] | +0.291 [-0.002, +0.526] | -0.126 [-0.464, +0.184] |
| U+K | fuseg | -0.026 [-0.434, +0.341] | +0.341 [+0.139, +0.493] | -0.367 [-0.664, -0.090] |
| U+K | azh | +0.695 [+0.489, +0.851] | -0.030 [-0.308, +0.244] | +0.725 [+0.456, +0.996] |
| U+K | mean | +0.319 [+0.062, +0.540] | +0.276 [+0.108, +0.436] | +0.044 [-0.121, +0.173] |
| U+M | medetec | +0.751 [+0.569, +0.890] | +0.859 [+0.809, +0.903] | -0.107 [-0.272, +0.018] |
| U+M | co2wounds | +0.812 [+0.630, +0.906] | +0.779 [+0.673, +0.873] | +0.033 [-0.167, +0.181] |
| U+M | fuseg | +0.657 [+0.372, +0.859] | +0.797 [+0.710, +0.870] | -0.140 [-0.395, +0.043] |
| U+M | azh | +0.662 [+0.405, +0.838] | +0.771 [+0.666, +0.858] | -0.108 [-0.307, +0.022] |
| U+M | mean | +0.721 [+0.525, +0.846] | +0.801 [+0.745, +0.863] | -0.081 [-0.260, +0.034] |

Hierarchical (methods + units + synthetic images) and data-only intervals, M and K:

| Set | Ref | bench (hier) | other (hier) | diff (hier) | diff (data-only) |
|---|---|---|---|---|---|
| M | medetec | +0.297 [-0.507, +0.821] | +0.218 [-0.283, +0.583] | +0.079 [-0.562, +0.595] | +0.079 [-0.212, +0.279] |
| M | co2wounds | +0.445 [-0.395, +0.782] | +0.088 [-0.386, +0.505] | +0.358 [-0.366, +0.793] | +0.358 [-0.024, +0.503] |
| M | fuseg | +0.279 [-0.502, +0.778] | +0.048 [-0.408, +0.508] | +0.230 [-0.484, +0.763] | +0.230 [-0.073, +0.482] |
| M | azh | +0.391 [-0.453, +0.824] | +0.148 [-0.364, +0.487] | +0.242 [-0.511, +0.821] | +0.242 [-0.042, +0.445] |
| M | mean | +0.353 [-0.106, +0.578] | +0.126 [-0.191, +0.442] | +0.227 [-0.235, +0.411] | +0.227 [+0.040, +0.294] |
| K | medetec | +0.165 [-0.481, +0.741] | +0.473 [+0.018, +0.721] | -0.308 [-0.782, +0.356] | -0.308 [-0.463, +0.054] |
| K | co2wounds | +0.064 [-0.628, +0.679] | +0.519 [+0.129, +0.738] | -0.456 [-0.951, +0.174] | -0.456 [-0.539, -0.228] |
| K | fuseg | -0.338 [-0.814, +0.453] | +0.396 [-0.125, +0.707] | -0.734 [-1.028, -0.081] | -0.734 [-0.747, -0.399] |
| K | azh | +0.559 [-0.047, +0.900] | +0.062 [-0.486, +0.561] | +0.497 [-0.241, +1.179] | +0.497 [+0.296, +0.734] |
| K | mean | +0.112 [-0.315, +0.584] | +0.363 [-0.032, +0.656] | -0.250 [-0.577, +0.161] | -0.250 [-0.299, -0.038] |

Verdict per method set (mean over the 4 references):

- U: diff +0.159 [-0.104, +0.314], not distinguishable from a clinical wound dataset as a proxy for the others (method bootstrap). Per-reference intervals excluding zero: co2wounds.
- F: diff +0.060 [-0.422, +0.409], not distinguishable from a clinical wound dataset as a proxy for the others (method bootstrap). Per-reference intervals excluding zero: none.
- K: diff -0.250 [-0.649, +0.082], not distinguishable from a clinical wound dataset as a proxy for the others (method bootstrap); hierarchical -0.250 [-0.577, +0.161], includes zero. Per-reference intervals excluding zero: fuseg.
- M: diff +0.227 [-0.186, +0.417], not distinguishable from a clinical wound dataset as a proxy for the others (method bootstrap); hierarchical +0.227 [-0.235, +0.411], includes zero. Per-reference intervals excluding zero: none.
- U+F: diff -0.184 [-0.435, +0.036], not distinguishable from a clinical wound dataset as a proxy for the others (method bootstrap). Per-reference intervals excluding zero: fuseg, azh.
- U+K: diff +0.044 [-0.121, +0.173], not distinguishable from a clinical wound dataset as a proxy for the others (method bootstrap). Per-reference intervals excluding zero: fuseg, azh.
- U+M: diff -0.081 [-0.260, +0.034], not distinguishable from a clinical wound dataset as a proxy for the others (method bootstrap). Per-reference intervals excluding zero: none.

## 1. Relative agreement, generator B amplitude 11

Method bootstrap intervals. Each cell: estimate [95% interval].

| Set | Ref | bench | other | diff |
|---|---|---|---|---|
| U | medetec | +0.485 [+0.014, +0.823] | +0.586 [+0.407, +0.743] | -0.101 [-0.475, +0.222] |
| U | co2wounds | +0.480 [+0.119, +0.730] | +0.280 [-0.175, +0.646] | +0.200 [-0.224, +0.610] |
| U | fuseg | +0.357 [-0.174, +0.761] | +0.367 [+0.003, +0.627] | -0.010 [-0.428, +0.368] |
| U | azh | +0.474 [+0.034, +0.797] | +0.232 [-0.183, +0.601] | +0.242 [-0.154, +0.652] |
| U | mean | +0.449 [+0.114, +0.680] | +0.366 [+0.139, +0.608] | +0.083 [-0.158, +0.253] |
| F | medetec | -0.010 [-0.546, +0.547] | +0.245 [-0.148, +0.631] | -0.255 [-0.771, +0.338] |
| F | co2wounds | +0.508 [-0.014, +0.818] | +0.168 [-0.250, +0.576] | +0.340 [-0.373, +0.953] |
| F | fuseg | +0.073 [-0.627, +0.730] | -0.045 [-0.513, +0.449] | +0.117 [-0.541, +0.743] |
| F | azh | +0.210 [-0.335, +0.661] | +0.282 [+0.000, +0.574] | -0.072 [-0.653, +0.445] |
| F | mean | +0.195 [-0.106, +0.432] | +0.163 [-0.154, +0.531] | +0.033 [-0.449, +0.400] |
| K | medetec | +0.152 [-0.430, +0.631] | +0.473 [+0.157, +0.699] | -0.321 [-0.779, +0.148] |
| K | co2wounds | +0.119 [-0.543, +0.667] | +0.519 [+0.211, +0.741] | -0.400 [-0.915, +0.092] |
| K | fuseg | -0.311 [-0.797, +0.297] | +0.396 [+0.003, +0.688] | -0.708 [-1.083, -0.287] |
| K | azh | +0.607 [+0.041, +0.936] | +0.062 [-0.487, +0.528] | +0.545 [-0.301, +1.275] |
| K | mean | +0.142 [-0.310, +0.559] | +0.363 [+0.048, +0.636] | -0.221 [-0.594, +0.089] |
| M | medetec | +0.182 [-0.586, +0.864] | +0.218 [-0.267, +0.598] | -0.036 [-0.602, +0.421] |
| M | co2wounds | +0.624 [+0.020, +0.875] | +0.088 [-0.333, +0.485] | +0.536 [+0.097, +0.953] |
| M | fuseg | +0.273 [-0.545, +0.753] | +0.048 [-0.441, +0.441] | +0.224 [-0.398, +0.645] |
| M | azh | +0.394 [-0.359, +0.890] | +0.148 [-0.222, +0.478] | +0.245 [-0.514, +0.768] |
| M | mean | +0.368 [-0.058, +0.643] | +0.126 [-0.116, +0.414] | +0.242 [-0.160, +0.442] |
| U+F | medetec | +0.209 [-0.193, +0.544] | +0.509 [+0.245, +0.710] | -0.300 [-0.751, +0.183] |
| U+F | co2wounds | +0.311 [-0.009, +0.576] | +0.496 [+0.225, +0.708] | -0.185 [-0.535, +0.145] |
| U+F | fuseg | -0.036 [-0.419, +0.318] | +0.547 [+0.312, +0.721] | -0.582 [-0.948, -0.226] |
| U+F | azh | +0.057 [-0.307, +0.371] | +0.561 [+0.330, +0.742] | -0.504 [-0.810, -0.219] |
| U+F | mean | +0.135 [-0.175, +0.402] | +0.528 [+0.301, +0.698] | -0.393 [-0.680, -0.120] |
| U+K | medetec | +0.382 [-0.011, +0.664] | +0.501 [+0.362, +0.611] | -0.119 [-0.448, +0.166] |
| U+K | co2wounds | -0.011 [-0.390, +0.339] | +0.291 [-0.002, +0.526] | -0.301 [-0.581, -0.008] |
| U+K | fuseg | -0.056 [-0.411, +0.300] | +0.341 [+0.139, +0.493] | -0.397 [-0.664, -0.105] |
| U+K | azh | +0.755 [+0.554, +0.885] | -0.030 [-0.308, +0.244] | +0.785 [+0.516, +1.056] |
| U+K | mean | +0.267 [+0.017, +0.482] | +0.276 [+0.108, +0.436] | -0.008 [-0.157, +0.136] |
| U+M | medetec | +0.413 [+0.031, +0.741] | +0.859 [+0.809, +0.903] | -0.445 [-0.806, -0.133] |
| U+M | co2wounds | +0.473 [+0.116, +0.707] | +0.779 [+0.673, +0.873] | -0.306 [-0.665, -0.070] |
| U+M | fuseg | +0.354 [-0.066, +0.711] | +0.797 [+0.710, +0.870] | -0.444 [-0.836, -0.099] |
| U+M | azh | +0.459 [+0.128, +0.694] | +0.771 [+0.666, +0.858] | -0.312 [-0.630, -0.092] |
| U+M | mean | +0.425 [+0.087, +0.686] | +0.801 [+0.745, +0.863] | -0.377 [-0.704, -0.122] |

Hierarchical (methods + units + synthetic images) and data-only intervals, M and K:

| Set | Ref | bench (hier) | other (hier) | diff (hier) | diff (data-only) |
|---|---|---|---|---|---|
| M | medetec | +0.182 [-0.571, +0.853] | +0.218 [-0.283, +0.583] | -0.036 [-0.563, +0.533] | -0.036 [-0.242, +0.188] |
| M | co2wounds | +0.624 [-0.152, +0.872] | +0.088 [-0.386, +0.505] | +0.536 [-0.136, +0.931] | +0.536 [+0.155, +0.652] |
| M | fuseg | +0.273 [-0.527, +0.774] | +0.048 [-0.408, +0.508] | +0.224 [-0.476, +0.723] | +0.224 [-0.015, +0.439] |
| M | azh | +0.394 [-0.468, +0.858] | +0.148 [-0.364, +0.487] | +0.245 [-0.543, +0.858] | +0.245 [+0.003, +0.445] |
| M | mean | +0.368 [-0.074, +0.616] | +0.126 [-0.191, +0.442] | +0.242 [-0.192, +0.418] | +0.242 [+0.086, +0.321] |
| K | medetec | +0.152 [-0.378, +0.760] | +0.473 [+0.018, +0.721] | -0.321 [-0.697, +0.378] | -0.321 [-0.413, +0.111] |
| K | co2wounds | +0.119 [-0.492, +0.717] | +0.519 [+0.129, +0.738] | -0.400 [-0.819, +0.207] | -0.400 [-0.466, -0.138] |
| K | fuseg | -0.311 [-0.777, +0.475] | +0.396 [-0.125, +0.707] | -0.708 [-0.962, -0.076] | -0.708 [-0.717, -0.369] |
| K | azh | +0.607 [-0.012, +0.910] | +0.062 [-0.486, +0.561] | +0.545 [-0.200, +1.165] | +0.545 [+0.290, +0.760] |
| K | mean | +0.142 [-0.254, +0.607] | +0.363 [-0.032, +0.656] | -0.221 [-0.495, +0.177] | -0.221 [-0.255, +0.005] |

Verdict per method set (mean over the 4 references):

- U: diff +0.083 [-0.158, +0.253], not distinguishable from a clinical wound dataset as a proxy for the others (method bootstrap). Per-reference intervals excluding zero: none.
- F: diff +0.033 [-0.449, +0.400], not distinguishable from a clinical wound dataset as a proxy for the others (method bootstrap). Per-reference intervals excluding zero: none.
- K: diff -0.221 [-0.594, +0.089], not distinguishable from a clinical wound dataset as a proxy for the others (method bootstrap); hierarchical -0.221 [-0.495, +0.177], includes zero. Per-reference intervals excluding zero: fuseg.
- M: diff +0.242 [-0.160, +0.442], not distinguishable from a clinical wound dataset as a proxy for the others (method bootstrap); hierarchical +0.242 [-0.192, +0.418], includes zero. Per-reference intervals excluding zero: co2wounds.
- U+F: diff -0.393 [-0.680, -0.120], distinguishable from a clinical wound dataset as a proxy for the others (method bootstrap). Per-reference intervals excluding zero: fuseg, azh.
- U+K: diff -0.008 [-0.157, +0.136], not distinguishable from a clinical wound dataset as a proxy for the others (method bootstrap). Per-reference intervals excluding zero: co2wounds, fuseg, azh.
- U+M: diff -0.377 [-0.704, -0.122], distinguishable from a clinical wound dataset as a proxy for the others (method bootstrap). Per-reference intervals excluding zero: medetec, co2wounds, fuseg, azh.

## 2. Texture effect (amplitude 40 minus 11)

Mean over seeds of agreement at amplitude 40 minus the same at 11; method bootstrap.

| Set | medetec | co2wounds | fuseg | azh | mean |
|---|---|---|---|---|---|
| U | +0.107 [-0.216, +0.406] | +0.325 [+0.093, +0.600] | +0.033 [-0.283, +0.355] | -0.160 [-0.579, +0.236] | +0.076 [-0.110, +0.246] |
| F | +0.056 [-0.047, +0.152] | -0.040 [-0.141, +0.064] | +0.025 [-0.084, +0.136] | +0.067 [-0.079, +0.324] | +0.027 [-0.061, +0.117] |
| K | +0.013 [-0.252, +0.266] | -0.056 [-0.346, +0.153] | -0.026 [-0.248, +0.164] | -0.048 [-0.385, +0.213] | -0.029 [-0.197, +0.071] |
| M | +0.115 [-0.240, +0.483] | -0.179 [-0.493, +0.032] | +0.006 [-0.385, +0.337] | -0.003 [-0.340, +0.329] | -0.015 [-0.142, +0.078] |
| U+F | +0.135 [-0.046, +0.308] | +0.314 [+0.131, +0.516] | +0.242 [+0.094, +0.417] | +0.143 [-0.067, +0.365] | +0.208 [+0.084, +0.359] |
| U+K | +0.062 [-0.098, +0.210] | +0.175 [+0.031, +0.314] | +0.031 [-0.139, +0.196] | -0.059 [-0.207, +0.084] | +0.052 [-0.042, +0.141] |
| U+M | +0.338 [+0.103, +0.617] | +0.340 [+0.117, +0.629] | +0.304 [+0.058, +0.605] | +0.204 [+0.003, +0.441] | +0.296 [+0.095, +0.555] |

- Within-group (U, F, K, M) per-dataset intervals excluding zero: 1 of 16 (U|co2wounds); expected by chance at 95% if every effect were zero: 0.8.
- All 7 sets (mixtures overlap with groups, so not independent): 8 of 28; chance 1.4.
- 'mean' intervals excluding zero: U+F, U+M.
- Hierarchical texture effect: M mean -0.015 [-0.218, +0.127]; K mean -0.029 [-0.236, +0.101].

## 3. Generator families

Texture (manifest) per benchmark and seed:

| Benchmark | s2024 | s7 | s101 |
|---|---|---|---|
| nrA | 2.100 | 2.101 | 2.101 |
| nrB0 | 3.228 | 2.976 | 3.064 |
| nrC0 | 2.338 | 2.406 | 2.351 |
| nrC34 | 3.201 | 3.328 | 3.282 |

### Agreement with mean of Medetec and AZH

| Set | seed | nrA | nrB0 | nrC0 | nrC34 | B0 - C34 [95%] |
|---|---|---|---|---|---|---|
| U | 2024 | +0.379 | +0.409 | +0.246 | +0.232 | +0.176 [-0.258, +0.604] |
| U | 7 | +0.334 | +0.453 | +0.251 | +0.230 | +0.223 [-0.216, +0.649] |
| U | 101 | +0.379 | +0.466 | +0.232 | +0.226 | +0.240 [-0.123, +0.634] |
| U | seedmean | +0.364 | +0.443 | +0.243 | +0.229 | +0.213 [-0.203, +0.630] |
| F | 2024 | +0.136 | +0.138 | +0.024 | +0.112 | +0.026 [-0.295, +0.365] |
| F | 7 | +0.015 | -0.007 | +0.031 | +0.070 | -0.077 [-0.646, +0.391] |
| F | 101 | +0.079 | +0.026 | +0.110 | +0.110 | -0.084 [-0.615, +0.413] |
| F | seedmean | +0.077 | +0.053 | +0.055 | +0.097 | -0.045 [-0.471, +0.379] |
| K | 2024 | +0.569 | +0.453 | +0.442 | +0.426 | +0.026 [-0.294, +0.347] |
| K | 7 | +0.569 | +0.429 | +0.446 | +0.462 | -0.033 [-0.397, +0.308] |
| K | 101 | +0.549 | +0.389 | +0.569 | +0.589 | -0.200 [-0.587, +0.074] |
| K | seedmean | +0.563 | +0.423 | +0.486 | +0.492 | -0.069 [-0.364, +0.186] |
| M | 2024 | +0.265 | +0.236 | +0.377 | +0.391 | -0.155 [-0.567, +0.210] |
| M | 7 | +0.246 | +0.136 | +0.205 | +0.268 | -0.132 [-0.540, +0.280] |
| M | 101 | +0.219 | +0.195 | +0.327 | +0.350 | -0.155 [-0.498, +0.226] |
| M | seedmean | +0.243 | +0.189 | +0.303 | +0.336 | -0.147 [-0.488, +0.182] |
| U+F | 2024 | -0.153 | +0.132 | +0.007 | +0.017 | +0.115 [-0.134, +0.351] |
| U+F | 7 | -0.168 | +0.143 | -0.058 | -0.075 | +0.218 [-0.050, +0.472] |
| U+F | 101 | -0.151 | +0.122 | +0.037 | +0.023 | +0.099 [-0.155, +0.371] |
| U+F | seedmean | -0.157 | +0.132 | -0.005 | -0.011 | +0.144 [-0.101, +0.387] |
| U+K | 2024 | +0.379 | +0.547 | +0.398 | +0.382 | +0.165 [-0.093, +0.464] |
| U+K | 7 | +0.347 | +0.559 | +0.416 | +0.416 | +0.143 [-0.121, +0.437] |
| U+K | 101 | +0.343 | +0.557 | +0.430 | +0.426 | +0.131 [-0.119, +0.441] |
| U+K | seedmean | +0.356 | +0.554 | +0.415 | +0.408 | +0.146 [-0.106, +0.447] |
| U+M | 2024 | -0.590 | +0.320 | +0.101 | +0.082 | +0.238 [-0.029, +0.490] |
| U+M | 7 | -0.589 | +0.345 | +0.174 | +0.129 | +0.216 [-0.029, +0.455] |
| U+M | 101 | -0.580 | +0.341 | +0.126 | +0.090 | +0.251 [-0.020, +0.532] |
| U+M | seedmean | -0.587 | +0.336 | +0.134 | +0.100 | +0.235 [-0.020, +0.481] |

Seed-mean orderings: U: nrB0>nrA>nrC0>nrC34; F: nrC34>nrA>nrC0>nrB0; K: nrA>nrC34>nrC0>nrB0; M: nrC34>nrC0>nrA>nrB0; U+F: nrB0>nrC0>nrC34>nrA; U+K: nrB0>nrC0>nrC34>nrA; U+M: nrB0>nrC0>nrC34>nrA.
Same ordering in all 7 sets: no.
Pairwise (cells where first beats second, out of 21 set x seed cells; and out of 7 seed means): nrA>nrB0 8/21, 3/7; nrA>nrC0 7/21, 3/7; nrA>nrC34 6/21, 2/7; nrB0>nrC0 14/21, 4/7; nrB0>nrC34 14/21, 4/7; nrC0>nrC34 13/21, 4/7.

### Agreement with mean of Medetec and CO2Wounds-V2

| Set | seed | nrA | nrB0 | nrC0 | nrC34 | B0 - C34 [95%] |
|---|---|---|---|---|---|---|
| U | 2024 | +0.424 | +0.370 | +0.389 | +0.357 | +0.013 [-0.252, +0.309] |
| U | 7 | +0.405 | +0.397 | +0.372 | +0.366 | +0.031 [-0.228, +0.318] |
| U | 101 | +0.424 | +0.425 | +0.373 | +0.343 | +0.083 [-0.151, +0.383] |
| U | seedmean | +0.418 | +0.398 | +0.378 | +0.355 | +0.042 [-0.201, +0.331] |
| F | 2024 | +0.154 | +0.316 | +0.266 | +0.343 | -0.026 [-0.384, +0.290] |
| F | 7 | +0.125 | +0.145 | +0.229 | +0.279 | -0.134 [-0.704, +0.311] |
| F | 101 | +0.167 | +0.209 | +0.297 | +0.312 | -0.103 [-0.635, +0.383] |
| F | seedmean | +0.149 | +0.223 | +0.264 | +0.311 | -0.088 [-0.527, +0.311] |
| K | 2024 | +0.431 | +0.338 | +0.481 | +0.484 | -0.145 [-0.538, +0.128] |
| K | 7 | +0.431 | +0.220 | +0.420 | +0.420 | -0.200 [-0.658, +0.218] |
| K | 101 | +0.420 | +0.224 | +0.559 | +0.589 | -0.365 [-0.830, -0.004] |
| K | seedmean | +0.427 | +0.261 | +0.487 | +0.497 | -0.237 [-0.589, +0.053] |
| M | 2024 | +0.495 | +0.364 | +0.377 | +0.336 | +0.027 [-0.405, +0.637] |
| M | 7 | +0.390 | +0.218 | +0.109 | +0.145 | +0.073 [-0.397, +0.554] |
| M | 101 | +0.280 | +0.341 | +0.205 | +0.264 | +0.077 [-0.341, +0.647] |
| M | seedmean | +0.388 | +0.308 | +0.230 | +0.248 | +0.059 [-0.337, +0.560] |
| U+F | 2024 | -0.039 | +0.246 | +0.235 | +0.245 | +0.001 [-0.204, +0.201] |
| U+F | 7 | -0.038 | +0.243 | +0.161 | +0.170 | +0.073 [-0.168, +0.295] |
| U+F | 101 | -0.030 | +0.238 | +0.238 | +0.239 | -0.001 [-0.239, +0.232] |
| U+F | seedmean | -0.036 | +0.242 | +0.211 | +0.218 | +0.024 [-0.197, +0.238] |
| U+K | 2024 | +0.361 | +0.208 | +0.306 | +0.291 | -0.083 [-0.270, +0.102] |
| U+K | 7 | +0.339 | +0.177 | +0.288 | +0.311 | -0.134 [-0.371, +0.075] |
| U+K | 101 | +0.355 | +0.193 | +0.303 | +0.303 | -0.110 [-0.345, +0.097] |
| U+K | seedmean | +0.352 | +0.193 | +0.299 | +0.302 | -0.109 [-0.302, +0.071] |
| U+M | 2024 | -0.569 | +0.312 | +0.173 | +0.151 | +0.161 [-0.118, +0.430] |
| U+M | 7 | -0.567 | +0.329 | +0.203 | +0.183 | +0.146 [-0.106, +0.392] |
| U+M | 101 | -0.568 | +0.315 | +0.170 | +0.156 | +0.159 [-0.101, +0.458] |
| U+M | seedmean | -0.568 | +0.319 | +0.182 | +0.164 | +0.155 [-0.105, +0.414] |

Seed-mean orderings: U: nrA>nrB0>nrC0>nrC34; F: nrC34>nrC0>nrB0>nrA; K: nrC34>nrC0>nrA>nrB0; M: nrA>nrB0>nrC34>nrC0; U+F: nrB0>nrC34>nrC0>nrA; U+K: nrA>nrC34>nrC0>nrB0; U+M: nrB0>nrC0>nrC34>nrA.
Same ordering in all 7 sets: no.
Pairwise (cells where first beats second, out of 21 set x seed cells; and out of 7 seed means): nrA>nrB0 10/21, 4/7; nrA>nrC0 10/21, 3/7; nrA>nrC34 10/21, 3/7; nrB0>nrC0 11/21, 4/7; nrB0>nrC34 11/21, 4/7; nrC0>nrC34 9/21, 2/7.

### Agreement with mean of all 4 wound datasets

| Set | seed | nrA | nrB0 | nrC0 | nrC34 | B0 - C34 [95%] |
|---|---|---|---|---|---|---|
| U | 2024 | +0.315 | +0.378 | +0.319 | +0.292 | +0.086 [-0.189, +0.345] |
| U | 7 | +0.293 | +0.410 | +0.310 | +0.295 | +0.115 [-0.160, +0.360] |
| U | 101 | +0.315 | +0.420 | +0.307 | +0.284 | +0.136 [-0.081, +0.364] |
| U | seedmean | +0.308 | +0.402 | +0.312 | +0.291 | +0.112 [-0.142, +0.356] |
| F | 2024 | +0.003 | +0.245 | +0.256 | +0.302 | -0.057 [-0.285, +0.151] |
| F | 7 | -0.035 | +0.109 | +0.264 | +0.271 | -0.163 [-0.613, +0.158] |
| F | 101 | -0.001 | +0.148 | +0.315 | +0.313 | -0.165 [-0.580, +0.147] |
| F | seedmean | -0.011 | +0.167 | +0.278 | +0.296 | -0.128 [-0.463, +0.141] |
| K | 2024 | +0.366 | +0.264 | +0.367 | +0.352 | -0.088 [-0.348, +0.082] |
| K | 7 | +0.366 | +0.238 | +0.335 | +0.352 | -0.113 [-0.430, +0.142] |
| K | 101 | +0.338 | +0.237 | +0.457 | +0.458 | -0.221 [-0.584, +0.064] |
| K | seedmean | +0.357 | +0.247 | +0.386 | +0.387 | -0.141 [-0.412, +0.057] |
| M | 2024 | +0.362 | +0.336 | +0.366 | +0.364 | -0.027 [-0.263, +0.192] |
| M | 7 | +0.297 | +0.270 | +0.255 | +0.286 | -0.016 [-0.282, +0.220] |
| M | 101 | +0.227 | +0.332 | +0.336 | +0.359 | -0.027 [-0.234, +0.195] |
| M | seedmean | +0.295 | +0.313 | +0.319 | +0.336 | -0.023 [-0.240, +0.175] |
| U+F | 2024 | -0.233 | +0.132 | +0.091 | +0.090 | +0.041 [-0.144, +0.219] |
| U+F | 7 | -0.223 | +0.142 | +0.026 | +0.011 | +0.131 [-0.080, +0.328] |
| U+F | 101 | -0.219 | +0.119 | +0.115 | +0.100 | +0.019 [-0.176, +0.218] |
| U+F | seedmean | -0.225 | +0.131 | +0.077 | +0.067 | +0.064 [-0.123, +0.246] |
| U+K | 2024 | +0.280 | +0.274 | +0.277 | +0.259 | +0.015 [-0.148, +0.192] |
| U+K | 7 | +0.262 | +0.271 | +0.271 | +0.287 | -0.016 [-0.196, +0.164] |
| U+K | 101 | +0.264 | +0.274 | +0.284 | +0.279 | -0.005 [-0.186, +0.185] |
| U+K | seedmean | +0.269 | +0.273 | +0.277 | +0.275 | -0.002 [-0.162, +0.171] |
| U+M | 2024 | -0.602 | +0.311 | +0.136 | +0.117 | +0.193 [-0.072, +0.450] |
| U+M | 7 | -0.598 | +0.334 | +0.197 | +0.165 | +0.169 [-0.077, +0.406] |
| U+M | 101 | -0.597 | +0.321 | +0.157 | +0.129 | +0.191 [-0.067, +0.483] |
| U+M | seedmean | -0.599 | +0.322 | +0.163 | +0.137 | +0.185 [-0.066, +0.437] |

Seed-mean orderings: U: nrB0>nrC0>nrA>nrC34; F: nrC34>nrC0>nrB0>nrA; K: nrC34>nrC0>nrA>nrB0; M: nrC34>nrC0>nrB0>nrA; U+F: nrB0>nrC0>nrC34>nrA; U+K: nrC0>nrC34>nrB0>nrA; U+M: nrB0>nrC0>nrC34>nrA.
Same ordering in all 7 sets: no.
Pairwise (cells where first beats second, out of 21 set x seed cells; and out of 7 seed means): nrA>nrB0 6/21, 1/7; nrA>nrC0 4/21, 0/7; nrA>nrC34 6/21, 1/7; nrB0>nrC0 11/21, 3/7; nrB0>nrC34 10/21, 3/7; nrC0>nrC34 14/21, 4/7.

## 4. Leave-one-method-out, M

| Dropped | bench40 mean | 6-pair wound | diff40 mean |
|---|---|---|---|
| _all | +0.353 | +0.126 | +0.227 |
| unet_r18_in | +0.371 | +0.105 | +0.266 |
| unetpp_r18_in | +0.422 | +0.182 | +0.240 |
| deeplabv3p_r18_in | +0.352 | +0.218 | +0.133 |
| fpn_r18_in | +0.359 | +0.133 | +0.225 |
| pspnet_r18_in | +0.238 | +0.057 | +0.182 |
| linknet_r18_in | +0.320 | +0.069 | +0.252 |
| manet_r18_in | +0.416 | +0.127 | +0.289 |
| segformer_mitb0_in | +0.369 | +0.115 | +0.254 |
| unet_mbv2_in | +0.395 | +0.137 | +0.258 |
| unet_effb0_in | +0.336 | +0.154 | +0.183 |
| nnunet_2d | +0.256 | +0.081 | +0.175 |

## 5. Mean Dice by group (how easy the benchmarks are)

| Set | U | F | K | M | S |
|---|---|---|---|---|---|
| medetec | 0.292 | 0.356 | 0.287 | 0.718 | 0.631 |
| co2wounds | 0.188 | 0.277 | 0.271 | 0.445 | 0.672 |
| fuseg | 0.061 | 0.531 | 0.076 | 0.791 | 0.824 |
| azh | 0.297 | 0.420 | 0.197 | 0.578 | 0.746 |
| genB_amp11_s2024 | 0.317 | 0.203 | 0.095 | 0.411 | 0.816 |
| genB_amp11_s7 | 0.316 | 0.230 | 0.100 | 0.420 | 0.809 |
| genB_amp11_s101 | 0.320 | 0.205 | 0.095 | 0.417 | 0.809 |
| genB_amp28_s2024 | 0.208 | 0.200 | 0.083 | 0.435 | 0.894 |
| genB_amp28_s7 | 0.209 | 0.215 | 0.075 | 0.445 | 0.859 |
| genB_amp28_s101 | 0.208 | 0.205 | 0.078 | 0.445 | 0.891 |
| genB_amp40_s2024 | 0.184 | 0.193 | 0.073 | 0.428 | 0.927 |
| genB_amp40_s7 | 0.187 | 0.201 | 0.063 | 0.439 | 0.895 |
| genB_amp40_s101 | 0.181 | 0.190 | 0.067 | 0.444 | 0.928 |
| nrA_s2024 | 0.692 | 0.242 | 0.841 | 0.011 | 0.826 |
| nrA_s7 | 0.698 | 0.245 | 0.843 | 0.014 | 0.842 |
| nrA_s101 | 0.693 | 0.245 | 0.841 | 0.017 | 0.830 |
| nrB0_s2024 | 0.344 | 0.208 | 0.106 | 0.352 | 0.759 |
| nrB0_s7 | 0.329 | 0.233 | 0.110 | 0.352 | 0.772 |
| nrB0_s101 | 0.341 | 0.201 | 0.103 | 0.345 | 0.753 |
| nrC0_s2024 | 0.348 | 0.159 | 0.157 | 0.220 | 0.874 |
| nrC0_s7 | 0.309 | 0.135 | 0.142 | 0.229 | 0.871 |
| nrC0_s101 | 0.316 | 0.161 | 0.147 | 0.228 | 0.845 |
| nrC34_s2024 | 0.306 | 0.155 | 0.153 | 0.211 | 0.858 |
| nrC34_s7 | 0.275 | 0.132 | 0.140 | 0.226 | 0.857 |
| nrC34_s101 | 0.278 | 0.158 | 0.143 | 0.224 | 0.826 |
| genB_amp11_seedmean | 0.318 | 0.212 | 0.097 | 0.416 | 0.811 |
| genB_amp28_seedmean | 0.208 | 0.207 | 0.079 | 0.442 | 0.882 |
| genB_amp40_seedmean | 0.184 | 0.195 | 0.068 | 0.437 | 0.917 |
| nrA_seedmean | 0.694 | 0.244 | 0.842 | 0.014 | 0.832 |
| nrB0_seedmean | 0.338 | 0.214 | 0.106 | 0.350 | 0.761 |
| nrC0_seedmean | 0.324 | 0.152 | 0.148 | 0.225 | 0.863 |
| nrC34_seedmean | 0.286 | 0.148 | 0.145 | 0.220 | 0.847 |

## Checks

- Max abs difference, per-image loader means vs pools_v5 means: K 3.3e-16, M 4.4e-16, S 3.3e-16, U 3.9e-16, F 2.2e-16.
- Max abs difference vs stats_v5.json (est, lo, hi): bench40 3.3e-16, texture 2.8e-16, matched 1.1e-16, pairs_est 0.0e+00.

## Limitations

- U and F have only per-method mean Dice on synthetic sets, so no hierarchical bootstrap for U, F or the mixtures.
- That the U and F synthetic means were computed on the same 45 images per set as the Kaggle scores cannot be checked from per-method means; textures for seed 2024 match the manifest.
- The four reference-level diffs share data (each wound pair appears in two 'other' terms), so per-reference intervals are not independent.
- Floor effect: M scores near zero on nrA (group mean Dice about 0.01; 67 to 100% of images below Dice 0.01 per method on seed 7), so M agreements involving nrA rank near-zero values and carry little information. K scores about 0.84 there, so in U+M and U+K the nrA ranking is driven by the gap between groups.
- In mixed sets (U+F, U+K, U+M) a Spearman over pooled methods mixes within-group ordering with between-group difficulty offsets; differences between mixtures and single groups partly reflect those offsets.
- Data-only intervals sit asymmetrically around the estimate because resampling images adds noise that pulls Spearman toward zero; they are context, not the primary interval.
