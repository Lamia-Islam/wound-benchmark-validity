# Phase 1 results: competence score (generated from novelty/results/p1.json)

Fit on 392 FUSeg training images (class sizes 131, 130, 131); threshold from 148 held-out FUSeg training images: −35.31. ResNet-50 embedding: skipped: torchvision not installed and the weight download was refused by the network proxy (HTTP 403). CO2Wounds-V2: dropped: images not available.

## Score by evaluation set

| Set | Images | Median score | Share below threshold |
|---|---|---|---|
| medetec | 152 | −82944055.43 | 100.0% |
| fuseg | 112 | −11.38 | 6.2% |
| azh | 146 | −82944346.01 | 100.0% |
| azh_train_slice | 78 | −82944659.48 | 100.0% |

## H1: failure prediction (95% unit-bootstrap intervals)

| Set | Non-empty | Majority misses | Score AUROC | Black share AUROC | Lesion fraction AUROC | Score Spearman | Partial Spearman (score given black share) |
|---|---|---|---|---|---|---|---|
| pooled | 468 | 73 | 0.92 (0.87 to 0.96) | 0.92 (0.88 to 0.96) | 0.75 (0.67 to 0.81) | 0.61 (0.53 to 0.68) | 0.20 (0.08 to 0.33) |
| medetec | 152 | 0 | not reported (too few misses) | not reported (too few misses) | not reported (too few misses) | −0.13 (−0.41 to 0.16) | −0.15 (−0.41 to 0.13) |
| fuseg | 108 | 3 | not reported (too few misses) | not reported (too few misses) | not reported (too few misses) | −0.13 (−0.27 to 0.02) | −0.12 (−0.27 to 0.06) |
| azh | 136 | 41 | 0.90 (0.83 to 0.96) | 0.88 (0.79 to 0.95) | 0.71 (0.59 to 0.83) | 0.77 (0.68 to 0.83) | 0.34 (0.17 to 0.48) |
| azh_train_slice | 72 | 29 | 0.86 (0.76 to 0.94) | 0.87 (0.78 to 0.95) | 0.69 (0.54 to 0.81) | 0.79 (0.67 to 0.87) | 0.16 (0.00 to 0.36) |

H1 verdict by the pre-registered rule: supported (pooled AUROC lower bound 0.87; partial Spearman interval 0.08 to 0.33).

## H2: competence-filtered agreement (M19, wound pairs among the three available wound datasets)

Images removed by the threshold: medetec 152 of 152, fuseg 7 of 112, azh 146 of 146.
Agreement with all images 0.12; filtered n/a; change n/a (n/a to n/a); random-removal control 95th percentile n/a.
H2 verdict by the pre-registered rule: not supported.
The threshold removes every image of at least one wound dataset, so filtered agreement is undefined.

## H3: benchmark shift (exploratory)

21 synthetic sets regenerated (texture of the regenerated sets differs from the stored values by at most 0.0000).

| Method set | Sets | Spearman(mean score, benchmark agreement) | Spearman(Frechet distance, benchmark agreement) |
|---|---|---|---|
| M19 | 21 | −0.83 | 0.83 |
| U+M19 | 21 | 0.49 | −0.49 |

## Note on the score (observation, not a change to the pre-registered analysis)

The FUSeg training images all have the same height and width, so the size features have zero variance in the fit. After standardisation and the variance floor, any image of another size receives a very low score. The score therefore flags every Medetec, AZH and control image, and its pooled ranking mixes image size with the other features.
