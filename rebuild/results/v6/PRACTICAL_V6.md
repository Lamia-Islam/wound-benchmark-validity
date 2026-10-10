# Practical consequences for choosing a model (V6, modern group M)

Script: `scripts/v6/practical_v6.py`. Full numbers: `results/v6/practical_v6.json`.

## Definitions

- Per-image Dice from `perimage_v6.load` (duplicates removed). On an empty reference mask, Dice is 1 if the prediction is empty and 0 otherwise.
- Mean Dice: mean over all kept images, empty references included.
- Bootstrap: 2,000 draws over resampling units (photograph for Medetec and AZH, image otherwise). Every image of a drawn unit counts once per draw of that unit. The draws are shared by all models (paired), and the intervals are 2.5/97.5 percentiles. The generator is `default_rng([20261010, dataset index])`.
- Indistinguishable from the top: the 95% paired interval of (top minus model) has a lower bound of 0 or less.
- Selection loss A to B: best mean on B minus the mean on B of the model ranked first on A. Random choice: mean over the 11 models of (best on B minus model on B).
- Synthetic chooser: generator B. For each seed, take the mean over all 45 images, then average the seeds 2024, 7 and 101.
- Miss: the reference is non-empty and Dice < t. The main threshold is t = 0.1; 0.05 and 0.2 are sensitivity checks. Miss rate = misses / non-empty images. Dice when detected = mean Dice on non-empty images with Dice >= t.
- Spearman: scipy, with average ranks for ties, across the 11 M models. Each figure is the mean over the 6 wound pairs.

## 1 and 2. Spread within M

| set | top | top mean [95% CI] | indistinguishable (of 10) | top wins draws | best minus median | best minus worst |
|---|---|---|---|---|---|---|
| Medetec | deeplabv3p_r18 | 0.771 [0.709, 0.824] | 4 | 0.57 | 0.045 | 0.137 |
| CO2Wounds | segformer_mitb0 | 0.486 [0.434, 0.538] | 3 | 0.66 | 0.041 | 0.087 |
| FUSeg | manet_r18 | 0.826 [0.784, 0.862] | 4 | 0.73 | 0.027 | 0.102 |
| AZH | nnunet_2d | 0.632 [0.587, 0.673] | 8 | 0.53 | 0.056 | 0.126 |
| control azh_train_slice | segformer_mitb0 | 0.627 [0.552, 0.701] | 2 | 0.92 | 0.079 | 0.185 |

The top model is different on every wound dataset.

## 3. Selection loss (rows: chosen on; columns: deployed on)

| | Medetec | CO2Wounds | FUSeg | AZH |
|---|---|---|---|---|
| Medetec | 0 | 0.053 | 0.041 | 0.126 |
| CO2Wounds | 0.045 | 0 | 0.054 | 0.021 |
| FUSeg | 0.021 | 0.062 | 0 | 0.012 |
| AZH | 0.007 | 0.012 | 0.032 | 0 |

- Off-diagonal values: mean 0.040, range 0.007 to 0.126.
- Random choice, expected loss: Medetec 0.053, CO2Wounds 0.041, FUSeg 0.035, AZH 0.054 (mean 0.046). Choosing on another wound set is only slightly better than choosing at random.
- Synthetic generator B: both amplitude 40 and amplitude 11 pick nnU-Net, which has a synthetic Dice of 0.79 and 0.81, against 0.60 or less for every other model. The resulting losses are 0.007, 0.012, 0.032 and 0 (mean 0.013).
- Minimax: nnU-Net has the smallest maximum gap from the best (0.032, on FUSeg). The next smallest are segformer at 0.054 and unet_r18 at 0.056.

## 4. Misses (t = 0.1)

| set | pretrained-10 mean miss | nnU-Net miss | U | F | K |
|---|---|---|---|---|---|
| Medetec | 0.009 | 0.000 | 0.18 | 0.23 | 0.25 |
| CO2Wounds | 0.255 | 0.260 | 0.48 | 0.42 | 0.38 |
| FUSeg | 0.030 | 0.046 | 0.81 | 0.17 | 0.77 |
| AZH | 0.301 | 0.014 | 0.19 | 0.38 | 0.60 |
| control | 0.367 | 0.042 | 0.25 | 0.50 | 0.70 |

Mean Spearman over the 6 wound pairs, at t = 0.05 / 0.1 / 0.2:

| quantity | t = 0.05 | t = 0.1 | t = 0.2 |
|---|---|---|---|
| miss rate | -0.05 | -0.06 | -0.09 |
| Dice when detected | 0.15 | 0.21 | 0.28 |

For comparison, mean Dice gives 0.13, or 0.12 on non-empty images only. Splitting Dice into misses and Dice when detected does not reveal a stable ranking.

## 5. AZH: what the data show

- The 10 pretrained models miss 30% of non-empty AZH images (range 24% to 39%), against 3% on FUSeg. nnU-Net misses 1.4% on AZH and 4.6% on FUSeg. The control set shows the same pattern: 37% against 4%.
- In 44 AZH images most pretrained models miss and nnU-Net detects. The reverse happens in 0.
- When they do detect, the pretrained models score better than nnU-Net: 0.79 Dice when detected, against 0.69.
- nnU-Net predicts something on all 10 empty AZH references, so each scores Dice 0. It still ranks first on AZH.
- Image size: AZH patches are 224 px. They are small wound crops pasted top-left on a black canvas, so 85% of pixels are black on average (FUSeg: 26%). The median non-black content is about 7,300 px.
- Lesion fraction: misses fall on small fractions (Spearman -0.48). But fraction and padding are strongly linked (Spearman -0.73).
- Within each lesion-fraction tercile, more heavily padded images are missed more often: 0.86 vs 0.28, 0.40 vs 0.05 and 0.15 vs 0.08. Padding is therefore a stronger driver of misses than lesion size.
- In the model's input frame, AZH lesions are not small: the median is 1,298 px at 256 px input, against 503 px for FUSeg.
- SIFT matches between 47 control crops and FUSeg frames show that one AZH pixel covers 1.20 FUSeg pixels (46 of 47 cases fall within 1.19 to 1.20). This means the pretrained models, which letterbox to 256 px, see AZH tissue at about 1.9 times their training magnification. nnU-Net, which works on native pixels, sees it at about 0.83 times.
- **Hypothesis (not tested):** the pretrained models fail on tight, mostly black, magnified crops because they lack the foot and skin context, and the scale, they were trained on. nnU-Net's native-scale input avoids this.

## 6. SAM / MedSAM (oracle ground-truth boxes; never pooled)

| model | Medetec | CO2Wounds | FUSeg | AZH | control |
|---|---|---|---|---|---|
| SAM mean Dice | 0.885 | 0.792 | 0.896 | 0.908 | 0.887 |
| MedSAM mean Dice | 0.376 | 0.553 | 0.752 | 0.585 | 0.559 |

Both models were given tight ground-truth component boxes, so these scores are not comparable with automatic segmentation. Empty references get no box, which produces an empty prediction and Dice 1.

The kit's MedSAM preprocessing does not match the authors' own. The kit uses `SamProcessor` with the `wanglab/medsam-vit-base` configuration: rescale by 1/255, ImageNet mean/std normalisation, longest side resized to 1024 and padded. The official `MedSAM_Inference.py` stretches the image to 1024x1024, min-max normalises it to [0, 1] with no mean/std step, and calls `image_encoder` directly. MedSAM's lower scores are therefore plausibly, at least in part, a preprocessing artefact. Measuring the size of that effect would need a rerun.

## Limitations

- Each set has 112 to 152 images, and Medetec has only 38 units, so the intervals are wide.
- On Medetec, 6 of the 11 models have a miss rate of 0, which makes the Spearman values for miss rate unstable there.
- The scale and padding explanation is correlational, and comes from one training run per model.
- The MedSAM preprocessing effect has not been measured.
