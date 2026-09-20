# Where each clinical reference comes from

None of these datasets are redistributed in this repository.

| reference | source |
|---|---|
| Medetec | `data/wound-segmentation/data/Medetec_foot_ulcer_224` in github.com/uwm-bigdata/wound-segmentation |
| FUSeg | `data/Foot Ulcer Segmentation Challenge` in the same repository; MICCAI 2021 challenge, fusc.grand-challenge.org |
| AZH | `data/wound_dataset/azh_wound_care_center_dataset_patches.zip` in the same repository. Wang et al., "Fully Automatic Wound Segmentation with Deep Convolutional Neural Networks", Scientific Reports 10:21897, 2020 |
| Kvasir-SEG | datasets.simula.no/kvasir-seg. Jha et al., MMM 2020 |
| Kvasir-SEG sessile | the sessile subset distributed with Kvasir-SEG |

## Handling notes

- Medetec masks are 0/1 and FUSeg masks 0/255, so wound masks are binarised on
  any non-zero value. A mid-range threshold silently empties the Medetec masks.
- Kvasir masks are JPEG-compressed with anti-aliased edges carrying values of
  1, 2, 3 and so on, so they are thresholded at 127 instead. Binarising those on
  any non-zero value inflates every lesion.
- Kvasir frames vary in size and are resampled to a 256 px long side, matching
  Medetec's resolution.
- Every image statistic is measured at a canonical 512 px. Measured natively,
  Medetec at 224 px and FUSeg at 512 px order in the opposite direction.
