# The agreement hierarchy over eleven references

Ten references in four modalities plus a within-dataset control, 32 methods. Spearman of per-method mean score.

## References

| reference | modality | source | n | empty GT | median lesion fraction |
|---|---|---|---|---|---|
| medetec | wound | Medetec | 152 | 0 | 0.098 |
| fuseg | wound | AZH | 120 | 4 | 0.008 |
| azh | wound | AZH | 150 | 10 | 0.019 |
| kvasir_disjoint | colonoscopy | Kvasir | 120 | 0 | 0.128 |
| kvasir_sessile | colonoscopy | Kvasir | 120 | 0 | 0.043 |
| cvc_clinicdb | colonoscopy | CVC | 150 | 0 | 0.067 |
| cvc_colondb | colonoscopy | CVC | 150 | 0 | 0.038 |
| etis_larib | colonoscopy | ETIS | 150 | 0 | 0.020 |
| ph2 | dermoscopy | PH2 | 150 | 0 | 0.231 |
| tn3k_test | ultrasound | TN3K | 150 | 0 | 0.066 |
| azh_train_slice | wound | AZH | 150 | 8 | 0.016 |

## DICE, all images

| level | pairs | mean rho, all 32 | 18 untrained | 14 trained |
|---|---|---|---|---|
| same dataset | 1 | +0.944 | +0.973 | +0.947 |
| same source | 3 | +0.857 | +0.689 | +0.648 |
| different source, same modality | 10 | +0.811 | +0.790 | +0.732 |
| across modality | 32 | -0.091 | +0.020 | +0.040 |

Modality permutation test (same-modality minus across-modality mean agreement, exact over 5040 relabelings): all 32 +0.912 (p 0.0004); 18 untrained +0.746 (p 0.0004); 14 trained +0.673 (p 0.0040).

| pair | level | rho | p | 95% CI |
|---|---|---|---|---|
| azh vs azh_train_slice | same dataset | +0.944 | 0.0000 | [+0.835, +0.984] |
| cvc_clinicdb vs cvc_colondb | same source | +0.961 | 0.0000 | [+0.888, +0.984] |
| kvasir_disjoint vs kvasir_sessile | same source | +0.950 | 0.0000 | [+0.864, +0.980] |
| fuseg vs azh | same source | +0.660 | 0.0001 | [+0.430, +0.804] |
| kvasir_disjoint vs cvc_clinicdb | different source, same modality | +0.950 | 0.0000 | [+0.852, +0.984] |
| kvasir_sessile vs cvc_clinicdb | different source, same modality | +0.947 | 0.0000 | [+0.872, +0.982] |
| kvasir_sessile vs cvc_colondb | different source, same modality | +0.931 | 0.0000 | [+0.835, +0.969] |
| kvasir_disjoint vs cvc_colondb | different source, same modality | +0.907 | 0.0000 | [+0.794, +0.949] |
| kvasir_sessile vs etis_larib | different source, same modality | +0.861 | 0.0000 | [+0.698, +0.929] |
| kvasir_disjoint vs etis_larib | different source, same modality | +0.849 | 0.0000 | [+0.675, +0.930] |
| cvc_colondb vs etis_larib | different source, same modality | +0.841 | 0.0000 | [+0.661, +0.921] |
| cvc_clinicdb vs etis_larib | different source, same modality | +0.797 | 0.0000 | [+0.586, +0.914] |
| medetec vs azh | different source, same modality | +0.568 | 0.0012 | [+0.256, +0.781] |
| medetec vs fuseg | different source, same modality | +0.461 | 0.0081 | [+0.085, +0.758] |
| etis_larib vs ph2 | across modality | +0.720 | 0.0000 | [+0.411, +0.901] |
| kvasir_sessile vs ph2 | across modality | +0.622 | 0.0002 | [+0.278, +0.839] |
| cvc_colondb vs ph2 | across modality | +0.616 | 0.0001 | [+0.252, +0.864] |
| cvc_clinicdb vs ph2 | across modality | +0.570 | 0.0009 | [+0.195, +0.841] |
| kvasir_disjoint vs ph2 | across modality | +0.541 | 0.0015 | [+0.155, +0.822] |
| etis_larib vs tn3k_test | across modality | +0.343 | 0.0537 | [+0.021, +0.582] |
| cvc_colondb vs tn3k_test | across modality | +0.246 | 0.1757 | [-0.111, +0.555] |
| kvasir_sessile vs tn3k_test | across modality | +0.241 | 0.1825 | [-0.093, +0.539] |
| cvc_clinicdb vs tn3k_test | across modality | +0.218 | 0.2300 | [-0.143, +0.546] |
| kvasir_disjoint vs tn3k_test | across modality | +0.184 | 0.3118 | [-0.157, +0.494] |
| ph2 vs tn3k_test | across modality | +0.168 | 0.3527 | [-0.155, +0.467] |
| medetec vs kvasir_disjoint | across modality | +0.032 | 0.8589 | [-0.310, +0.373] |
| medetec vs cvc_clinicdb | across modality | +0.027 | 0.8812 | [-0.313, +0.365] |
| medetec vs kvasir_sessile | across modality | -0.042 | 0.8185 | [-0.397, +0.338] |
| medetec vs cvc_colondb | across modality | -0.068 | 0.7126 | [-0.395, +0.295] |
| medetec vs etis_larib | across modality | -0.158 | 0.3924 | [-0.475, +0.215] |
| azh vs kvasir_disjoint | across modality | -0.203 | 0.2680 | [-0.512, +0.159] |
| azh vs cvc_clinicdb | across modality | -0.231 | 0.2046 | [-0.544, +0.142] |
| medetec vs ph2 | across modality | -0.238 | 0.1877 | [-0.546, +0.126] |
| azh vs kvasir_sessile | across modality | -0.277 | 0.1289 | [-0.579, +0.081] |
| medetec vs tn3k_test | across modality | -0.295 | 0.1038 | [-0.589, +0.050] |
| azh vs cvc_colondb | across modality | -0.330 | 0.0639 | [-0.611, +0.011] |
| fuseg vs cvc_clinicdb | across modality | -0.460 | 0.0085 | [-0.679, -0.137] |
| fuseg vs kvasir_disjoint | across modality | -0.481 | 0.0058 | [-0.705, -0.157] |
| azh vs etis_larib | across modality | -0.509 | 0.0037 | [-0.720, -0.213] |
| fuseg vs cvc_colondb | across modality | -0.518 | 0.0026 | [-0.746, -0.165] |
| azh vs ph2 | across modality | -0.525 | 0.0018 | [-0.690, -0.257] |
| fuseg vs tn3k_test | across modality | -0.532 | 0.0026 | [-0.722, -0.243] |
| azh vs tn3k_test | across modality | -0.539 | 0.0023 | [-0.788, -0.180] |
| fuseg vs kvasir_sessile | across modality | -0.552 | 0.0011 | [-0.741, -0.224] |
| fuseg vs etis_larib | across modality | -0.733 | 0.0001 | [-0.861, -0.489] |
| fuseg vs ph2 | across modality | -0.743 | 0.0000 | [-0.848, -0.526] |

## DICE, excluding empty ground truth

| level | pairs | mean rho, all 32 | 18 untrained | 14 trained |
|---|---|---|---|---|
| same dataset | 1 | +0.920 | +0.973 | +0.943 |
| same source | 3 | +0.759 | +0.691 | +0.666 |
| different source, same modality | 10 | +0.807 | +0.788 | +0.741 |
| across modality | 32 | -0.039 | +0.016 | +0.057 |

Modality permutation test (same-modality minus across-modality mean agreement, exact over 5040 relabelings): all 32 +0.835 (p 0.0004); 18 untrained +0.750 (p 0.0004); 14 trained +0.666 (p 0.0036).

## NSD, all images

| level | pairs | mean rho, all 32 | 18 untrained | 14 trained |
|---|---|---|---|---|
| same dataset | 1 | +0.960 | +0.979 | +0.908 |
| same source | 3 | +0.898 | +0.855 | +0.927 |
| different source, same modality | 10 | +0.773 | +0.756 | +0.723 |
| across modality | 32 | +0.032 | +0.281 | +0.005 |

Modality permutation test (same-modality minus across-modality mean agreement, exact over 5040 relabelings): all 32 +0.770 (p 0.0004); 18 untrained +0.498 (p 0.0004); 14 trained +0.765 (p 0.0004).

| pair | level | rho | p | 95% CI |
|---|---|---|---|---|
| azh vs azh_train_slice | same dataset | +0.960 | 0.0000 | [+0.890, +0.982] |
| cvc_clinicdb vs cvc_colondb | same source | +0.975 | 0.0000 | [+0.929, +0.986] |
| kvasir_disjoint vs kvasir_sessile | same source | +0.937 | 0.0000 | [+0.847, +0.974] |
| fuseg vs azh | same source | +0.782 | 0.0000 | [+0.533, +0.931] |
| kvasir_sessile vs cvc_clinicdb | different source, same modality | +0.864 | 0.0000 | [+0.715, +0.935] |
| kvasir_disjoint vs cvc_clinicdb | different source, same modality | +0.861 | 0.0000 | [+0.730, +0.925] |
| kvasir_sessile vs etis_larib | different source, same modality | +0.847 | 0.0000 | [+0.667, +0.941] |
| kvasir_sessile vs cvc_colondb | different source, same modality | +0.842 | 0.0000 | [+0.664, +0.932] |
| cvc_clinicdb vs etis_larib | different source, same modality | +0.834 | 0.0000 | [+0.682, +0.911] |
| kvasir_disjoint vs cvc_colondb | different source, same modality | +0.826 | 0.0000 | [+0.650, +0.919] |
| kvasir_disjoint vs etis_larib | different source, same modality | +0.820 | 0.0000 | [+0.607, +0.917] |
| cvc_colondb vs etis_larib | different source, same modality | +0.794 | 0.0000 | [+0.584, +0.913] |
| medetec vs azh | different source, same modality | +0.580 | 0.0007 | [+0.232, +0.816] |
| medetec vs fuseg | different source, same modality | +0.461 | 0.0087 | [+0.074, +0.757] |
| etis_larib vs ph2 | across modality | +0.716 | 0.0000 | [+0.496, +0.832] |
| cvc_clinicdb vs ph2 | across modality | +0.697 | 0.0000 | [+0.394, +0.859] |
| cvc_colondb vs ph2 | across modality | +0.642 | 0.0002 | [+0.318, +0.837] |
| kvasir_disjoint vs ph2 | across modality | +0.562 | 0.0010 | [+0.216, +0.813] |
| kvasir_sessile vs ph2 | across modality | +0.503 | 0.0041 | [+0.162, +0.755] |
| medetec vs kvasir_sessile | across modality | +0.353 | 0.0462 | [+0.038, +0.625] |
| kvasir_disjoint vs tn3k_test | across modality | +0.286 | 0.1129 | [-0.094, +0.601] |
| medetec vs kvasir_disjoint | across modality | +0.266 | 0.1392 | [-0.069, +0.549] |
| cvc_clinicdb vs tn3k_test | across modality | +0.218 | 0.2302 | [-0.154, +0.545] |
| cvc_colondb vs tn3k_test | across modality | +0.212 | 0.2409 | [-0.161, +0.555] |
| medetec vs etis_larib | across modality | +0.199 | 0.2793 | [-0.140, +0.496] |
| kvasir_sessile vs tn3k_test | across modality | +0.154 | 0.3933 | [-0.214, +0.476] |
| etis_larib vs tn3k_test | across modality | +0.153 | 0.3976 | [-0.231, +0.507] |
| medetec vs cvc_clinicdb | across modality | +0.133 | 0.4684 | [-0.214, +0.466] |
| azh vs kvasir_sessile | across modality | +0.084 | 0.6457 | [-0.240, +0.373] |
| medetec vs cvc_colondb | across modality | +0.072 | 0.6960 | [-0.272, +0.405] |
| ph2 vs tn3k_test | across modality | +0.060 | 0.7433 | [-0.289, +0.411] |
| azh vs kvasir_disjoint | across modality | -0.029 | 0.8764 | [-0.359, +0.274] |
| azh vs etis_larib | across modality | -0.055 | 0.7695 | [-0.404, +0.294] |
| fuseg vs kvasir_sessile | across modality | -0.132 | 0.4658 | [-0.477, +0.238] |
| medetec vs tn3k_test | across modality | -0.146 | 0.4202 | [-0.474, +0.206] |
| medetec vs ph2 | across modality | -0.160 | 0.3831 | [-0.473, +0.214] |
| azh vs cvc_colondb | across modality | -0.240 | 0.1841 | [-0.527, +0.105] |
| azh vs cvc_clinicdb | across modality | -0.248 | 0.1687 | [-0.538, +0.109] |
| fuseg vs kvasir_disjoint | across modality | -0.284 | 0.1169 | [-0.628, +0.082] |
| azh vs tn3k_test | across modality | -0.303 | 0.0951 | [-0.604, +0.080] |
| fuseg vs etis_larib | across modality | -0.327 | 0.0682 | [-0.652, +0.044] |
| fuseg vs cvc_colondb | across modality | -0.351 | 0.0491 | [-0.630, +0.057] |
| fuseg vs cvc_clinicdb | across modality | -0.403 | 0.0229 | [-0.682, +0.018] |
| azh vs ph2 | across modality | -0.471 | 0.0071 | [-0.700, -0.155] |
| fuseg vs tn3k_test | across modality | -0.476 | 0.0054 | [-0.669, -0.186] |
| fuseg vs ph2 | across modality | -0.666 | 0.0000 | [-0.821, -0.405] |

## NSD, excluding empty ground truth

| level | pairs | mean rho, all 32 | 18 untrained | 14 trained |
|---|---|---|---|---|
| same dataset | 1 | +0.962 | +0.979 | +0.934 |
| same source | 3 | +0.875 | +0.862 | +0.892 |
| different source, same modality | 10 | +0.778 | +0.755 | +0.738 |
| across modality | 32 | +0.059 | +0.281 | +0.029 |

Modality permutation test (same-modality minus across-modality mean agreement, exact over 5040 relabelings): all 32 +0.742 (p 0.0004); 18 untrained +0.499 (p 0.0004); 14 trained +0.744 (p 0.0004).

## Pairs split by the training-domain boundary, DICE, all images

| category | pairs | all 32 | 18 untrained | 14 trained |
|---|---|---|---|---|
| wound x wound | 3 | +0.563 | +0.458 | +0.158 |
| non-wound, same modality | 10 | +0.899 | +0.859 | +0.879 |
| non-wound, across modality | 11 | +0.406 | +0.035 | +0.142 |
| wound x non-wound | 21 | -0.351 | +0.012 | -0.014 |

## Pairs split by the training-domain boundary, DICE, excluding empty ground truth

| category | pairs | all 32 | 18 untrained | 14 trained |
|---|---|---|---|---|
| wound x wound | 3 | +0.452 | +0.454 | +0.204 |
| non-wound, same modality | 10 | +0.899 | +0.859 | +0.879 |
| non-wound, across modality | 11 | +0.406 | +0.035 | +0.142 |
| wound x non-wound | 21 | -0.273 | +0.005 | +0.013 |

## Pairs split by the training-domain boundary, NSD, all images

| category | pairs | all 32 | 18 untrained | 14 trained |
|---|---|---|---|---|
| wound x wound | 3 | +0.608 | +0.786 | +0.396 |
| non-wound, same modality | 10 | +0.860 | +0.777 | +0.882 |
| non-wound, across modality | 11 | +0.382 | +0.071 | +0.119 |
| wound x non-wound | 21 | -0.152 | +0.391 | -0.055 |

## Pairs split by the training-domain boundary, NSD, excluding empty ground truth

| category | pairs | all 32 | 18 untrained | 14 trained |
|---|---|---|---|---|
| wound x wound | 3 | +0.601 | +0.790 | +0.410 |
| non-wound, same modality | 10 | +0.860 | +0.777 | +0.882 |
| non-wound, across modality | 11 | +0.382 | +0.071 | +0.119 |
| wound x non-wound | 21 | -0.111 | +0.392 | -0.018 |

## Mean Dice by subset on each reference

| reference | modality | 14 trained | 18 untrained | gap |
|---|---|---|---|---|
| medetec | wound | 0.356 | 0.292 | +0.064 |
| fuseg | wound | 0.543 | 0.062 | +0.481 |
| azh | wound | 0.420 | 0.297 | +0.122 |
| kvasir_disjoint | colonoscopy | 0.127 | 0.229 | -0.102 |
| kvasir_sessile | colonoscopy | 0.053 | 0.111 | -0.057 |
| cvc_clinicdb | colonoscopy | 0.089 | 0.180 | -0.090 |
| cvc_colondb | colonoscopy | 0.051 | 0.125 | -0.074 |
| etis_larib | colonoscopy | 0.015 | 0.067 | -0.052 |
| ph2 | dermoscopy | 0.081 | 0.490 | -0.409 |
| tn3k_test | ultrasound | 0.007 | 0.102 | -0.095 |
