### Wound and polyp segmentation datasets, dataset bias and the effect of framing

Public wound segmentation data are few, and several of the main sets come from one clinic. Wang et al. (2020) released 1,109 foot ulcer photographs from 889 patients at the AZH Wound and Vascular Center in Milwaukee, taken under uncontrolled lighting and backgrounds. Their pipeline localises each wound with a detector, crops a patch around it and zero-pads the patch to 224 × 224 pixels. The FUSeg challenge at MICCAI 2021 drew on the same clinic and patients: 1,210 annotated images, zero-padded to 512 × 512 pixels, split 810/200/200, with entries ranked by Dice (Wang et al., 2024). The two releases therefore share a source but differ in how each frame is prepared. The Medetec wound database is a web collection of free stock images covering leg ulcers, pressure ulcers, diabetic foot ulcers, burns and other wounds, many digitised from 35 mm transparencies; the page provides no masks or citation instructions, so it is usually cited as a web resource (Medetec, 2014). The DFUC 2022 challenge released 2,000 training and 2,000 test images of diabetic foot ulcers and ranked 26 teams by Dice (Yap et al., 2024). Its dataset paper reports that clinicians' delineations agree less well with model predictions than contours refined by image processing, which shows that the annotation convention itself shapes the scores (Kendrick et al., 2022). Zhang et al. (2022) survey deep learning for wound classification, detection and segmentation, including the public datasets and the preprocessing applied to them.

For polyps, Kvasir-SEG provides 1,000 colonoscopy images with masks verified by a gastroenterologist, at resolutions from 332 × 487 to 1920 × 1072 pixels (Jha et al., 2020). CVC-ClinicDB consists of frames extracted from colonoscopy videos with polyp masks and served as the training set of a MICCAI 2015 sub-challenge (Bernal et al., 2015). PolypGen collects 8,037 frames from six centres in five countries and was built to test whether methods generalise to an unseen centre (Ali et al., 2023).

Each of these collections reflects choices about which cases to include, how to frame them and how to draw the boundary. Torralba and Efros (2011) showed that a simple classifier can tell which dataset an image came from at well above chance, and that detectors lose much of their accuracy when tested on a different dataset. Geirhos et al. (2020) describe shortcut learning, where a model relies on simple cues that hold on the benchmark but fail under other conditions. Castro et al. (2020) separate dataset shift in medical imaging into population, acquisition, annotation and prevalence shift and selection bias. Under their terms, cropping a wound tightly or leaving it small in a wide frame is a form of acquisition and selection that changes the image distribution without changing the disease. Maier-Hein et al. (2018) found that challenge rankings change with the choice of test cases, metric and aggregation. These results explain why we expect agreement about which method is better to drop between independent datasets, and why methods trained on one modality fail on another. Our cross-dataset and cross-modality results are consistent with them.

Region size is a known factor in segmentation evaluation. Liu et al. (2024) show that Dice and cross-entropy losses each contain an implicit region-size penalty and that Dice favours certain extremely imbalanced solutions. Reinke et al. (2024) list sensitivity to structure size among metric pitfalls and note that errors on small structures can be hidden when they are averaged with large ones. Normalised surface Dice scores boundary agreement within a set tolerance rather than area overlap (Nikolov et al., 2021), which is why we report it alongside Dice. Alajmi et al. (2026) compared five preprocessing pipelines for optic disc and cup segmentation across three datasets and three seeds. Region-of-interest cropping scored lower on REFUGE but was best or close to best for the optic disc on ORIGA and Drishti-GS, so whether cropping helps depended on the dataset. The authors treat their rankings as indicative, and their crop centres came from ground-truth centroids.

Our work differs from these studies in what varies and what is measured. Prior work shows that cropping or region size changes performance, or which pipeline wins, from one dataset to another. We hold the dataset fixed and vary the lesion fraction of the clinical reference itself, which is a framing and curation convention. We then ask what the reference says about synthetic benchmark design. Within one modality, the reference's verdict on lesion-to-surround contrast moves monotonically with its lesion fraction, under both Dice and normalised surface Dice. A reference therefore cannot be treated as a neutral judge of a synthetic benchmark until its framing convention is stated.

### References

1. Ali, S., Jha, D., Ghatwary, N., et al. (2023). A multi-centre polyp detection and segmentation dataset for generalisability assessment. *Scientific Data*, 10, 75. https://doi.org/10.1038/s41597-023-01981-y. Checked: https://www.nature.com/articles/s41597-023-01981-y

2. Alajmi, A., Elnahal, Y., Othman, M., Aljuhani, M., Alharbi, A., & Abdelhady, G. (2026). A Systematic Multi-Dataset, Multi-Seed Evaluation of Preprocessing Strategies for Retinal Optic Disc and Cup Segmentation. *Diagnostics*, 16(17), 2880. https://doi.org/10.3390/diagnostics16172880. Checked: https://www.mdpi.com/2075-4418/16/17/2880

3. Bernal, J., Sánchez, F. J., Fernández-Esparrach, G., Gil, D., Rodríguez, C., & Vilariño, F. (2015). WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians. *Computerized Medical Imaging and Graphics*, 43, 99-111. https://doi.org/10.1016/j.compmedimag.2015.02.007. Checked: https://portalrecerca.uab.cat/ca/publications/wm-dova-maps-for-accurate-polyp-highlighting-in-colonoscopy-valid/ and the CVC-ClinicDB page naming this as the required citation: https://polyp.grand-challenge.org/CVCClinicDB/

4. Castro, D. C., Walker, I., & Glocker, B. (2020). Causality matters in medical imaging. *Nature Communications*, 11, 3673. https://doi.org/10.1038/s41467-020-17478-w. Checked: https://www.nature.com/articles/s41467-020-17478-w

5. Geirhos, R., Jacobsen, J.-H., Michaelis, C., et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11), 665-673. https://doi.org/10.1038/s42256-020-00257-z. Checked: https://www.nature.com/articles/s42256-020-00257-z

6. Jha, D., Smedsrud, P. H., Riegler, M. A., Halvorsen, P., de Lange, T., Johansen, D., & Johansen, H. D. (2020). Kvasir-SEG: A Segmented Polyp Dataset. In *MultiMedia Modeling (MMM 2020)*, Lecture Notes in Computer Science, vol. 11962, pp. 451-462. Springer. https://doi.org/10.1007/978-3-030-37734-2_37. Checked: https://link.springer.com/chapter/10.1007/978-3-030-37734-2_37; dataset size and resolution from https://datasets.simula.no/kvasir-seg/

7. Kendrick, C., Cassidy, B., Pappachan, J. M., et al. (2022). Translating Clinical Delineation of Diabetic Foot Ulcers into Machine Interpretable Segmentation. arXiv:2204.11618. Checked: https://arxiv.org/abs/2204.11618

8. Liu, B., Dolz, J., Galdran, A., Kobbi, R., & Ben Ayed, I. (2024). Do we really need dice? The hidden region-size biases of segmentation losses. *Medical Image Analysis*, 91, 103015. https://doi.org/10.1016/j.media.2023.103015. Checked: https://dsp.tecnalia.com/items/a28c9911-511f-4c6d-8303-f34fb9d66434/full (preprint: arXiv:2104.08717)

9. Maier-Hein, L., Eisenmann, M., Reinke, A., et al. (2018). Why rankings of biomedical image analysis competitions should be interpreted with care. *Nature Communications*, 9, 5217. https://doi.org/10.1038/s41467-018-07619-7. Checked: https://www.nature.com/articles/s41467-018-07619-7

10. Medetec (2014). Medetec Wound Database: medical image databases (S. Thomas, page author). http://www.medetec.co.uk/files/medetec-image-databases.html (accessed 8 October 2026). Checked: same URL

11. Nikolov, S., Blackwell, S., Zverovitch, A., et al. (2021). Clinically Applicable Segmentation of Head and Neck Anatomy for Radiotherapy: Deep Learning Algorithm Development and Validation Study. *Journal of Medical Internet Research*, 23(7), e26151. https://doi.org/10.2196/26151. Checked: https://www.jmir.org/2021/7/e26151

12. Reinke, A., Tizabi, M. D., Baumgartner, M., et al. (2024). Understanding metric-related pitfalls in image analysis validation. *Nature Methods*, 21(2), 182-194. https://doi.org/10.1038/s41592-023-02150-0. Checked: https://www.nature.com/articles/s41592-023-02150-0

13. Torralba, A., & Efros, A. A. (2011). Unbiased look at dataset bias. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 1521-1528. https://doi.org/10.1109/CVPR.2011.5995347. Checked: https://mlanthology.org/cvpr/2011/torralba2011cvpr-unbiased and full text https://people.csail.mit.edu/torralba/publications/datasets_cvpr11.pdf

14. Wang, C., Anisuzzaman, D. M., Williamson, V., Dhar, M. K., Rostami, B., Niezgoda, J., Gopalakrishnan, S., & Yu, Z. (2020). Fully automatic wound segmentation with deep convolutional neural networks. *Scientific Reports*, 10, 21897. https://doi.org/10.1038/s41598-020-78799-w. Checked: https://www.nature.com/articles/s41598-020-78799-w

15. Wang, C., Mahbod, A., Ellinger, I., Galdran, A., Gopalakrishnan, S., Niezgoda, J., & Yu, Z. (2024). FUSeg: The Foot Ulcer Segmentation Challenge. *Information*, 15(3), 140. https://doi.org/10.3390/info15030140. Checked: https://www.mdpi.com/2078-2489/15/3/140

16. Yap, M. H., Cassidy, B., Byra, M., et al. (2024). Diabetic foot ulcers segmentation challenge report: Benchmark and analysis. *Medical Image Analysis*, 94, 103153. https://doi.org/10.1016/j.media.2024.103153. Checked: https://recercat.cat/handle/10230/72832?show=full

17. Zhang, R., Tian, D., Xu, D., Qian, W., & Yao, Y. (2022). A Survey of Wound Image Analysis Using Deep Learning: Classification, Detection, and Segmentation. *IEEE Access*, 10, 79502-79515. https://doi.org/10.1109/ACCESS.2022.3194529. Checked: https://researchwith.stevens.edu/en/publications/a-survey-of-wound-image-analysis-using-deep-learning-classificati/

### Verification notes

- Nikolov et al. (2021): the JMIR page confirmed the title, venue, first author, 29 authors in total and the last author. The second and third authors (Blackwell, Zverovitch) were confirmed from the arXiv preprint of the same study (arXiv:1809.04430, https://arxiv.org/abs/1809.04430), not from the JMIR page itself.
- Liu et al. (2024): the DOI resolver refused the request (rate limit), so the details were checked on the Tecnalia institutional repository record instead.
- Medetec has no formal citation or licence text on its page; the entry above cites the page as a web resource with an access date.
- CVC-ClinicDB frame count is not stated in the text of the pages checked, so the subsection gives no count.

### Unverified, do not cite

- None. Every requested item was found and checked against a primary or repository page.
