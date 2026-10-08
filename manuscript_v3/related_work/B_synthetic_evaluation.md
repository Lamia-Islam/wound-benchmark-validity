### Synthetic data for evaluation, simulation-based validation and domain randomization

Most work on synthetic images uses them to train models. Domain randomization trains on simulated images whose rendering is varied, often in deliberately unrealistic ways, so that the real world looks like one more variation. Tobin et al. (2017) trained an object localizer only on simulated images with random, non-realistic textures and transferred it to real camera images for robotic grasping. Tremblay et al. (2018) applied the same idea to car detection, varying lighting, pose and texture, and found that fine-tuning on real data after synthetic pre-training outperformed training on real data alone. Structured domain randomization (Prakash et al., 2019) placed objects according to problem-specific context distributions instead of uniformly at random. In medical imaging, SynthSeg (Billot et al., 2023) trains a brain MRI segmentation network on images generated from label maps with randomized contrast and resolution, and segments real scans of many modalities without retraining. Chen et al. (2021) review the growing use of synthetic data in medical machine learning and raise concerns about its weaknesses and about how ready current policy is to handle them. All of these works ask whether synthetic data produces a good model. Our question is different: whether a synthetic test set orders a set of existing methods the same way clinical data does. A training set benefits from variation wide enough to cover the real domain. A benchmark is useful only if its difficulty structure resembles the clinical one, and our finding that matching the clinical distribution helps validation sits in contrast to the randomization advice for training.

Simulation has a long history as a validation tool in medical imaging, mainly because simulated data comes with exact ground truth. The BrainWeb digital phantom (Collins et al., 1998) stores tissue fractions per voxel and drives MR and PET simulators so that segmentation and registration methods can be tested against a known answer. Alfano et al. (2011) built an MRI brain phantom with normal tissue classes and an optional multiple sclerosis lesion class, and used it to test segmentation under different noise levels and slice thicknesses. The VICTRE in-silico trial (Badano et al., 2018) simulated 2,986 virtual breasts with inserted lesions, imaged them with simulated tomosynthesis and mammography, and read them with a computational observer. It found an advantage for tomosynthesis, larger for masses than for calcifications, and the authors report that this agrees with an earlier comparative clinical trial. VICTRE checks one comparison between two imaging systems against one clinical result. Phantom studies such as BrainWeb assume that agreement with simulated ground truth carries over to patients. We test that assumption directly for segmentation benchmarks, across 32 methods and several clinical reference sets.

A smaller body of work evaluates synthetic data as a test set. Shoshan et al. (2023) showed that synthetic images can replace a held-out validation set for model selection in image classification when real data is scarce, and proposed calibrating synthetic error estimates to match real error. Van Breugel et al. (2023) generated synthetic test sets for small subgroups and simulated distribution shifts, and reported more accurate performance estimates than real test data alone. Kang (2022) compared the Kendall rank correlation of ten person re-identification algorithms across real and synthetic datasets and concluded that one synthetic dataset ranks methods like real data. Maheshwari et al. (2024) found that LLM-generated benchmarks reflected method comparisons on simple NLP tasks but not on harder ones. Borsukiewicz et al. (2026) tested 12 synthetic face datasets against 7 real benchmarks with 24 pre-trained models and found that benchmarking fidelity varies widely, with two datasets agreeing within the disagreement already present among real benchmarks. These studies share our rank-based view of validity. They use generative or rendered data from outside medicine and a fixed set of models. Our benchmarks are parametric, so design properties can be changed one at a time, and we show that the answer depends on both the clinical reference and the method pool.

For generated medical images, Deo et al. (2025) found that common no-reference image quality metrics correlate poorly with downstream segmentation suitability on brain MRI and miss localized anatomical errors. For wounds, Sarp et al. (2021) generated chronic wound images with a GAN for clinical training and compared them with real images by mean squared error and visual inspection, and Basiri et al. (2023) generated diabetic foot ulcer images with a diffusion model and found that FID and KID did not align with clinician judgements of realism. We found no prior work that uses synthetic or simulated wound images to rank segmentation methods. Our results extend the point made by Deo et al. (2025) from realism metrics to benchmark statistics: benchmarks matched on a single texture statistic differ widely in rank fidelity, so validity has to be measured against clinical rankings rather than inferred from image properties.

## References

1. Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., Abbeel, P. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 23-30. DOI: 10.1109/IROS.2017.8202133. arXiv:1703.06907.
   Checked: https://arxiv.org/abs/1703.06907 ; https://api.crossref.org/works/10.1109/IROS.2017.8202133

2. Tremblay, J., Prakash, A., Acuna, D., et al. (2018). Training deep networks with synthetic data: Bridging the reality gap by domain randomization. IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 969-977. DOI: 10.1109/CVPRW.2018.00143. arXiv:1804.06516.
   Checked: https://arxiv.org/abs/1804.06516 ; https://mlanthology.org/cvprw/2018/tremblay2018cvprw-training

3. Prakash, A., Boochoon, S., Brophy, M., et al. (2019). Structured domain randomization: Bridging the reality gap by context-aware synthetic data. 2019 International Conference on Robotics and Automation (ICRA). DOI: 10.1109/ICRA.2019.8794443. arXiv:1810.10093.
   Checked: https://arxiv.org/abs/1810.10093 ; https://research.nvidia.com/labs/amri/publication/prakash2019structured

4. Billot, B., Greve, D. N., Puonti, O., et al. (2023). SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining. Medical Image Analysis, 86, 102789. DOI: 10.1016/j.media.2023.102789. arXiv:2107.09559.
   Checked: https://api.crossref.org/works/10.1016/j.media.2023.102789 ; https://arxiv.org/abs/2107.09559

5. Chen, R. J., Lu, M. Y., Chen, T. Y., Williamson, D. F. K., Mahmood, F. (2021). Synthetic data in machine learning for medicine and healthcare. Nature Biomedical Engineering, 5(6), 493-497. DOI: 10.1038/s41551-021-00751-8.
   Checked: https://www.nature.com/articles/s41551-021-00751-8

6. Collins, D. L., Zijdenbos, A. P., Kollokian, V., et al. (1998). Design and construction of a realistic digital brain phantom. IEEE Transactions on Medical Imaging, 17(3), 463-468. DOI: 10.1109/42.712135.
   Checked: https://api.crossref.org/works/10.1109/42.712135 ; https://acoustic.univ-rennes1.fr/lib/exe/fetch.php/pdf:collins1998.pdf

7. Alfano, B., Comerci, M., Larobina, M., et al. (2011). An MRI digital brain phantom for validation of segmentation methods. Medical Image Analysis, 15(3), 329-339. DOI: 10.1016/j.media.2011.01.004.
   Checked: https://www.iris.unina.it/handle/11588/384408 ; https://www.iac.cnr.it/mri-digital-brain-phantom-validation-segmentation-methods-0

8. Badano, A., Graff, C. G., Badal, A., et al. (2018). Evaluation of digital breast tomosynthesis as replacement of full-field digital mammography using an in silico imaging trial. JAMA Network Open, 1(7), e185474. DOI: 10.1001/jamanetworkopen.2018.5474.
   Checked: https://api.crossref.org/works/10.1001/jamanetworkopen.2018.5474 ; https://www.imagwiki.nibib.nih.gov/index.php/resources/publications/evaluation-digital-breast-tomosynthesis-replacement-full-field-digital

9. Shoshan, A., Bhonker, N., Kviatkovsky, I., Fintz, M., Medioni, G. (2023). Synthetic data for model selection. Proceedings of the 40th International Conference on Machine Learning (ICML), PMLR 202, pp. 31633-31656. arXiv:2105.00717.
   Checked: https://mlanthology.org/icml/2023/shoshan2023icml-synthetic

10. van Breugel, B., Seedat, N., Imrie, F., van der Schaar, M. (2023). Can you rely on your model evaluation? Improving model evaluation with synthetic test data. arXiv:2310.16524. (Listed as a NeurIPS 2023 poster in search results; venue page not fetched.)
    Checked: https://arxiv.org/abs/2310.16524

11. Kang, C. (2022). Is synthetic dataset reliable for benchmarking generalizable person re-identification? arXiv:2209.05047.
    Checked: https://arxiv.org/abs/2209.05047

12. Maheshwari, G., Ivanov, D., El Haddad, K. (2024). Efficacy of synthetic data as a benchmark. arXiv:2409.11968. DOI: 10.48550/arXiv.2409.11968.
    Checked: https://arxiv.org/abs/2409.11968

13. Borsukiewicz, P., Lunghi, D., Ouédraogo, W. C., Klein, J., Bissyandé, T. F. (2026). Benchmarking face recognition without real faces. arXiv:2607.14932.
    Checked: https://arxiv.org/abs/2607.14932

14. Deo, Y., Jia, Y., Lassila, T., et al. (2025). Metrics that matter: Evaluating image quality metrics for medical image generation. arXiv:2505.07175. DOI: 10.48550/arXiv.2505.07175.
    Checked: https://arxiv.org/abs/2505.07175 ; https://pure.york.ac.uk/portal/en/publications/metrics-that-matter-evaluating-image-quality-metrics-for-medical-/

15. Sarp, S., Kuzlu, M., Wilson, E., Guler, O. (2021). WG2AN: Synthetic wound image generation using generative adversarial network. The Journal of Engineering, 2021(5), 286-294. DOI: 10.1049/tje2.12033.
    Checked: https://doaj.org/article/1f28f75881364d7eb809173e864b0146 ; https://digitalcommons.odu.edu/engtech_fac_pubs/175

16. Basiri, R., Manji, K., Harton, F., Poonja, A., Popovic, M. R., Khan, S. S. (2023). Synthesizing diabetic foot ulcer images with diffusion model. arXiv:2310.20140.
    Checked: https://arxiv.org/abs/2310.20140

## Unverified, do not cite

- Page numbers for Prakash et al. (2019), ICRA. The DOI and venue were confirmed, but no fetched page showed the page range (Crossref, Semantic Scholar and IEEE Xplore were rate-limited or empty). Add pages only after checking https://doi.org/10.1109/ICRA.2019.8794443.
- Peer-reviewed venues for van Breugel et al. (2023, possibly NeurIPS 2023) and Kang (2022). Only the arXiv records were checked, so they are cited as arXiv preprints.
