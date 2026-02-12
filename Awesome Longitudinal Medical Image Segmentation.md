# Awesome Longitudinal Medical Image Segmentation

A curated list of papers, code, and datasets for **longitudinal (temporal/sequential) medical image segmentation** — methods that exploit multiple timepoints, prior scans, or temporal context to improve segmentation accuracy and consistency. This list covers registration-based approaches, recurrent and attention architectures, self-supervised temporal pretraining, change detection, foundation model adaptations, and clinical applications including adaptive radiotherapy.

Contributions welcome! Papers are organized by methodological category. Each entry includes the paper link, publication venue, and official code repository where available.

**Legend:** 🔗 Paper | 💻 Code | ⭐ Highly cited / influential

---

## Contents

- [Registration-Based Methods](#registration-based-methods)
- [Recurrent and Sequential Architectures](#recurrent-and-sequential-architectures)
- [Attention and Transformer-Based Temporal Fusion](#attention-and-transformer-based-temporal-fusion)
- [Prior-Conditioned Segmentation](#prior-conditioned-segmentation)
- [Self-Supervised Temporal Pretraining](#self-supervised-temporal-pretraining)
- [Joint Change Detection and Segmentation](#joint-change-detection-and-segmentation)
- [Foundation Models and Emerging Approaches](#foundation-models-and-emerging-approaches)
- [Clinical Applications: Adaptive RT and Longitudinal Monitoring](#clinical-applications-adaptive-rt-and-longitudinal-monitoring)
- [Frameworks and Toolkits](#frameworks-and-toolkits)
- [Datasets and Challenges](#datasets-and-challenges)
- [Related Workshops](#related-workshops)

---

## Registration-Based Methods

Methods that jointly learn or leverage deformable registration to propagate segmentations across timepoints or to enforce anatomical correspondence in longitudinal studies.

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2019 | ⭐ **VoxelMorph: A Learning Framework for Deformable Medical Image Registration** (Balakrishnan et al.) | IEEE TMI | 🔗 [arXiv](https://arxiv.org/abs/1809.05231) 💻 [GitHub](https://github.com/voxelmorph/voxelmorph) |
| 2019 | **DeepAtlas: Joint Semi-Supervised Learning of Image Registration and Segmentation** (Xu & Niethammer) | MICCAI 2019 | 🔗 [arXiv](https://arxiv.org/abs/1904.08465) 💻 [GitHub](https://github.com/uncbiag/DeepAtlas) |
| 2021 | **Segis-Net: Simultaneous Segmentation and Registration for Longitudinal Diffusion MRI** (Li et al.) | NeuroImage | 🔗 [arXiv](https://arxiv.org/abs/2012.14230) 💻 [GitLab](https://gitlab.com/blibli/segis-net) |
| 2022 | ⭐ **TransMorph: Transformer for Unsupervised Medical Image Registration** (Chen et al.) | Medical Image Analysis | 🔗 [arXiv](https://arxiv.org/abs/2111.10480) 💻 [GitHub](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) |
| 2022 | **SynthMorph: Learning Contrast-Invariant Registration Without Acquired Images** (Hoffmann et al.) | IEEE TMI | 🔗 [arXiv](https://arxiv.org/abs/2004.10282) 💻 [GitHub](https://github.com/voxelmorph/voxelmorph) |
| 2022 | **RSegNet: A Joint Learning Framework for Deformable Registration and Segmentation** (Qiu & Ren) | IEEE T-ASE | 🔗 [IEEE](https://ieeexplore.ieee.org/document/9462124/) | 
| 2023 | **Seq2Morph: Deep Learning Deformable Registration for Longitudinal Studies and Adaptive RT** (Lee et al.) | Medical Physics | 🔗 [DOI](https://doi.org/10.1002/mp.16026) |
| 2024 | **SAMReg: One Registration is Worth Two Segmentations** (Huang et al.) | MICCAI 2024 | 🔗 [arXiv](https://arxiv.org/abs/2405.10879) 💻 [GitHub](https://github.com/sqhuang0103/SAMReg) |

---

## Recurrent and Sequential Architectures

Architectures that process temporal sequences of medical images using recurrent units (ConvLSTM, GRU), neural ODEs, or sequential pipelines to capture disease progression patterns.

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2023 | **LCTformer: ConvLSTM Coordinated Longitudinal Transformer for Tumor Growth Prediction** (Ma et al.) | Computers in Biology and Medicine | 🔗 [DOI](https://doi.org/10.1016/j.compbiomed.2023.107313) |
| 2024 | **CAVE: Cerebral Artery-Vein Segmentation in Digital Subtraction Angiography** (Su et al.) | Computerized Medical Imaging and Graphics | 🔗 [arXiv](https://arxiv.org/abs/2208.02355) 💻 [GitHub](https://github.com/RuishengSu/CAVE_DSA) |
| 2024 | **DeepGrowth: Vestibular Schwannoma Growth Prediction from Longitudinal MRI by Time-Conditioned Neural Fields** (Chen et al.) | MICCAI 2024 | 🔗 [arXiv](https://arxiv.org/abs/2404.02614) 💻 [GitHub](https://github.com/cyjdswx/DeepGrowth) |
| 2024 | **Longitudinal Segmentation of MS Lesions via Temporal Difference Weighting** (Rokuss et al.) | MICCAI 2024 Workshop (LDTM) | 🔗 [arXiv](https://arxiv.org/abs/2409.13416) 💻 [GitHub](https://github.com/MIC-DKFZ/Longitudinal-Difference-Weighting) |

---

## Attention and Transformer-Based Temporal Fusion

Methods using attention mechanisms, Transformers, or memory networks to fuse information across timepoints for temporally consistent segmentation.

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2024 | **Dynamic-Guided Spatiotemporal Attention for Echocardiography Video Segmentation** (Lin et al.) | IEEE TMI | 🔗 [IEEE](https://ieeexplore.ieee.org/document/10535993) |
| 2024 | ⭐ **MemSAM: Taming Segment Anything Model for Echocardiography Video Segmentation** (Deng et al.) | CVPR 2024 (Oral) | 🔗 [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Deng_MemSAM_Taming_Segment_Anything_Model_for_Echocardiography_Video_Segmentation_CVPR_2024_paper.html) 💻 [GitHub](https://github.com/dengxl0520/MemSAM) |
| 2024 | **Self-Supervised Spatial-Temporal Transformer Fusion for Federated 4D Cardiovascular Segmentation** | Information Fusion | 🔗 [DOI](https://doi.org/10.1016/j.inffus.2024.102256) |
| 2025 | **CSTM: Continuous Spatio-Temporal Memory Networks for 4D Cardiac Cine MRI Segmentation** (Ye et al.) | WACV 2025 | 🔗 [arXiv](https://arxiv.org/abs/2410.23191) 💻 [GitHub](https://github.com/DeepTag/CSTM) |
| 2025 | **TAM: Motion-Enhanced Cardiac Anatomy Segmentation via Insertable Temporal Attention Module** (Hasan et al.) | arXiv | 🔗 [arXiv](https://arxiv.org/abs/2501.14929) 💻 [GitHub](https://github.com/kamruleee51/TAM) |
| 2025 | **Semi-Supervised Cine Cardiac MRI Segmentation via Joint Registration and Temporal Attention Perceiver** (Qin et al.) | Medical Physics | 🔗 [DOI](https://doi.org/10.1002/mp.70094) |

---

## Prior-Conditioned Segmentation

Approaches that condition the current segmentation on prior timepoint images, labels, or anatomical priors — especially prominent in adaptive radiotherapy where planning CTs/MRIs serve as strong priors.

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2022 | **Patient-Specific Daily Updated Deep Learning Auto-Segmentation for MRI-Guided Adaptive RT** (Li et al.) | Radiotherapy and Oncology | 🔗 [DOI](https://doi.org/10.1016/j.radonc.2022.11.004) |
| 2023 | **SPIRS: Joint Registration and Segmentation of Brain Metastases on MRI** (Patel et al.) | MLHC 2023 | 🔗 [PMLR](https://proceedings.mlr.press/v219/patel23a.html) |
| 2024 | **CloverNet: Leveraging Planning Annotations for Enhanced Procedural MR Segmentation in Adaptive RT** (De Benetti et al.) | MICCAI Workshop (CLIP 2024) | 🔗 [DOI](https://doi.org/10.1007/978-3-031-73083-2_1) |
| 2025 | **Enhancing Patient-Specific Segmentation for MR-Guided RT: A Framework Conditioned on Prior Segmentation** (De Benetti et al.) | Physics and Imaging in Radiation Oncology | 🔗 [DOI](https://doi.org/10.1016/j.phro.2025.100766) |
| 2025 | **Personalized Deep Learning Auto-Segmentation Models for Adaptive Fractionated MR-Guided RT** (Kawula et al.) | Medical Physics | 🔗 [DOI](https://doi.org/10.1002/mp.17580) |
| 2025 | **LesiOnTime: Joint Temporal and Clinical Modeling for Small Breast Lesion Segmentation in Longitudinal DCE-MRI** (Kamran et al.) | MICCAI Workshop (Deep-Breath 2025) | 🔗 [Springer](https://link.springer.com/chapter/10.1007/978-3-032-05559-0_33) 💻 [GitHub](https://github.com/cirmuw/LesiOnTime) |

---

## Self-Supervised Temporal Pretraining

Self-supervised and contrastive learning methods that leverage temporal structure in longitudinal data (e.g., ordering, disease progression, patient identity) for pretraining representations.

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2021 | ⭐ **LSSL: Longitudinal Self-Supervised Learning** (Zhao et al.) | Medical Image Analysis | 🔗 [arXiv](https://arxiv.org/abs/2006.06930) 💻 [GitHub](https://github.com/ZucksLiu/LSSL) |
| 2021 | **CL-TCI: Contrastive Learning with Temporal Correlated Medical Images** (Zeng et al.) | arXiv | 🔗 [arXiv](https://arxiv.org/abs/2109.03233) 💻 [GitHub](https://github.com/dewenzeng/CL-TCI) |
| 2021 | **MedAug: Contrastive Learning Leveraging Patient Metadata Improves Representations for Chest X-Ray** (Vu et al.) | MLHC 2021 | 🔗 [arXiv](https://arxiv.org/abs/2102.10663) 💻 [GitHub](https://github.com/stanfordmlgroup/MedAug) |
| 2022 | **LNE: Self-Supervised Learning of Neighborhood Embedding for Longitudinal MRI** (Ouyang et al.) | Medical Image Analysis (ext. of MICCAI 2021) | 🔗 [arXiv](https://arxiv.org/abs/2103.03840) 💻 [GitHub](https://github.com/ouyangjiahong/longitudinal-neighbourhood-embedding) |
| 2022 | **Local Spatiotemporal Representation Learning for Longitudinally-Consistent Neuroimage Analysis** (Ren et al.) | NeurIPS 2022 (Oral) | 🔗 [arXiv](https://arxiv.org/abs/2206.04281) 💻 [GitHub](https://github.com/mengweiren/longitudinal-representation-learning) |
| 2023 | **LSOR: Longitudinally-Consistent Self-Organized Representation Learning** (Ouyang et al.) | MICCAI 2023 | 🔗 [arXiv](https://arxiv.org/abs/2310.00213) 💻 [GitHub](https://github.com/ouyangjiahong/longitudinal-som-single-modality) |
| 2024 | **Spatiotemporal Representation Learning for Short and Long Medical Image Time Series** (Shen et al.) | MICCAI 2024 | 🔗 [arXiv](https://arxiv.org/abs/2403.07513) 💻 [GitHub](https://github.com/Leooo-Shen/tvrl) |
| 2024 | **Time-to-Event Pretraining for 3D Medical Imaging** (Huo et al.) | ICLR 2025 | 🔗 [arXiv](https://arxiv.org/abs/2411.09361) |
| 2025 | **ChronoCon: Chronological Contrastive Learning for Few-Shot Progression Assessment** | OpenReview (under review) | 🔗 [OpenReview](https://openreview.net/forum?id=c1UkGC3MVq) |

---

## Joint Change Detection and Segmentation

Methods that simultaneously detect longitudinal changes (new/growing/shrinking lesions, atrophy) and segment structures across timepoints.

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2023 | **SimU-Net: Liver Lesion Changes Analysis in Longitudinal CECT via Simultaneous Deep Learning Voxel Classification** (Szeskin et al.) | Medical Image Analysis | 🔗 [DOI](https://doi.org/10.1016/j.media.2022.102675) |
| 2023 | **Graph-Theoretic Automatic Lesion Tracking and Detection of Patterns in Longitudinal CT** (Di Veroli et al.) | MICCAI 2023 | 🔗 [DOI](https://doi.org/10.1007/978-3-031-43904-9_11) |
| 2024 | **Regional Deep Atrophy (RDA): Self-Supervised Learning for Alzheimer's Progression from Longitudinal MRI** (Dong et al.) | Imaging Neuroscience | 🔗 [arXiv](https://arxiv.org/abs/2304.04673) |
| 2024 | **GeoLongSeg: Geodesic Shape Regression Based Deep Learning for Longitudinal Hippocampal Atrophy** | NeuroImage: Clinical | 🔗 [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2213158224000627) |

---

## Foundation Models and Emerging Approaches

Adaptations of foundation models (SAM, SAM2, LLMs, diffusion models) for longitudinal or video-based medical segmentation, and vision-language temporal models.

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2023 | **BioViL-T: Learning to Exploit Temporal Structure for Biomedical Vision-Language Processing** (Bannur et al.) | CVPR 2023 | 🔗 [arXiv](https://arxiv.org/abs/2301.04558) 💻 [GitHub](https://github.com/microsoft/hi-ml) |
| 2023 | **SADM: Sequence-Aware Diffusion Model for Longitudinal Medical Image Generation** (Yoon et al.) | IPMI 2023 | 🔗 [arXiv](https://arxiv.org/abs/2212.08228) 💻 [GitHub](https://github.com/ubc-tea/SADM-Longitudinal-Medical-Image-Generation) |
| 2024 | **LLMSeg: LLM-Driven Multimodal Target Volume Contouring in Radiation Oncology** (Oh et al.) | Nature Communications | 🔗 [arXiv](https://arxiv.org/abs/2311.01908) 💻 [GitHub](https://github.com/tvseg/MM-LLM-RO) |
| 2024 | **Medical-SAM2: Segment Medical Images as Video via SAM 2** (Zhu et al.) | arXiv | 🔗 [arXiv](https://arxiv.org/abs/2408.00874) 💻 [GitHub](https://github.com/SuperMedIntel/Medical-SAM2) |
| 2024 | **TADM: Temporally-Aware Diffusion Model for Neurodegenerative Progression on Brain MRI** (Litrico et al.) | MICCAI 2024 | 🔗 [arXiv](https://arxiv.org/abs/2406.12411) 💻 [GitHub](https://github.com/MattiaLitrico/TADM-Temporally-Aware-Diffusion-Model-for-Neurodegenerative-Progression-on-Brain-MRI) |
| 2025 | ⭐ **MedSAM2: Segment Anything in 3D Medical Images and Videos** (Ma et al.) | arXiv | 🔗 [arXiv](https://arxiv.org/abs/2504.03600) 💻 [GitHub](https://github.com/bowang-lab/MedSAM2) |
| 2025 | **SAMed-2: Selective Memory Enhanced Medical Segment Anything Model** | MICCAI 2025 | 🔗 [arXiv](https://arxiv.org/abs/2507.03698) 💻 [GitHub](https://github.com/ZhilingYan/Medical-SAM-Bench) |
| 2025 | **TaDiff: Treatment-Aware Diffusion for Longitudinal MRI Generation and Glioma Growth Prediction** (Liu et al.) | IEEE TMI | 🔗 [arXiv](https://arxiv.org/abs/2309.05406) 💻 [GitHub](https://github.com/samleoqh/TaDiff-Net) |
| 2025 | **SAT: Large-Vocabulary Segmentation for Medical Images with Text Prompts** (Zhao et al.) | npj Digital Medicine | 🔗 [Nature](https://doi.org/10.1038/s41746-025-01964-w) 💻 [GitHub](https://github.com/zhaoziheng/SAT) |
| 2025 | **MediViSTA-SAM: Medical Video Segmentation via Temporal Fusion SAM Adaptation** (Kim et al.) | IEEE JBHI | 🔗 [arXiv](https://arxiv.org/abs/2309.13539) 💻 [GitHub](https://github.com/kimsekeun/MediViSTA-SAM) |
| 2025 | ⭐ **LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging** (Rokuss et al.) | CVPR 2025 | 🔗 [arXiv](https://arxiv.org/abs/2502.20985) 💻 [GitHub](https://github.com/MIC-DKFZ/LesionLocator) |

---

## Clinical Applications: Adaptive RT and Longitudinal Monitoring

Papers applying longitudinal segmentation to clinical tasks: adaptive radiotherapy, lung nodule malignancy, brain segmentation across timepoints, and longitudinal tumor monitoring.

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2021 | **Deformation Driven Seq2Seq Longitudinal Tumor and OAR Prediction for Radiotherapy** (Lee et al.) | Medical Physics | 🔗 [arXiv](https://arxiv.org/abs/2106.09076) |
| 2023 | ⭐ **SAMSEG-Long: Open-Source Tool for Longitudinal Whole-Brain and White Matter Lesion Segmentation** (Cerri et al.) | NeuroImage: Clinical | 🔗 [DOI](https://doi.org/10.1016/j.nicl.2023.103354) 💻 [GitHub](https://github.com/freesurfer/samseg) |
| 2023 | ⭐ **SynthSeg: Segmentation of Brain MRI of Any Contrast and Resolution Without Retraining** (Billot et al.) | Medical Image Analysis | 🔗 [DOI](https://doi.org/10.1016/j.media.2023.102789) 💻 [GitHub](https://github.com/BBillot/SynthSeg) |
| 2023 | **Prior CT Improves Deep Learning for Malignancy Risk Estimation of Screening-Detected Pulmonary Nodules** (Venkadesh et al.) | Radiology | 🔗 [DOI](https://doi.org/10.1148/radiol.223308) |
| 2024 | **DeepCAD-NLM-L: Enhancing Cancer Prediction in Screen-Detected Lung Nodules Using Time-Series Deep Learning** (Aslani et al.) | Computerized Medical Imaging and Graphics | 🔗 [arXiv](https://arxiv.org/abs/2203.16606) |

---

## Frameworks and Toolkits

General-purpose frameworks designed for longitudinal medical image segmentation.

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2024–2025 | ⭐ **LongiSeg: nnU-Net Extension for Longitudinal Medical Image Segmentation** (MIC-DKFZ) | — | 💻 [GitHub](https://github.com/MIC-DKFZ/LongiSeg) |

LongiSeg extends the widely-used **nnU-Net** framework with temporal feature merging modules (including the Difference Weighting Block). It is fully compatible with the nnU-Net ecosystem and was used in the winning solution of the MICCAI autoPET IV challenge. It supports multi-timepoint input for longitudinal segmentation tasks.

---

## Datasets and Challenges

Longitudinal and multi-timepoint medical imaging datasets and segmentation challenges relevant to this field.

| Name | Domain | Description | Links |
|------|--------|-------------|-------|
| **HNTS-MRG 2024** | Head & Neck Tumor (MRI) | 200 HNC cases, T2w MRI at pre-RT and mid-RT timepoints, with GTVp/GTVn segmentations. MICCAI 2024 satellite challenge. | 🔗 [arXiv](https://arxiv.org/abs/2411.18585) 💻 [GitHub](https://github.com/kwahid/HNTSMRG_2024) 🌐 [Challenge](https://hntsmrg24.grand-challenge.org/) |
| **NYUMets-Brain** | Brain Metastases (MRI) | 1,429+ patients, 8,000+ MRI studies (T1, T1ce, T2, FLAIR). World's largest longitudinal brain metastases dataset with clinical follow-up. | 🔗 [Nature Comms](https://doi.org/10.1038/s41467-024-52414-2) 💻 [GitHub](https://github.com/nyumets/nyumets) 🌐 [Website](https://nyumets.org/) |
| **autoPET/CT IV** | Whole-Body PET/CT & Longitudinal CT | Task 1: PET/CT lesion segmentation (1,600+ scans). Task 2: Longitudinal CT lesion segmentation (300+ melanoma cases, baseline + follow-up). MICCAI 2025 challenge. | 💻 [GitHub](https://github.com/lab-midas/autoPETCTIV) 🌐 [Challenge](https://autopet-iv.grand-challenge.org/) |
| **UCSF-ALPTDG** | Diffuse Glioma (MRI) | 298 patients, multimodal MRI (FLAIR, T1, T2, T1CE) at two consecutive follow-ups with voxelwise tumor subregion segmentations and longitudinal change labels. | 🔗 [DOI](https://doi.org/10.1148/ryai.230182) 💻 [GitHub (benchmarks)](https://github.com/rachitsaluja/UCSF-ALPTDG-benchmarks) 🌐 [Dataset](https://imagingdatasets.ucsf.edu/dataset/2) |
| **MSSEG-2** | Multiple Sclerosis (MRI) | 100 patients, longitudinal 3D FLAIR at two timepoints, focused on new lesion detection. MICCAI 2021 challenge. | 🌐 [Challenge](https://portal.fli-iam.irisa.fr/msseg-2/) |
| **BraTS 2024 Post-Treatment** | Post-Treatment Glioma (MRI) | First BraTS challenge on post-treatment MRI including resection cavities; relevant for longitudinal tumor monitoring. | 🔗 [arXiv](https://arxiv.org/abs/2405.18368) |

---

## Additional Recent Papers (2024–2025)

Important papers not covered in the main categories above, primarily from MICCAI 2024/2025, CVPR 2024/2025, and recent preprints.

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2024 | **LoCI-DiffCom: Longitudinal Consistency-Informed Diffusion Model for 3D Infant Brain Image Completion** (Zhu et al.) | MICCAI 2024 | 🔗 [arXiv](https://arxiv.org/abs/2405.10691) |
| 2024 | **A Unified Model for Longitudinal Multi-Modal Multi-View Prediction with Missingness** (Chen et al.) | MICCAI 2024 | 🔗 [MICCAI Proceedings](https://papers.miccai.org/miccai-2024/) |
| 2024 | **Longitudinally Consistent Individualized Prediction of Infant Cortical Morphological Development** (Yuan et al.) | MICCAI 2024 | — |
| 2025 | **TADM-3D: Temporally-Aware Diffusion Model with Bidirectional Temporal Regularisation** (Litrico et al.) | arXiv | 🔗 [arXiv](https://arxiv.org/abs/2509.03141) 💻 [GitHub](https://github.com/MattiaLitrico/TADM-Temporally-Aware-Diffusion-Model-for-Neurodegenerative-Progression-on-Brain-MRI) |
| 2025 | **BrLP: Brain Latent Progression — Individual-Based Spatiotemporal Disease Progression via Latent Diffusion** (Puglisi et al.) | Medical Image Analysis | 🔗 [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1361841525002816) |
| 2025 | **Text-Promptable Propagation for Referring Medical Image Sequence Segmentation** (Yuan et al.) | arXiv | 🔗 [arXiv](https://arxiv.org/abs/2502.11093) |
| 2025 | **Test Time Training for 4D Medical Image Interpolation** | arXiv | 🔗 [arXiv](https://arxiv.org/abs/2502.02341) |
| 2025 | **Few-Shot Medical Video Object Segmentation via Spatiotemporal Consistency Relearning** | arXiv | 🔗 [arXiv](https://arxiv.org/abs/2503.14958) |

---

## Related Workshops

- **LDTM: Longitudinal Disease Tracking and Modelling** — Recurring MICCAI workshop (2024, 2025). Published in Springer LNCS vol. 15401. Dedicated to methods for tracking and modeling disease over time in medical imaging.

---

## Code Availability Summary

Across all **54 primary entries**, **31 have publicly available code repositories**. The table below summarizes code availability by category:

| Category | Papers | With Code | Percentage |
|----------|--------|-----------|------------|
| Registration-Based | 8 | 6 | 75% |
| Recurrent/Sequential | 4 | 3 | 75% |
| Attention/Transformer | 6 | 3 | 50% |
| Prior-Conditioned | 6 | 1 | 17% |
| Self-Supervised Pretraining | 9 | 7 | 78% |
| Change Detection | 4 | 0 | 0% |
| Foundation Models | 12 | 11 | 92% |
| Clinical Applications | 5 | 2 | 40% |

Foundation models and self-supervised pretraining methods have the highest code release rates, while prior-conditioned segmentation (mostly clinical/RT papers) and change detection methods have the lowest.

---

## How to contribute

This list aims to be a living resource. To add a paper, please ensure it meets the following criteria:

1. The method explicitly uses **temporal/longitudinal information** from multiple imaging timepoints
2. The application domain is **medical image segmentation** (or closely related: registration for segmentation, change detection, temporal consistency)
3. Include: title, authors, year, venue, paper link (arXiv or DOI preferred), and code link if available

---

*Last updated: February 2026*