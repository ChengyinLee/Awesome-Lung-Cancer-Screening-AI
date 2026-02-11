# 🫁 Awesome Lung Cancer Screening AI (2020–2025)

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A curated list of deep learning papers for AI-driven lung cancer screening, covering nodule detection, malignancy prediction, risk assessment, segmentation, vision-language models, foundation models, and clinical deployment. Focused on **2020–2025**.

---

## Table of Contents

- [1. Pulmonary Nodule Detection](#1-pulmonary-nodule-detection)
- [2. Nodule Malignancy Prediction & Classification](#2-nodule-malignancy-prediction--classification)
  - [2.1 Single-Timepoint Models](#21-single-timepoint-models)
  - [2.2 Longitudinal / Temporal Models](#22-longitudinal--temporal-models)
- [3. Lung Cancer Risk Prediction](#3-lung-cancer-risk-prediction)
  - [3.1 Sybil & Validations](#31-sybil--validations)
  - [3.2 Other Risk Prediction Models](#32-other-risk-prediction-models)
- [4. Nodule & Lung Segmentation](#4-nodule--lung-segmentation)
- [5. Foundation Models for Lung CT](#5-foundation-models-for-lung-ct)
- [6. Vision-Language Models & LLMs](#6-vision-language-models--llms)
- [7. Federated Learning](#7-federated-learning)
- [8. Diffusion Models & Generative Methods](#8-diffusion-models--generative-methods)
- [9. Explainability & Uncertainty](#9-explainability--uncertainty)
- [10. Clinical Trials, Deployment & AI-as-Reader Studies](#10-clinical-trials-deployment--ai-as-reader-studies)
- [11. Datasets & Benchmarks](#11-datasets--benchmarks)
- [12. Surveys & Reviews](#12-surveys--reviews)
- [Key Clinical Milestones](#key-clinical-milestones)
- [FDA/CE-Cleared AI Tools](#fdace-cleared-ai-tools)

---

## 1. Pulmonary Nodule Detection

| Title | Year | Venue | Paper | Code |
|-------|------|-------|-------|------|
| nnDetection: A Self-configuring Method for Medical Object Detection | 2021 | MICCAI | [arXiv](https://arxiv.org/abs/2106.00817) | [GitHub](https://github.com/MIC-DKFZ/nnDetection) |
| SANet: A Slice-Aware Network for Pulmonary Nodule Detection | 2021 | IEEE TPAMI | [PubMed](https://pubmed.ncbi.nlm.nih.gov/33687839/) | [GitHub](https://github.com/mj129/SANet) |
| SCPM-Net: An Anchor-free 3D Lung Nodule Detection Network Using Sphere Representation and Center Points Matching | 2022 | Medical Image Analysis | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1361841521003327) | [GitHub](https://github.com/HuluoluoTech/SCPM-Net) |
| LSSANet: A Long Short Slice-Aware Network for Pulmonary Nodule Detection | 2022 | MICCAI | [MICCAI](https://conferences.miccai.org/2022/papers/309-Paper2658.html) | — |
| TiCNet: Transformer in Convolutional Neural Network for Pulmonary Nodule Detection on CT Images | 2024 | Journal of Digital Imaging | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10976926/) | — |
| Deep Learning for the Detection of Benign and Malignant Pulmonary Nodules in Non-screening Chest CT Scans | 2023 | Communications Medicine | [Nature](https://www.nature.com/articles/s43856-023-00388-5) | — |
| NODE21: Nodule Detection on Chest Radiographs (Detection + Generation Challenge) | 2021 | Grand Challenge | [Paper](https://node21.grand-challenge.org/) | [Zenodo](https://zenodo.org/records/5548363) |

## 2. Nodule Malignancy Prediction & Classification

### 2.1 Single-Timepoint Models

| Title | Year | Venue | Paper | Code |
|-------|------|-------|-------|------|
| Deep Learning for Malignancy Risk Estimation of Pulmonary Nodules Detected at Low-Dose Screening CT | 2021 | Radiology | [PubMed](https://pubmed.ncbi.nlm.nih.gov/34003056/) | [Grand Challenge](https://grand-challenge.org/algorithms/pulmonary-nodule-malignancy-prediction/) |
| Assessing the Accuracy of a Deep Learning Method to Risk Stratify Indeterminate Pulmonary Nodules (LCP-CNN) | 2020 | AJRCCM | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7365375/) | — |
| Lung Cancer Prediction by Deep Learning to Identify Benign Lung Nodules (LUCINDA Validation) | 2021 | Lung Cancer | [PubMed](https://pubmed.ncbi.nlm.nih.gov/33556604/) | — |
| Deep Learning Models for Predicting Malignancy Risk in CT-Detected Pulmonary Nodules: A Systematic Review and Meta-analysis | 2024 | Lung | [Springer](https://link.springer.com/article/10.1007/s00408-024-00706-1) | — |
| Data-driven Risk Stratification and Precision Management of Pulmonary Nodules Detected on Chest CT (C-Lung-RADS) | 2024 | Nature Medicine | [Nature](https://www.nature.com/articles/s41591-024-03211-3) | — |
| AI-Enhanced Diagnostic Model for Pulmonary Nodule Classification | 2024 | Frontiers in Oncology | [Frontiers](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2024.1417753/full) | — |

### 2.2 Longitudinal / Temporal Models

| Title | Year | Venue | Paper | Code |
|-------|------|-------|-------|------|
| Prior CT Improves Deep Learning for Malignancy Risk Estimation of Screening-detected Pulmonary Nodules | 2023 | Radiology | [RSNA](https://pubs.rsna.org/doi/abs/10.1148/radiol.223308) | — |
| Enhancing Cancer Prediction in Challenging Screen-Detected Incident Lung Nodules Using Time-Series Deep Learning (DeepCAD-NLM-L) | 2024 | Computerized Medical Imaging and Graphics | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0895611124000764) | — |
| Prediction of Lung Cancer Risk at Follow-up Screening with Low-Dose CT: A Training and Validation Study of a Deep Learning Method | 2020 | Lancet Digital Health | [PubMed](https://pubmed.ncbi.nlm.nih.gov/32864596/) | — |
| Assessments of Lung Nodules by an AI Chatbot Using Longitudinal CT Images (GPT-4o Evaluation) | 2025 | Cell Reports Medicine | [Cell](https://www.cell.com/cell-reports-medicine/fulltext/S2666-3791(25)00061-8) | — |

## 3. Lung Cancer Risk Prediction

### 3.1 Sybil & Validations

| Title | Year | Venue | Paper | Code |
|-------|------|-------|-------|------|
| Sybil: A Validated Deep Learning Model to Predict Future Lung Cancer Risk From a Single Low-Dose Chest CT | 2023 | Journal of Clinical Oncology | [JCO](https://ascopubs.org/doi/10.1200/JCO.22.01345) | [GitHub](https://github.com/reginabarzilaygroup/Sybil) |
| MA02.11 Validation of the Sybil Deep Learning Lung Cancer Risk Prediction Model in Three Independent Screening Studies | 2024 | Journal of Thoracic Oncology | [JTO](https://www.jto.org/article/S1556-0864(24)00980-8/fulltext) | — |
| External Testing of a Deep Learning Model for Lung Cancer Risk from Low-Dose Chest CT (Japan Cohort) | 2025 | Radiology | [RSNA](https://pubs.rsna.org/doi/10.1148/radiol.243393) | — |
| WCLC 2025: Validation of Sybil in Predominantly Black Population at Urban Safety-Net Hospital | 2025 | WCLC | [ecancer](https://ecancer.org/en/news/27045-wclc-2025-study-validates-ai-lung-cancer-risk-model-sybil-in-predominantly-black-population-at-urban-safety-net-hospital) | — |
| Sybil AI Model Demonstrates Strong Performance in Predicting Lung Cancer Risk with Minimal Bias | 2025 | ILCN/WCLC | [ILCN](https://www.ilcn.org/sybil-ai-model-demonstrates-strong-performance-in-predicting-lung-cancer-risk-with-minimal-bias/) | — |
| Predicting Future Lung Cancer Risk from a Single Low-Dose CT Using Deep Learning | 2023 | PMC Commentary | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10240246/) | — |

### 3.2 Other Risk Prediction Models

| Title | Year | Venue | Paper | Code |
|-------|------|-------|-------|------|
| End-to-End Lung Cancer Screening with 3D Deep Learning on Low-Dose Chest CT (Google Health) | 2019 | Nature Medicine | [Nature](https://www.nature.com/articles/s41591-019-0447-x) | — |
| Assistive AI in Lung Cancer Screening: A Retrospective Multinational Study in the United States and Japan (Google Multi-Reader) | 2024 | Radiology: AI | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11140517/) | — |
| Deep Learning Using Chest Radiographs to Identify High-Risk Smokers for Lung Cancer Screening CT (CXR-LC) | 2020 | Annals of Internal Medicine | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9200444/) | — |

## 4. Nodule & Lung Segmentation

| Title | Year | Venue | Paper | Code |
|-------|------|-------|-------|------|
| A Smart Hybrid Framework for 3D Lung Nodule Segmentation Using CT (TransAtenNet) | 2025 | SN Computer Science | [Springer](https://link.springer.com/article/10.1007/s42979-025-04558-1) | — |
| A Lung Nodule Segmentation Model Based on the Transformer with Multiple Thresholds and Coordinate Attention (MCAT-Net) | 2024 | Scientific Reports | [Nature](https://www.nature.com/articles/s41598-024-82877-8) | — |
| Lung Nodule Segmentation in LDCT: Modified 3D nnUNet with Unified Focal Loss | 2023 | ICECET | [NCKU](https://researchoutput.ncku.edu.tw/en/publications/lung-nodule-segmentation-in-ldct-modified-3d-nnunet-with-unified-) | — |
| Federated Lung Nodule Segmentation Using a Hybrid Transformer–U-Net Architecture | 2025 | Scientific Reports | [Nature](https://www.nature.com/articles/s41598-026-35243-9) | — |

## 5. Foundation Models for Lung CT

| Title | Year | Venue | Paper | Code |
|-------|------|-------|-------|------|
| Medical Multimodal Multitask Foundation Model for Lung Cancer Screening (M3FM) | 2025 | Nature Communications | [Nature](https://www.nature.com/articles/s41467-025-56822-w) | — |
| A Lung CT Vision Foundation Model Facilitating Disease Diagnosis and Medical Imaging (LCTfound) | 2025 | Nature Communications | [Nature](https://www.nature.com/articles/s41467-025-66620-z) | — |
| A Computationally Frugal, Open-Source Chest CT Foundation Model for Thoracic Disease Detection in Lung Cancer Screening Programmes (TANGERINE) | 2025 | Communications Medicine | [Nature](https://www.nature.com/articles/s43856-025-01328-1) | — |
| Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography (CT-CLIP / CT-RATE) | 2024 | arXiv / NeurIPS | [arXiv](https://arxiv.org/abs/2403.17834) | [GitHub](https://github.com/ibrahimethemhamamci/CT-CLIP) |

## 6. Vision-Language Models & LLMs

| Title | Year | Venue | Paper | Code |
|-------|------|-------|-------|------|
| From Image to Report: Automating Lung Cancer Screening Interpretation and Reporting with Vision-Language Models (LUMEN) | 2025 | Radiology | [PubMed](https://pubmed.ncbi.nlm.nih.gov/41083099/) | — |
| Development and Validation of a Dynamic-Template-Constrained Large Language Model for Generating Fully-Structured Radiology Reports | 2024 | arXiv | [arXiv](https://arxiv.org/abs/2409.18319) | — |
| Reasoning Language Model for Personalized Lung Cancer Screening | 2025 | arXiv | [arXiv](https://arxiv.org/pdf/2509.06169) | — |
| Opportunities and Challenges in Lung Cancer Care in the Era of Large Language Models and Vision Language Models | 2025 | Translational Lung Cancer Research | [TLCR](https://tlcr.amegroups.org/article/view/100592/html) | — |
| Vision-Language Model-Based Semantic-Guided Imaging Biomarker for Lung Nodule Malignancy Prediction | 2025 | arXiv | [arXiv](https://arxiv.org/html/2504.21344v3) | — |

## 7. Federated Learning

| Title | Year | Venue | Paper | Code |
|-------|------|-------|-------|------|
| ARGOS: Federated Deep Learning Infrastructure Across 4 Continents, 8 Countries, 12 Hospitals for Lung Cancer GTV Segmentation | 2025 | JMIR AI | [JMIR](https://ai.jmir.org/2025/1/e60847/PDF) | — |
| Advancing Breast, Lung and Prostate Cancer Research with Federated Learning: A Systematic Review | 2025 | npj Digital Medicine | [Nature](https://www.nature.com/articles/s41746-025-01591-5) | — |
| Predicting Treatment Response in Multicenter Non-Small Cell Lung Cancer Patients Based on Federated Learning | 2024 | BMC Cancer | [BMC](https://bmccancer.biomedcentral.com/articles/10.1186/s12885-024-12456-7) | — |

## 8. Diffusion Models & Generative Methods

| Title | Year | Venue | Paper | Code |
|-------|------|-------|-------|------|
| DiffusionCT: Latent Diffusion Model for CT Image Standardization | 2024 | AMIA | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10785850/) | — |
| Lung-DDPM+: Efficient Thoracic CT Image Synthesis Using Diffusion Probabilistic Model | 2025 | arXiv | [arXiv](https://arxiv.org/html/2508.09327v2) | — |

## 9. Explainability & Uncertainty

| Title | Year | Venue | Paper | Code |
|-------|------|-------|-------|------|
| Enhancing a Deep Learning Model for Pulmonary Nodule Malignancy Risk Estimation in Chest CT with Uncertainty Estimation | 2024 | European Radiology | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11399205/) | — |
| Leveraging Expert Input for Robust and Explainable AI-Assisted Lung Cancer Detection in Chest X-rays (ClinicXAI) | 2024 | arXiv | [arXiv](https://arxiv.org/abs/2403.19444) | — |
| A Critical Review of Explainable Deep Learning in Lung Cancer Diagnosis | 2025 | Artificial Intelligence Review | [Springer](https://link.springer.com/article/10.1007/s10462-025-11445-x) | — |
| Economic Viability of AI Assistance in Lung Cancer Screening Using CT | 2024 | Computational and Systems Oncology | [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/cso2.70000) | — |

## 10. Clinical Trials, Deployment & AI-as-Reader Studies

| Title | Year | Venue | Paper | Code |
|-------|------|-------|-------|------|
| Feasibility of AI as First Reader in the 4-IN-THE-LUNG-RUN Lung Cancer Screening Trial | 2025 | European Journal of Cancer | [EJC](https://www.ejcancer.com/article/S0959-8049(24)01821-5/fulltext) | — |
| MA02.07 AI as Primary Reader in 4-IN-THE-LUNG-RUN: Impact on Negative Misclassification and Referral Rate | 2024 | JTO (WCLC Abstract) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1556086424009778) | — |
| Histological Proven AI Performance in the UKLS CT Lung Cancer Screening Study: Potential for Workload Reduction | 2025 | European Journal of Cancer | [EJC](https://www.ejcancer.com/article/S0959-8049(25)00105-4/fulltext) | — |
| Low-Dose CT for Lung Cancer Screening in a High-Risk Population (SUMMIT Study) | 2025 | Lancet Oncology | [Lancet](https://www.thelancet.com/journals/lanonc/article/PIIS1470-2045(25)00082-8/fulltext) | — |
| Lunit INSIGHT CXR Excels in Lung Nodule Detection — Head-to-Head Study (Project AIR) | 2024 | Radiology | [PR Newswire](https://www.prnewswire.com/news-releases/lunit-insight-cxr-excels-in-lung-nodule-detection---exceptional-performance-in-head-to-head-study-published-in-radiology-302036747.html) | — |
| Artificial Intelligence for Low-Dose CT Lung Cancer Screening: Comparison of Utilization Scenarios | 2025 | AJR | [AJR](https://ajronline.org/doi/pdf/10.2214/AJR.25.32829?download=true) | — |
| Artificial Intelligence for Detection and Characterization of Pulmonary Nodules in Lung Cancer CT Screening: Ready for Practice? | 2021 | Translational Lung Cancer Research | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8182724/) | — |

## 11. Datasets & Benchmarks

| Dataset | Size | Year | Description | Link |
|---------|------|------|-------------|------|
| NLST (National Lung Screening Trial) | 53,000+ subjects, 75,000+ LDCT | Ongoing | Pathology-confirmed outcomes; gold standard for risk prediction | [NCI](https://cdas.cancer.gov/nlst/) |
| LIDC-IDRI | 1,018 CTs | 2011 | 4-radiologist annotations; most widely used detection benchmark | [TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/) |
| LUNA16 | 888 CTs (subset of LIDC-IDRI) | 2016 | Standardized detection challenge | [Grand Challenge](https://luna16.grand-challenge.org/) |
| PN9 | 8,798 CTs, 40,439 nodules | 2021 | 9 nodule categories; largest diversity dataset | [SANet Paper](https://pubmed.ncbi.nlm.nih.gov/33687839/) |
| NLSTseg | 605 CTs, 715 lesion annotations | 2025 | Pixel-level segmentation extension of NLST | [Nature](https://www.nature.com/articles/s41597-025-05742-x) |
| NODE21 | 4,882 CXRs | 2021 | CXR nodule detection + generation challenge | [Grand Challenge](https://node21.grand-challenge.org/) |
| CT-RATE | 25,692 3D chest CTs + reports | 2024 | First open-source 3D CT-text paired dataset | [HuggingFace](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) |
| Duke LCSD | 3,000+ CTs | 2024 | Contemporary CT technology screening dataset | — |

## 12. Surveys & Reviews

| Title | Year | Venue | Paper | Code |
|-------|------|-------|-------|------|
| Pulmonary Nodule Detection, Segmentation and Classification Using Deep Learning: A Comprehensive Literature Review | 2023 | MDPI | [MDPI](https://www.mdpi.com/2673-7426/4/3/111) | — |
| Progress and Challenges of Artificial Intelligence in Lung Cancer Clinical Translation | 2025 | npj Precision Oncology | [Nature](https://www.nature.com/articles/s41698-025-00986-7) | — |
| AI-Enabled Lung Cancer Diagnosis and Treatment in China: Challenges, Opportunities, and Future Prospects | 2024 | ACM ISAIMS | [ACM](https://dl.acm.org/doi/10.1145/3706890.3706928) | — |
| A Systematic Review of AI Performance in Lung Cancer Detection on CT Thorax | 2025 | PMC | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12250385/) | — |
| Artificial Intelligence and Machine Learning in Lung Cancer: Advances in Imaging, Detection, and Prognosis | 2025 | Cancers (MDPI) | [MDPI](https://www.mdpi.com/2072-6694/17/24/3985) | — |
| CT-based AI in Lung Disease — COPD | 2024 | MedComm – Future Medicine | [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/mef2.73) | — |
| Screening for Lung Cancer: 2023 Guideline Update from the American Cancer Society | 2024 | CA: A Cancer Journal for Clinicians | [Wiley](https://acsjournals.onlinelibrary.wiley.com/doi/10.3322/caac.21811) | — |

---

## Key Clinical Milestones

| Year | Milestone |
|------|-----------|
| 2020.01 | **NELSON Trial** full results published in NEJM — LDCT reduces lung cancer mortality by 24% in men |
| 2021.03 | **USPSTF** updates screening guidelines: age lowered to 50, pack-years to 20 |
| 2022.02 | **CMS/Medicare** expands coverage to match USPSTF 2021 criteria |
| 2022.12 | **EU Council** recommends lung cancer screening for the first time |
| 2023.01 | **Sybil** published in JCO — single LDCT predicts 1–6 year cancer risk |
| 2024 | **ACS** updates screening guideline to age 50–80 with ≥20 pack-years |
| 2024 | **C-Lung-RADS** published in Nature Medicine for Chinese population |
| 2025 | **4-ITLR & UKLS** trials demonstrate AI-as-first-reader feasibility (60–79% workload reduction) |
| 2025 | **M3FM, LCTfound, TANGERINE** — foundation model era begins |
| ~2026.02 | **eyonis LCS** (Median Technologies) receives FDA 510(k) — first CADe/CADx combo for screening |

## FDA/CE-Cleared AI Tools

| Product | Company | Clearance | Year | Function |
|---------|---------|-----------|------|----------|
| InferRead Lung CT.AI | Infervision | FDA 510(k) + CE | 2020 | Lung segmentation, nodule detection & classification |
| Virtual Nodule Clinic | Optellum | FDA 510(k) | 2021 | IPN risk stratification (LCP score) |
| INSIGHT CXR | Lunit | FDA 510(k) + CE | 2021 | CXR AI triage (AUC 0.93 in Project AIR) |
| Veye Lung Nodules | Aidence / DeepHealth | CE IIb (EU MDR) | 2021 | Nodule detection, volumetry, growth tracking |
| qXR LN | Qure.ai | FDA cleared | 2024 | CXR lung nodule detection |
| qCT LN Quant | Qure.ai | FDA 510(k) | 2024 | CT nodule quantification (volume, VDT, Brock) |
| eyonis LCS | Median Technologies | FDA 510(k) | ~2026 | CADe + CADx combo (sensitivity 93.3%, specificity 92.4%) |
| RevealAI-Lung | RevealDx | FDA + EU MDR | ~2025 | Malignancy similarity index (mSI score) |

---

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to add papers, correct information, or suggest new categories.

## Citation

If you find this list useful for your research, please consider starring ⭐ this repository.

## License

This project is licensed under the MIT License.
