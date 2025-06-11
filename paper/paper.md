---
title: 'PyMilo: A Python Library for ML I/O'
tags:
  - Python
  - Machine Learning 
  - Model Deployment
  - Model Serialization
  - Transparency
  - MLOPS
authors:
  - name: AmirHosein Rostami
    orcid: 0009-0000-0638-2263
    corresponding: true
    affiliation: 1 
  - name: Sepand Haghighi
    orcid: 0000-0001-9450-2375
    corresponding: false
    affiliation: 1
  - name: Sadra Sabouri
    orcid: 0000-0003-1047-2346
    corresponding: false
    affiliation: 1
  - name: Alireza Zolanvari
    orcid: 0000-0003-2367-8343
    corresponding: false
    affiliation: 1
affiliations:
 - name: Open Science Lab
   index: 1

date: 10 June 2025
bibliography: paper.bib
---

# Summary
PyMilo is an open-source Python package that addresses the limitations of existing machine learning (ML) model storage formats by providing a transparent, reliable, end-to-end, and safe method for exporting and deploying trained models. Current tools rely on black-box or executable formats that obscure internal model structures, making them difficult to audit, verify, or safely share. Others apply structural transformations during export that may degrade predictive performance and reduce the model to a limited inference-only interface. In contrast, PyMilo serializes models in a transparent human-readable format that preserves end-to-end model fidelity and enables reliable, safe, and interpretable exchange. This package is designed to make the preservation and reuse of trained ML models safer, more interpretable, and easier to manage across different stages of the workflow.

# Statement of Need
Modern machine learning development is largely centered around the Python ecosystem, which has become a dominant platform for building and training models due to its rich libraries and community support [@Raschka2020]. However, once a model is trained, sharing or deploying it securely and transparently remains a significant challenge. This issue is especially important in high-stakes domains such as healthcare, where ensuring model accountability and integrity is critical [@Garbin2022].
In such settings, any lack of clarity about a model’s internal logic or origin can reduce trust in its predictions. Researchers have increasingly emphasized that greater transparency in AI systems is critical for maintaining user trust and protecting privacy in machine learning applications [@Bodimani2024].

Despite ongoing concerns around transparency and safety, the dominant approach for exchanging Python-trained models remains ad hoc binary serialization, most commonly through Python’s `pickle` module or its variant `joblib`. These formats allow developers to store complex model objects with minimal effort, but they were never designed with security or human interpretability in mind. In fact, loading a pickle file will execute arbitrary code contained within it, a known vulnerability that can be exploited if the file is maliciously crafted [@Brownlee2018]. While this method, whether using `pickle` or `joblib`, preserves full model fidelity within the Python ecosystem, it poses serious security risks and lacks transparency, as the serialized files are opaque binary blobs that cannot be inspected without loading. Furthermore, compatibility is fragile because pickled models often depend on specific library versions, which may hinder long-term reproducibility [@Brownlee2018].

To improve portability across environments, several standardized model interchange formats have been developed alongside `pickle`. Most notably, ONNX (Open Neural Network Exchange) and PMML (Predictive Model Markup Language) convert trained models into framework-neutral representations [@Verma2023; @ONNX2017], enabling deployment in diverse systems without relying on the original training code. ONNX uses a graph-based structure built from primitive operators (e.g., linear transforms, activations), while PMML provides an XML-based specification for traditional models like decision trees and regressions.


Despite the variety of tools available (see Table \ref{toolcomparison}), there remains a conspicuous gap in machine learning model storage and exchange methods. No existing solution fully satisfies the core requirements of security, transparency, and end-to-end fidelity while maintaining broad applicability.

**Table 1**: Comparison of PyMilo with existing model serialization tools.[]{#toolcomparison}

| Package           | Transparent | Multi-Framework | End-to-End Preservation | Secure |
|------------------|-------------|------------------|--------------------------|--------|
| **Pickle**        | No          | Yes              | Yes                      | No     |
| **Joblib**        | No          | Yes              | Yes                      | No     |
| **ONNX**          | No          | Yes              | No                       | Yes    |
| **PMML**          | Yes         | No               | No                       | Yes    |
| **SKOPS**         | No          | No               | Yes                      | Yes    |
| **TensorFlow.js** | Yes         | No               | No                       | Yes    |
| **PyMilo**        | Yes         | Yes              | Yes                      | Yes    |

PyMilo is proposed to address the above gaps. It is an open-source Python library designed as an end-to-end solution for exporting and importing machine learning models in a safe, non-executable, and human-readable format such as JSON. PyMilo serializes trained models from machine learning frameworks into a transparent format and deserializes them back into the same original model, preserving structure, functionality, and behavior. PyMilo fully recovers the original model without structural changes, which does not affect inference time or model performance. The approach ensures that models can be transported to any target device and imported without requiring additional dependencies, allowing seamless execution in inference mode. This provides a general solution for creating human-readable, transparent, and safe machine learning models that can be easily shared, inspected, and deployed. PyMilo benefits a wide range of stakeholders, including machine learning engineers, data scientists, and AI practitioners, by facilitating the development of more transparent and accountable AI systems. Furthermore, researchers working on transparent AI [@rauker2023toward], user privacy in ML [@bodimani2024assessing], and safe AI [@macrae2019governing] can use PyMilo as a framework that provides transparency and safety in the machine learning environment.

# References