---
title: 'PyMilo: A Python Library for ML I/O'
tags:
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
PyMilo is an open-source Python package that addresses the limitations of existing machine learning (ML) model storage formats by providing a transparent, reliable, end-to-end, and safe method for exporting and deploying trained models. 
Current tools rely on black-box or executable formats that obscure internal model structures, making them difficult to audit, verify, or safely share. 
Others apply structural transformations during export that may degrade predictive performance and reduce the model to a limited inference-only interface. 
In contrast, PyMilo serializes models in a transparent human-readable format that preserves end-to-end model fidelity and enables reliable, safe, and interpretable exchange. 
This package is designed to make the preservation and reuse of trained ML models safer, more interpretable, and easier to manage across different stages of the ML workflow.

# Statement of Need
Modern machine learning development is largely centered around the Python ecosystem, which has become a dominant platform for building and training models due to its rich libraries and community support [@Raschka2020]. 
However, once a model is trained, sharing or deploying it securely and transparently remains a significant challenge. This issue is especially important in high-stake domains such as healthcare, where ensuring model accountability and integrity is critical [@Garbin2022].
In such settings, any lack of clarity about a model’s internal logic or origin can reduce trust in its predictions. Researchers have increasingly emphasized that greater transparency in AI systems is critical for maintaining user trust and protecting privacy in machine learning applications [@bodimani2024assessing].

Despite ongoing concerns around transparency and safety, the dominant approach for exchanging pretrained models remains ad hoc binary serialization, most commonly through Python’s `pickle` module or its variant `joblib`. 
These formats allow developers to store complex model objects with minimal effort, but they were never designed with security or human interpretability in mind. In fact, loading a pickle file may execute arbitrary code contained within it, a known vulnerability that can be exploited if the file is maliciously crafted [@Brownlee2018]. 
While these methods preserves full model fidelity within the Python ecosystem, it poses serious security risks and lacks transparency, as the serialized files are opaque binary blobs that cannot be inspected without loading. 
Furthermore, compatibility is fragile because pickled models often depend on specific library versions, which may hinder long-term reproducibility [@Brownlee2018].

To improve portability across environments, several standardized model interchange formats have been developed alongside `pickle`. 
Most notably, Open Neural Network Exchange (ONNX) and Predictive Model Markup Language (PMML) convert trained models into framework-agnostic representations [@Verma2023; @ONNX2017], enabling deployment in diverse systems without relying on the original training code. 
ONNX uses a graph-based structure built from primitive operators (e.g., linear transforms, activations), while PMML provides an XML-based specification for traditional models like decision trees and regressions.

Although these formats enhance security by avoiding executable serialization, they introduce compatibility and fidelity challenges. 
Exporting complex pipelines to ONNX or PMML often leads to structural approximations, missing metadata, or unsupported components, especially for customized models [@Guazzelli2009; @Wang2020]. 
As a result, the exported model may differ in behavior, resulting in performance degradation or loss of accuracy. 
For example Wang et. al. reported accuracy drops of up to 10 to 15 percent after exporting models to ONNX in certain scenarios [@Wang2020]. This highlights the risk of behavioral drift between the original and exported versions.

Beyond concerns about end-to-end model preservation, ONNX and PMML also present limitations in transparency, scope, and reversibility. ONNX uses a binary protocol buffer format that is not human-readable, which limits interpretability and makes auditing difficult. 
PMML, although readable, is verbose and narrowly scoped, supporting only a limited subset of scikit-learn models. Moreover, PMML does not provide a way to restore exported models back into Python, making it a one-way format unsuitable for end-to-end workflows.

Other tools have been developed to address specific use cases, though they remain limited in scope. 
SKOPS improves the safety of scikit-learn model storage by avoiding executable serialization and enabling limited inspection of model contents [@Noyan2023]. 
However, it supports only scikit-learn models, lacks compatibility with other frameworks, and does not provide a fully transparent or human-readable structure. 
TensorFlow.js targets JavaScript environments by converting TensorFlow or Keras models into JSON and binary weight files for browser-based execution [@TFJS2018]. 
This process requires significant modifications to the original model architecture, which often leads to compatibility issues, degraded performance, and changes in inference time. 
Models from other frameworks, such as scikit-learn or PyTorch, must be re-implemented or retrained in TensorFlow to be exported. 
Additionally, running complex models in JavaScript runtimes introduces memory and speed limitations, making deployment of large neural networks prohibitively slow or even infeasible in the browser context [@NerdCorner2025].

In summary, current solutions force practitioners into a trade-offs between security, transparency, end-to-end fidelity, and performance preservation (see Table 1). 
The machine learning community still lacks a safe and transparent end-to-end model serialization framework through which users can securely share models, inspect them easily, and accurately reconstruct them for use across diverse frameworks and environments.

**Table 1**: Comparison of PyMilo with existing model serialization tools.

| Package           | Transparent | Multi-Framework | End-to-End Preservation | Secure |
|------------------|-------------|------------------|--------------------------|--------|
| **Pickle**        | No          | Yes              | Yes                      | No     |
| **Joblib**        | No          | Yes              | Yes                      | No     |
| **ONNX**          | No          | Yes              | No                       | Yes    |
| **PMML**          | Yes         | No               | No                       | Yes    |
| **SKOPS**         | No          | No               | Yes                      | Yes    |
| **TensorFlow.js** | Yes         | No               | No                       | Yes    |
| **PyMilo**        | Yes         | Yes              | Yes                      | Yes    |

PyMilo is proposed to address the above gaps. It is an open-source Python library that provides an end-to-end solution for exporting and importing machine learning models in a safe, non-executable, and human-readable format such as JSON. PyMilo serializes trained models into a transparent format and fully reconstructs them without structural changes, preserving their original functionality and behavior. 
This process does not affect inference time or performance and imports models on any target device without additional dependencies, enabling seamless execution in inference mode. 
PyMilo benefits a wide range of stakeholders, including machine learning engineers, data scientists, and AI practitioners, by facilitating the development of more transparent and accountable AI systems. Furthermore, researchers working on transparent AI [@rauker2023toward], user privacy in ML [@bodimani2024assessing], and safe AI [@macrae2019governing] can use PyMilo as a framework that provides transparency and safety in the machine learning environment.

# References