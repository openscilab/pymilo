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
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: 1 
  - name: Sepand Haghighi
    orcid: 0000-0000-0000-0000
    corresponding: false
    affiliation: 1
  - name: Sadra Sabouri
    orcid: 0000-0003-1047-2346
    corresponding: false
    affiliation: 1
  - name: Alireza Zolanvari
    orcid: 0000-0000-0000-0000
    corresponding: false
    affiliation: 1
affiliations:
 - name: Open Science Lab
   index: 1

date: 10 June 2025
bibliography: paper.bib
---

# Summary
PyMilo is an open-source Python package that addresses the limitations of existing machine learning (ML) model storage formats by providing a transparent, reliable, end-to-end, and safe method for exporting and deploying trained models. Current tools rely on opaque or executable formats that obscure internal model structures, making them difficult to audit, verify, or safely share. Others apply structural transformations during export that may degrade predictive performance and reduce the model to a limited inference-only interface. In contrast, PyMilo serializes models in a human-readable, non-executable format that preserves end-to-end model fidelity and enables reliable, safe, and interpretable exchange. This package is designed to make the preservation and reuse of trained ML models safer, more interpretable, and easier to manage across different stages of the workflow.

# Statement of Need
Modern machine learning development is largely centered around the Python ecosystem, which has become a dominant platform for building and training models due to its rich libraries and community support [@Raschka2020]. However, once a model is trained, sharing or deploying it securely and transparently remains a significant challenge. This issue is especially important in high-stakes domains such as healthcare, where ensuring model accountability and integrity is critical [@Garbin2022].
In such settings, any lack of clarity about a model’s internal logic or origin can reduce trust in its predictions. Researchers have increasingly emphasized that greater transparency in AI systems is critical for maintaining user trust and protecting privacy in machine learning applications [@Bodimani2024].

Despite ongoing concerns around transparency and safety, the dominant approach for exchanging Python-trained models remains ad hoc binary serialization, most commonly through Python’s `pickle` module or its variant `joblib`. These formats allow developers to store complex model objects with minimal effort, but they were never designed with security or human interpretability in mind. In fact, loading a pickle file will execute arbitrary code contained within it, a known vulnerability that can be exploited if the file is maliciously crafted [@Brownlee2018]. 

Alongside pickle, several standardized model interchange formats have been introduced to improve portability. ONNX (Open Neural Network Exchange) and PMML (Predictive Model Markup Language) convert trained models into framework-neutral representations [@Verma2023; @ONNX2017], enabling the use of the model in diverse environments.

However, beyond security and transparency issues, existing model export solutions face compatibility and fidelity challenges. Converting a complex model pipeline into ONNX or PMML may result in structural differences, approximations, or the loss of critical training details, as these formats often use alternative implementations of algorithms. Such discrepancies can degrade the model’s performance or accuracy [@Guazzelli2009; @Wang2020]. A recent study, for example, documented that exporting certain machine learning models to ONNX led to significant drops in accuracy, sometimes up to 10–15%, highlighting that the converted models did not fully preserve the original behavior [@Guazzelli2009; @Wang2020].

In summary, current solutions force practitioners into a trade-off between security, transparency, end-to-end fidelity, and performance preservation. Binary formats like Pickle offer convenience but pose serious safety and transparency risks. Meanwhile, interoperable formats such as ONNX and PMML are safer and more portable, but they fail to preserve full model behavior and predictive performance. In addition, interoperable formats like ONNX and PMML do not provide end-to-end preservation of models, as the re-imported versions differ in internal structure, functionality, or interface compared to the original. The machine learning community still lacks a truly end-to-end solution that allows models to be shared safely (with no risk of arbitrary code execution), inspected easily by humans, and faithfully reconstructed for seamless use across diverse environments.

Pickle/Joblib: The most common solution for saving Python machine learning models is to use the pickle module (often via joblib in scikit-learn) to serialize the model object to disk [@Brownlee2018]. This approach preserves all details of the model within Python and allows for easy restoration of the exact same object in a compatible environment. Unfortunately, pickle’s convenience comes with serious security drawbacks. Because unpickling will execute whatever bytecode is present in the file, a malformed or malicious pickle can carry out arbitrary operations on the host system [@Brownlee2018]. The official Python documentation explicitly warns that pickle is not secure against hostile data. Furthermore, pickle files are opaque binary blobs; there is no straightforward way to inspect their contents without loading them. Thus, while pickle provides an end-to-end model export/import capability within Python, it fails in terms of safety and transparency. The reliance on matching library versions is another subtle issue – a pickle generated in one version of a library may not load correctly in a future version, raising concerns about the longevity and reproducibility of models.

ONNX/PMML: To enable cross-platform model sharing, standardized formats like ONNX and PMML have been developed. ONNX provides a graph-based representation of machine learning models that many frameworks can export to or import from [@Verma2023]. It defines a set of primitive operators (linear transforms, activations, etc.) such that a model saved in ONNX can be run using any runtime that implements these operators. Similarly, PMML is an XML-based standard from the Data Mining Group that describes predictive models in a language-agnostic way (covering, for example, decision trees, regressions, and clustering models) [@ONNX2017]. Using these formats, one can take a model trained in Python and deploy it in a Java or C++ system without directly relying on the original training code. The trade-off, however, is that the model is no longer the same object but rather a translated version. Complex pipeline objects or custom model logic often cannot be expressed in ONNX/PMML and are lost or must be re-implemented. For example, ONNX has shown a significant performance degradation during model export, with up to 10-15% accuracy loss in certain scenarios [@Wang2020]. Additionally, ONNX and similar formats sacrifice readability – the ONNX file is a binary protocol buffer that cannot be understood without specialized tools, and while PMML is a human-readable XML format, but it tends to be verbose and limited to a restricted set of supported model classes. In summary, interchange formats improve portability at the expense of guaranteed end-to-end reproducibility, and the transformation process also affects inference time and model performance due to changes in the model structure.

SKOPS/TensorFlow.js: A few recent tools attempt to bridge the gaps for specific sub-communities. SKOPS is a library introduced to more securely persist scikit-learn models without using pickle. It serializes models into a custom format that avoids executing code on load and even allows some inspection of the file’s contents (for example, viewing model hyperparameters) [@Noyan2023]. By integrating with online model hubs, SKOPS has facilitated sharing scikit-learn pipelines on platforms like the Hugging Face Hub. However, SKOPS is inherently limited in scope: it only supports models built with scikit-learn and related Python libraries, and its output format remains a specialized (non-standard) schema for Python objects [@Noyan2023]. While it mitigates the direct code injection risk, it does not provide a truly human-readable representation (the serialized file is structured data that still needs SKOPS tooling to interpret) and cannot be used for models from other frameworks, such as deep learning libraries. Another tool with a different aim is TensorFlow.js, which enables deployment of TensorFlow models in JavaScript environments (browsers or Node.js) [@TFJS2018]. TensorFlow.js provides conversion utilities that take a trained TensorFlow (or Keras) model and produce a set of files (JSON for model architecture and binary weights) that can be loaded and executed in JavaScript [@TFJS2018]. This allows machine learning models to be run client-side, tapping into WebGL for acceleration. While TensorFlow.js exports are indeed in a non-executable, human-readable format (JSON), it is limited to TensorFlow models and can be inefficient when dealing with large models [@tensorflow2015_whitepaper]. Additionally, TensorFlow.js requires significant modifications to the original model architecture, which can lead to compatibility issues, performance degradation, and impact inference time. A scikit-learn or PyTorch model cannot be exported with TensorFlow.js without first re-implementing or retraining it in TensorFlow. Moreover, running complex models in a JavaScript runtime carries performance and memory penalties – large neural networks that run efficiently in Python/C++ may become prohibitively slow or even infeasible in the browser context [@NerdCorner2025].

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