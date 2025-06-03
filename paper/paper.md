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

