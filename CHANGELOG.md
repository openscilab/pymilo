# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
### Changed

## [0.1] - 2023-06-29
### Added
- scikit-learn linear models support
- `Export` class
- `Import` class

## [0.2] - 2023-08-2
### Added
- scikit-learn neural network models 
- `MLP Regressor` model 
- `MLP Classifier` model
- `BernoulliRBN` model
- `SGDOptimizer` transporter
- `RandomState(MT19937)` transporter
- `Adamoptimizer` transporter
- Neural Network chain
- Neural Network exceptions 
- `ndarray_to_list` method in `GeneralDataStructureTransporter`
- `list_to_ndarray` method in `GeneralDataStructureTransporter` 
- `neural_network_chain.py` chain
### Changed
- `GeneralDataStructure` Transporter updated
- `LabelBinerizer` Transporter updated
- `linear model` chain updated
- GeneralDataStructure transporter enhanced
- LabelBinerizer transporter updated
- transporters' chain router added to `pymilo func`
- NeuralNetwork params initialized in `pymilo_param`
- `pymilo_test` updated to support multiple models
- `linear_model_chain` refactored


[Unreleased]: https://github.com/openscilab/pymilo/compare/v0.2...dev
[0.1]: https://github.com/openscilab/pymilo/compare/e887108...v0.1
[0.2]: https://github.com/openscilab/pymilo/compare/v0.1...v0.2


