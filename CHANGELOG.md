# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `GaussianMixture` model
- `OPTICS` model
- `DBSCAN` model
- `AgglomerativeClustering` model
- `SpectralClustering` model
- `MeanShift` model 
- `AffinityPropagation` model
- `Kmeans` model
- Clustering models test runner
- Clustering chain 
### Changed
- Tests config modified
- Clustering data set preparation added to `data_exporter.py`
- Clustering params initialized in `pymilo_param`
- Clustering support added to `pymilo_func.py`
## [0.3] - 2023-09-27
### Added
- scikit-learn decision tree models
- `ExtraTreeClassifier` model
- `ExtraTreeRegressor` model
- `DecisionTreeClassifier` model
- `DecisionTreeRegressor` model
- `Tree` Transporter
- Decision Tree chain
### Changed
- Tests config modified
- DecisionTree params initialized in `pymilo_param`
- Decision Tree support added to `pymilo_func.py`
## [0.2] - 2023-08-02
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
## [0.1] - 2023-06-29
### Added
- scikit-learn linear models support
- `Export` class
- `Import` class

[Unreleased]: https://github.com/openscilab/pymilo/compare/v0.3...dev
[0.3]: https://github.com/openscilab/pymilo/compare/v0.2...v0.3
[0.2]: https://github.com/openscilab/pymilo/compare/v0.1...v0.2
[0.1]: https://github.com/openscilab/pymilo/compare/e887108...v0.1