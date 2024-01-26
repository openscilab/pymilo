# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
### Changed
## [0.5] - 2024-01-31
### Added
- `reset` function in the `Transport` interface
- `reset` function implementation in `AbstractTransporter`
- `Gaussian Naive Bayes` declared as `GaussianNB` model 
- `Multinomial Naive Bayes` model declared as `MultinomialNB` model
- `Complement Naive Bayes` model declared as `ComplementNB` model
- `Bernoulli Naive Bayes` model declared as `BernoulliNB` model
- `Categorical Naive Bayes` model declared as `CategoricalNB` model
- Naive Bayes models test runner
- Naive Bayes chain 
### Changed
- `Transport` function of `AbstractTransporter` updated
- fix the order of `CFNode` fields serialization in `CFNodeTransporter`
- `GeneralDataStructureTransporter` support list of ndarray with different shapes
- Tests config modified
- Naive Bayes params initialized in `pymilo_param`
- Naive Bayes support added to `pymilo_func.py`
- `SUPPORTED_MODELS.md` updated
- `README.md` updated
## [0.4] - 2024-01-22
### Added
- `has_named_parameter` method in `util.py`
- `CFSubcluster` Transporter(inside `CFNode` Transporter)
- `CFNode` Transporter
- `Birch` model
- `SpectralBiclustering` model
- `SpectralCoclustering` model
- `MiniBatchKMeans` model
- `feature_request.yml` template
- `config.yml` for issue template
- `BayesianGaussianMixture` model
- `serialize_tuple` method in `GeneralDataStructureTransporter`
- `import_function` method in `util.py`
- `Function` Transporter
- `FeatureAgglomeration` model
- `HDBSCAN` model
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
- `LossFunctionTransporter` enhanced to handle scikit 1.4.0 `_loss_function_` field
- Codacy Static Code Analyzer's suggestions applied
- Spectral Clustering test folder refactored
- Bug report template modified
- `GeneralDataStructureTransporter` updated
- Tests config modified
- Clustering data set preparation added to `data_exporter.py`
- Clustering params initialized in `pymilo_param`
- Clustering support added to `pymilo_func.py`
- `Python 3.12` added to `test.yml`
- `dev-requirements.txt` updated
- Code quality badges added to `README.md`
- `SUPPORTED_MODELS.md` updated
- `README.md` updated
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

[Unreleased]: https://github.com/openscilab/pymilo/compare/v0.5...dev
[0.5]: https://github.com/openscilab/pymilo/compare/v0.4...v0.5
[0.4]: https://github.com/openscilab/pymilo/compare/v0.3...v0.4
[0.3]: https://github.com/openscilab/pymilo/compare/v0.2...v0.3
[0.2]: https://github.com/openscilab/pymilo/compare/v0.1...v0.2
[0.1]: https://github.com/openscilab/pymilo/compare/e887108...v0.1