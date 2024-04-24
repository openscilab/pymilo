# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `util.py` in chains
- `BinMapperTransporter` Transporter
- `BunchTransporter` Transporter
- `GeneratorTransporter` Transporter
- `LabelEncoderTransporter` Transporter
- `OneHotEncoderTransporter` Transporter
- `TreePredictorTransporter` Transporter
- `AdaboostClassifier` model
- `AdaboostRegressor` model
- `BaggingClassifier` model
- `BaggingRegressor` model
- `ExtraTreesClassifier` model
- `ExtraTreesRegressor` model
- `GradientBoosterClassifier` model
- `GradientBoosterRegressor` model
- `HistGradientBoosterClassifier` model
- `HistGradientBoosterRegressor` model
- `RandomForestClassifier` model
- `RandomForestRegressor` model
- `IsolationForest` model
- `RandomTreesEmbedding` model
- `StackingClassifier` model
- `StackingRegressor` model
- `VotingClassifier` model
- `VotingRegressor` model
- `Pipeline` model
- Ensemble models test runner
- Ensemble chain
### Changed
- `util.py` in utils updated
- `test_pymilo.py` updated
- `pymilo_func.py` updated
- `linear_model_chain.py` updated
- `neural_network_chain.py` updated
- `decision_tree_chain.py` updated
- `clustering_chain.py` updated
- `naive_bayes_chain.py` updated
- `neighbours_chain.py` updated
- `svm_chain.py` updated
- `GeneralDataStructure` Transporter updated
- `LossFunction` Transporter updated
- `AbstractTransporter` updated
- Tests config modified
- Unequal sklearn version error added in `pymilo_param.py`
- Ensemble params initialized in `pymilo_param`
- Ensemble support added to `pymilo_func.py`
- `SUPPORTED_MODELS.md` updated
- `README.md` updated
## [0.7] - 2024-04-03
### Added
- `pymilo_nearest_neighbor_test` function added to `test_pymilo.py`
- `NeighborsTreeTransporter` Transporter
- `LocalOutlierFactor` model
- `RadiusNeighborsClassifier` model
- `RadiusNeighborsRegressor` model
- `NearestCentroid` model
- `NearestNeighbors` model
- `KNeighborsClassifier` model
- `KNeighborsRegressor` model
- Neighbors models test runner
- Neighbors chain
### Changed
- Tests config modified
- Neighbors params initialized in `pymilo_param`
- Neighbors support added to `pymilo_func.py`
- `SUPPORTED_MODELS.md` updated
- `README.md` updated
## [0.6] - 2024-03-27
### Added
- `deserialize_primitive_type` function in `GeneralDataStructureTransporter`
- `is_deserialized_ndarray` function in `GeneralDataStructureTransporter`
- `deep_deserialize_ndarray` function in `GeneralDataStructureTransporter`
- `deep_serialize_ndarray`  function in `GeneralDataStructureTransporter`
- `SVR` model
- `SVC` model
- `One Class SVM` model
- `NuSVR` model
- `NuSVC` model
- `Linear SVR` model
- `Linear SVC` model
- SVM models test runner
- SVM chain
### Changed
- `pymilo_param.py` updated
- `pymilo_obj.py` updated to use predefined strings
- `TreeTransporter` updated
- `get_homogeneous_type` function in `util.py` updated
- `GeneralDataStructureTransporter` updated to use deep ndarray serializer & deserializer
- `check_str_in_iterable` updated
- `Label Binarizer` Transporter updated
- `Function` Transporter updated
- `CFNode` Transporter updated
- `Bisecting Tree` Transporter updated
- Tests config modified
- SVM params initialized in `pymilo_param`
- SVM support added to `pymilo_func.py`
- `SUPPORTED_MODELS.md` updated
- `README.md` updated
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

[Unreleased]: https://github.com/openscilab/pymilo/compare/v0.7...dev
[0.7]: https://github.com/openscilab/pymilo/compare/v0.6...v0.7
[0.6]: https://github.com/openscilab/pymilo/compare/v0.5...v0.6
[0.5]: https://github.com/openscilab/pymilo/compare/v0.4...v0.5
[0.4]: https://github.com/openscilab/pymilo/compare/v0.3...v0.4
[0.3]: https://github.com/openscilab/pymilo/compare/v0.2...v0.3
[0.2]: https://github.com/openscilab/pymilo/compare/v0.1...v0.2
[0.1]: https://github.com/openscilab/pymilo/compare/e887108...v0.1