# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `validate_http_url` function in `streaming.util.py`
- `validate_websocket_url` function in `streaming.util.py`
- `ML Streaming` WebSocket testcases
- `CommunicationProtocol` Enum in `streaming.communicator.py`
- `WebSocketClientCommunicator` class in `streaming.communicator.py`
- `WebSocketServerCommunicator` class in `streaming.communicator.py`
- PyMilo exception types added in `pymilo/exceptions/__init__.py`
- PyMilo exception types added in `pymilo/__init__.py`
### Changed
- `client_communicator` parameter added to `PyMiloClient` class
- `server_communicator` parameter added to `PyMiloServer` class
- ML Streaming testcases updated to support protocol selection
- `Python 3.13` added to `test.yml`
## [1.0] - 2024-09-16
### Added
- Compression method test in `ML Streaming` RESTful testcases
- `CLI` handler in `tests/test_ml_streaming/run_server.py`
- `Compression` Enum in `streaming.compressor.py`
- `GZIPCompressor` class in `streaming.compressor.py`
- `ZLIBCompressor` class in `streaming.compressor.py`
- `LZMACompressor` class in `streaming.compressor.py`
- `BZ2Compressor` class in `streaming.compressor.py`
- `encrypt_compress` function in `PymiloClient`
- `parse` function in `RESTServerCommunicator`
- `is_callable_attribute` function in `PymiloServer`
- `streaming.param.py`
- `attribute_type` function in `RESTServerCommunicator`
- `AttributeTypePayload` class in `RESTServerCommunicator`
- `attribute_type` function in `RESTClientCommunicator`
- `Mode` Enum in `PymiloClient`
- Import from url testcases
- `download_model` function in `utils.util.py`
- `PymiloServer` class in `streaming.pymilo_server.py`
- `PymiloClient` class in `PymiloClient`
- `Communicator` interface in `streaming.interfaces.py`
- `RESTClientCommunicator` class in `streaming.communicator.py`
- `RESTServerCommunicator` class in `streaming.communicator.py`
- `Compressor` interface in `streaming.interfaces.py`
- `DummyCompressor` class in `streaming.compressor.py`
- `Encryptor` interface in `streaming.interfaces.py`
- `DummyEncryptor` class in `streaming.encryptor.py`
- `ML Streaming` RESTful testcases
- `streaming-requirements.txt`
### Changed
- `README.md` updated
- `ML Streaming` RESTful testcases
- `attribute_call` function in `RESTServerCommunicator`
- `AttributeCallPayload` class in `RESTServerCommunicator`
- upload function in `RESTClientCommunicator`
- download function in `RESTClientCommunicator`
- `__init__` function in `RESTClientCommunicator`
- `attribute_calls` function in `RESTClientCommunicator`
- `requests` added to `requirements.txt`
- `uvicorn`, `fastapi`, `requests` and `pydantic` added to `dev-requirements.txt`
- `ML Streaming` RESTful testcases
- `__init__` function in `PymiloServer`
- `__getattr__` function in `PymiloClient`
- `__init__` function in `PymiloClient`
- `toggle_mode` function in `PymiloClient`
- `upload` function in `PymiloClient`
- `download` function in `PymiloClient`
- `__init__` function in `PymiloServer`
- `serialize_cfnode` function in `transporters.cfnode_transporter.py`
- `__init__` function in `Import` class
- `serialize` function in `transporters.tree_transporter.py`
- `deserialize` function in `transporters.tree_transporter.py`
- `serialize` function in `transporters.sgdoptimizer_transporter.py`
- `deserialize` function in `transporters.sgdoptimizer_transporter.py`
- `serialize` function in `transporters.randomstate_transporter.py`
- `deserialize` function in `transporters.randomstate_transporter.py`
- `serialize` function in `transporters.bunch_transporter.py`
- `deserialize` function in `transporters.bunch_transporter.py`
- `serialize` function in `transporters.adamoptimizer_transporter.py`
- `deserialize` function in `transporters.adamoptimizer_transporter.py`
- `serialize_linear_model` function in `chains.linear_model_chain.py`
- `serialize_ensemble` function in `chains.ensemble_chain.py`
- `serialize` function in `GeneralDataStructureTransporter` Transporter refactored
- `get_deserialized_list` function in `GeneralDataStructureTransporter` Transporter refactored
- `Export` class call by reference bug fixed
## [0.9] - 2024-07-01
### Added
- Anaconda workflow
- `prefix_list` function in `utils.util.py`
- `KBinsDiscretizer` preprocessing model
- `PowerTransformer` preprocessing model
- `SplineTransformer` preprocessing model
- `TargetEncoder` preprocessing model
- `QuantileTransformer` preprocessing model
- `RobustScaler` preprocessing model
- `PolynomialFeatures` preprocessing model
- `OrdinalEncoder` preprocessing model
- `Normalizer` preprocessing model
- `MaxAbsScaler` preprocessing model
- `MultiLabelBinarizer` preprocessing model
- `KernelCenterer` preprocessing model
- `FunctionTransformer` preprocessing model
- `Binarizer` preprocessing model
- Preprocessing models test runner
### Changed
- `Command` enum class in `transporter.py`
- `SerializationErrorTypes` enum class in `serialize_exception.py`
- `DeserializationErrorTypes` enum class in `deserialize_exception.py`
- `meta.yaml` modified
- `NaN` type in `pymilo_param`
- `NaN` type transportation in `GeneralDataStructureTransporter` Transporter
- `BSpline` Transportation in `PreprocessingTransporter` Transporter
- one layer deeper transportation in `PreprocessingTransporter` Transporter
- dictating outer ndarray dtype in `GeneralDataStructureTransporter` Transporter 
- preprocessing params fulfilled in `pymilo_param`
- `SUPPORTED_MODELS.md` updated
- `README.md` updated
- `serialize_possible_ml_model` in the Ensemble chain
## [0.8] - 2024-05-06
### Added
- `StandardScaler` Transformer in `pymilo_param.py`
- `PreprocessingTransporter` Transporter
- ndarray shape config in `GeneralDataStructure` Transporter
- `util.py` in chains
- `BinMapperTransporter` Transporter
- `BunchTransporter` Transporter
- `GeneratorTransporter` Transporter
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
- `SECURITY.md`
### Changed
- `Pipeline` test updated
- `LabelBinarizer`,`LabelEncoder` and `OneHotEncoder` got embedded in `PreprocessingTransporter`
- Preprocessing support added to Ensemble chain
- Preprocessing params initialized in `pymilo_param`
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

[Unreleased]: https://github.com/openscilab/pymilo/compare/v1.0...dev
[1.0]: https://github.com/openscilab/pymilo/compare/v0.9...v1.0
[0.9]: https://github.com/openscilab/pymilo/compare/v0.8...v0.9
[0.8]: https://github.com/openscilab/pymilo/compare/v0.7...v0.8
[0.7]: https://github.com/openscilab/pymilo/compare/v0.6...v0.7
[0.6]: https://github.com/openscilab/pymilo/compare/v0.5...v0.6
[0.5]: https://github.com/openscilab/pymilo/compare/v0.4...v0.5
[0.4]: https://github.com/openscilab/pymilo/compare/v0.3...v0.4
[0.3]: https://github.com/openscilab/pymilo/compare/v0.2...v0.3
[0.2]: https://github.com/openscilab/pymilo/compare/v0.1...v0.2
[0.1]: https://github.com/openscilab/pymilo/compare/e887108...v0.1
