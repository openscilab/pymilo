# -*- coding: utf-8 -*-
"""Parameters and constants."""
import numpy as np
import sklearn.linear_model as linear_model
import sklearn.neural_network as neural_network
import sklearn.tree as tree
import sklearn.cluster as cluster
import sklearn.mixture as mixture
import sklearn.naive_bayes as naive_bayes
import sklearn.svm as svm
import sklearn.neighbors as neighbors
import sklearn.dummy as dummy
import sklearn.ensemble as ensemble
import sklearn.pipeline as pipeline
import sklearn.preprocessing as preprocessing

quantile_regressor_support = False
try:
    from sklearn.linear_model import QuantileRegressor
    quantile_regressor_support = True
except BaseException:
    pass

glm_support = {
    'GammaRegressor': False,
    'PoissonRegressor': False,
    'TweedieRegressor': False
}
try:
    from sklearn.linear_model import TweedieRegressor
    glm_support['TweedieRegressor'] = True
    from sklearn.linear_model import PoissonRegressor
    glm_support['PoissonRegressor'] = True
    from sklearn.linear_model import GammaRegressor
    glm_support['GammaRegressor'] = True
except BaseException:
    pass

sgd_one_class_svm_support = False
try:
    from sklearn.linear_model import SGDOneClassSVM
    sgd_one_class_svm_support = True
except BaseException:
    pass


bisecting_kmeans_support = False
try:
    from sklearn.cluster import BisectingKMeans
    bisecting_kmeans_support = True
except BaseException:
    pass

hdbscan_support = False
try:
    from sklearn.cluster import HDBSCAN
    hdbscan_support = True
except BaseException:
    pass


hist_gradient_boosting_support = False
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier
    hist_gradient_boosting_support = True
except BaseException:
    pass


PYMILO_VERSION = "0.8"
NOT_SUPPORTED = "NOT_SUPPORTED"
PYMILO_VERSION_DOES_NOT_EXIST = "Corrupted JSON file, `pymilo_version` doesn't exist in this file."
UNEQUAL_PYMILO_VERSIONS = "warning: Installed PyMilo version differes from the PyMilo version used to create the JSON file."
UNEQUAL_SKLEARN_VERSIONS = "warning: Installed Scikit version differes from the Scikit version used to create the JSON file and it may prevent PyMilo from transporting seamlessly."

SKLEARN_LINEAR_MODEL_TABLE = {
    "DummyRegressor": dummy.DummyRegressor,
    "DummyClassifier": dummy.DummyClassifier,
    "LinearRegression": linear_model.LinearRegression,
    "Ridge": linear_model.Ridge,
    "RidgeCV": linear_model.RidgeCV,
    "RidgeClassifier": linear_model.RidgeClassifier,
    "RidgeClassifierCV": linear_model.RidgeClassifierCV,
    "Lasso": linear_model.Lasso,
    "LassoCV": linear_model.LassoCV,
    "LassoLars": linear_model.LassoLars,
    "LassoLarsCV": linear_model.LassoLarsCV,
    "LassoLarsIC": linear_model.LassoLarsIC,
    "MultiTaskLasso": linear_model.MultiTaskLasso,
    "MultiTaskLassoCV": linear_model.MultiTaskLassoCV,
    "ElasticNet": linear_model.ElasticNet,
    "ElasticNetCV": linear_model.ElasticNetCV,
    "MultiTaskElasticNet": linear_model.MultiTaskElasticNet,
    "MultiTaskElasticNetCV": linear_model.MultiTaskElasticNetCV,
    "OrthogonalMatchingPursuit": linear_model.OrthogonalMatchingPursuit,
    "OrthogonalMatchingPursuitCV": linear_model.OrthogonalMatchingPursuitCV,
    "BayesianRidge": linear_model.BayesianRidge,
    "ARDRegression": linear_model.ARDRegression,
    "LogisticRegression": linear_model.LogisticRegression,
    "LogisticRegressionCV": linear_model.LogisticRegressionCV,
    "TweedieRegressor": TweedieRegressor if glm_support['TweedieRegressor'] else NOT_SUPPORTED,
    "PoissonRegressor": PoissonRegressor if glm_support['PoissonRegressor'] else NOT_SUPPORTED,
    "GammaRegressor": GammaRegressor if glm_support['GammaRegressor'] else NOT_SUPPORTED,
    "SGDRegressor": linear_model.SGDRegressor,
    "SGDClassifier": linear_model.SGDClassifier,
    "SGDOneClassSVM": SGDOneClassSVM if sgd_one_class_svm_support else NOT_SUPPORTED,
    "Perceptron": linear_model.Perceptron,
    "PassiveAggressiveRegressor": linear_model.PassiveAggressiveRegressor,
    "PassiveAggressiveClassifier": linear_model.PassiveAggressiveClassifier,
    "RANSACRegressor": linear_model.RANSACRegressor,
    "TheilSenRegressor": linear_model.TheilSenRegressor,
    "HuberRegressor": linear_model.HuberRegressor,
    "QuantileRegressor": QuantileRegressor if quantile_regressor_support else NOT_SUPPORTED,
}

SKLEARN_NEURAL_NETWORK_TABLE = {
    "MLPRegressor": neural_network.MLPRegressor,
    "MLPClassifier": neural_network.MLPClassifier,
    "BernoulliRBM": neural_network.BernoulliRBM,
}

SKLEARN_DECISION_TREE_TABLE = {
    "DecisionTreeRegressor": tree.DecisionTreeRegressor,
    "DecisionTreeClassifier": tree.DecisionTreeClassifier,
    "ExtraTreeRegressor": tree.ExtraTreeRegressor,
    "ExtraTreeClassifier": tree.ExtraTreeClassifier
}

SKLEARN_CLUSTERING_TABLE = {
    "KMeans": cluster.KMeans,
    "MiniBatchKMeans": cluster.MiniBatchKMeans,
    "BisectingKMeans": BisectingKMeans if bisecting_kmeans_support else NOT_SUPPORTED,
    "AffinityPropagation": cluster.AffinityPropagation,
    "MeanShift": cluster.MeanShift,
    "SpectralClustering": cluster.SpectralClustering,
    "SpectralBiclustering": cluster.SpectralBiclustering,
    "SpectralCoclustering": cluster.SpectralCoclustering,
    "AgglomerativeClustering": cluster.AgglomerativeClustering,
    "FeatureAgglomeration": cluster.FeatureAgglomeration,
    "DBSCAN": cluster.DBSCAN,
    "HDBSCAN": HDBSCAN if hdbscan_support else NOT_SUPPORTED,
    "OPTICS": cluster.OPTICS,
    "Birch": cluster.Birch,
    "GaussianMixture": mixture.GaussianMixture,
    "BayesianGaussianMixture": mixture.BayesianGaussianMixture,
}

SKLEARN_NAIVE_BAYES_TABLE = {
    "GaussianNB": naive_bayes.GaussianNB,
    "MultinomialNB": naive_bayes.MultinomialNB,
    "ComplementNB": naive_bayes.ComplementNB,
    "BernoulliNB": naive_bayes.BernoulliNB,
    "CategoricalNB": naive_bayes.CategoricalNB,
}

SKLEARN_SVM_TABLE = {
    "LinearSVC": svm.LinearSVC,
    "LinearSVR": svm.LinearSVR,
    "NuSVC": svm.NuSVC,
    "NuSVR": svm.NuSVR,
    "OneClassSVM": svm.OneClassSVM,
    "SVC": svm.SVC,
    "SVR": svm.SVR,
}

SKLEARN_NEIGHBORS_TABLE = {
    "KNeighborsRegressor": neighbors.KNeighborsRegressor,
    "KNeighborsClassifier": neighbors.KNeighborsClassifier,
    "RadiusNeighborsRegressor": neighbors.RadiusNeighborsRegressor,
    "RadiusNeighborsClassifier": neighbors.RadiusNeighborsClassifier,
    "NearestNeighbors": neighbors.NearestNeighbors,
    "NearestCentroid": neighbors.NearestCentroid,
    "LocalOutlierFactor": neighbors.LocalOutlierFactor,
}

SKLEARN_ENSEMBLE_TABLE = {
    "AdaBoostRegressor": ensemble.AdaBoostRegressor,
    "AdaBoostClassifier": ensemble.AdaBoostClassifier,
    "BaggingRegressor": ensemble.BaggingRegressor,
    "BaggingClassifier": ensemble.BaggingClassifier,
    "ExtraTreesClassifier": ensemble.ExtraTreesClassifier,
    "ExtraTreesRegressor": ensemble.ExtraTreesRegressor,
    "GradientBoostingRegressor": ensemble.GradientBoostingRegressor,
    "GradientBoostingClassifier": ensemble.GradientBoostingClassifier,
    "IsolationForest": ensemble.IsolationForest,
    "RandomForestClassifier": ensemble.RandomForestClassifier,
    "RandomForestRegressor": ensemble.RandomForestRegressor,
    "RandomTreesEmbedding": ensemble.RandomTreesEmbedding,
    "StackingRegressor": ensemble.StackingRegressor,
    "StackingClassifier": ensemble.StackingClassifier,
    "VotingRegressor": ensemble.VotingRegressor,
    "VotingClassifier": ensemble.VotingClassifier,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor if hist_gradient_boosting_support else NOT_SUPPORTED,
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier if hist_gradient_boosting_support else NOT_SUPPORTED,
    ####
    "Pipeline": pipeline.Pipeline,
}

SKLEARN_PREPROCESSING_TABLE = {
    "StandardScaler": preprocessing.StandardScaler,
    "OneHotEncoder": preprocessing.OneHotEncoder,
    "LabelBinarizer": preprocessing.LabelBinarizer,
    "LabelEncoder": preprocessing.LabelEncoder,
}

KEYS_NEED_PREPROCESSING_BEFORE_DESERIALIZATION = {
    "_label_binarizer": preprocessing.LabelBinarizer,  # in Ridge Classifier
    "active_": np.int32,  # in Lasso Lars
    "n_nonzero_coefs_": np.int64,  # in OMP-CV
    "scores_": dict,  # in Logistic Regression CV,
    "_base_loss": {},  # BaseLoss in Logistic Regression,
    "loss_function_": {},  # LossFunction in SGD Classifier,
    "estimator_": {},  # LinearRegression model inside RANSAC
}

NUMPY_TYPE_DICT = {
    "numpy.intc": np.intc,
    "numpy.int32": np.int32,
    "numpy.int64": np.int64,
    "numpy.float64": np.float64,
    "numpy.infinity": lambda _: np.inf,
    "numpy.uint8": np.uint8,
    "numpy.uint64": np.uint64,
    "numpy.dtype": np.dtype,
}

EXPORTED_MODELS_PATH = {
    "LINEAR_MODEL": "exported_linear_models",
    "NEURAL_NETWORK": "exported_neural_networks",
    "DECISION_TREE": "exported_decision_trees",
    "CLUSTERING": "exported_clusterings",
    "NAIVE_BAYES": "exported_naive_bayes",
    "SVM": "exported_svms",
    "NEIGHBORS": "exported_neighbors",
    "ENSEMBLE": "exported_ensembles",
}
