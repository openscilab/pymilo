# -*- coding: utf-8 -*-
"""Parameters and constants."""
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralCoclustering
from sklearn.cluster import SpectralBiclustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import ExtraTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import VotingClassifier

hist_gradient_boosting_support = False
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier
    hist_gradient_boosting_support = True
except BaseException:
    pass

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import LocalOutlierFactor

from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM
from sklearn.svm import NuSVR
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVR
from sklearn.svm import LinearSVC

from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import LabelBinarizer
from numpy import uint8
from numpy import intc
from numpy import inf
from numpy import float64
from numpy import int32
from numpy import int64
from numpy import uint64

from sklearn.dummy import DummyRegressor
from sklearn.dummy import DummyClassifier

from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

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

PYMILO_VERSION = "0.7"
NOT_SUPPORTED = "NOT_SUPPORTED"
PYMILO_VERSION_DOES_NOT_EXIST = "Corrupted JSON file, `pymilo_version` doesn't exist in this file."
UNEQUAL_PYMILO_VERSIONS = "warning: Installed Pymilo version differes from pymilo version used to create the JSON file."

SKLEARN_LINEAR_MODEL_TABLE = {
    "DummyRegressor": DummyRegressor,
    "DummyClassifier": DummyClassifier,
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "RidgeCV": RidgeCV,
    "RidgeClassifier": RidgeClassifier,
    "RidgeClassifierCV": RidgeClassifierCV,
    "Lasso": Lasso,
    "LassoCV": LassoCV,
    "LassoLars": LassoLars,
    "LassoLarsCV": LassoLarsCV,
    "LassoLarsIC": LassoLarsIC,
    "MultiTaskLasso": MultiTaskLasso,
    "MultiTaskLassoCV": MultiTaskLassoCV,
    "ElasticNet": ElasticNet,
    "ElasticNetCV": ElasticNetCV,
    "MultiTaskElasticNet": MultiTaskElasticNet,
    "MultiTaskElasticNetCV": MultiTaskElasticNetCV,
    "OrthogonalMatchingPursuit": OrthogonalMatchingPursuit,
    "OrthogonalMatchingPursuitCV": OrthogonalMatchingPursuitCV,
    "BayesianRidge": BayesianRidge,
    "ARDRegression": ARDRegression,
    "LogisticRegression": LogisticRegression,
    "LogisticRegressionCV": LogisticRegressionCV,
    "TweedieRegressor": TweedieRegressor if glm_support['TweedieRegressor'] else NOT_SUPPORTED,
    "PoissonRegressor": PoissonRegressor if glm_support['PoissonRegressor'] else NOT_SUPPORTED,
    "GammaRegressor": GammaRegressor if glm_support['GammaRegressor'] else NOT_SUPPORTED,
    "SGDRegressor": SGDRegressor,
    "SGDClassifier": SGDClassifier,
    "SGDOneClassSVM": SGDOneClassSVM if sgd_one_class_svm_support else NOT_SUPPORTED,
    "Perceptron": Perceptron,
    "PassiveAggressiveRegressor": PassiveAggressiveRegressor,
    "PassiveAggressiveClassifier": PassiveAggressiveClassifier,
    "RANSACRegressor": RANSACRegressor,
    "TheilSenRegressor": TheilSenRegressor,
    "HuberRegressor": HuberRegressor,
    "QuantileRegressor": QuantileRegressor if quantile_regressor_support else NOT_SUPPORTED,
}

SKLEARN_NEURAL_NETWORK_TABLE = {
    "MLPRegressor": MLPRegressor,
    "MLPClassifier": MLPClassifier,
    "BernoulliRBM": BernoulliRBM,
}

SKLEARN_DECISION_TREE_TABLE = {
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "ExtraTreeRegressor": ExtraTreeRegressor,
    "ExtraTreeClassifier": ExtraTreeClassifier
}

SKLEARN_CLUSTERING_TABLE = {
    "KMeans": KMeans,
    "MiniBatchKMeans": MiniBatchKMeans,
    "BisectingKMeans": BisectingKMeans if bisecting_kmeans_support else NOT_SUPPORTED,
    "AffinityPropagation": AffinityPropagation,
    "MeanShift": MeanShift,
    "SpectralClustering": SpectralClustering,
    "SpectralBiclustering": SpectralBiclustering,
    "SpectralCoclustering": SpectralCoclustering,
    "AgglomerativeClustering": AgglomerativeClustering,
    "FeatureAgglomeration": FeatureAgglomeration,
    "DBSCAN": DBSCAN,
    "HDBSCAN": HDBSCAN if hdbscan_support else NOT_SUPPORTED,
    "OPTICS": OPTICS,
    "Birch": Birch,
    "GaussianMixture": GaussianMixture,
    "BayesianGaussianMixture": BayesianGaussianMixture,
}

SKLEARN_NAIVE_BAYES_TABLE = {
    "GaussianNB": GaussianNB,
    "MultinomialNB": MultinomialNB,
    "ComplementNB": ComplementNB,
    "BernoulliNB": BernoulliNB,
    "CategoricalNB": CategoricalNB,
}

SKLEARN_SVM_TABLE = {
    "LinearSVC": LinearSVC,
    "LinearSVR": LinearSVR,
    "NuSVC": NuSVC,
    "NuSVR": NuSVR,
    "OneClassSVM": OneClassSVM,
    "SVC": SVC,
    "SVR": SVR,
}

SKLEARN_NEIGHBORS_TABLE = {
    "KNeighborsRegressor": KNeighborsRegressor,
    "KNeighborsClassifier": KNeighborsClassifier,
    "RadiusNeighborsRegressor": RadiusNeighborsRegressor,
    "RadiusNeighborsClassifier": RadiusNeighborsClassifier,
    "NearestNeighbors": NearestNeighbors,
    "NearestCentroid": NearestCentroid,
    "LocalOutlierFactor": LocalOutlierFactor,
}

SKLEARN_ENSEMBLE_TABLE = {
    "AdaBoostRegressor": AdaBoostRegressor,
    "AdaBoostClassifier": AdaBoostClassifier,
    "BaggingRegressor": BaggingRegressor,
    "BaggingClassifier": BaggingClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "IsolationForest": IsolationForest,
    "RandomForestClassifier": RandomForestClassifier,
    "RandomForestRegressor": RandomForestRegressor,
    "RandomTreesEmbedding": RandomTreesEmbedding,
    "StackingRegressor": StackingRegressor,
    "StackingClassifier": StackingClassifier,
    "VotingRegressor": VotingRegressor,
    "VotingClassifier": VotingClassifier,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor if hist_gradient_boosting_support else NOT_SUPPORTED,
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier if hist_gradient_boosting_support else NOT_SUPPORTED,
}

KEYS_NEED_PREPROCESSING_BEFORE_DESERIALIZATION = {
    "_label_binarizer": LabelBinarizer,  # in Ridge Classifier
    "active_": int32,  # in Lasso Lars
    "n_nonzero_coefs_": int64,  # in OMP-CV
    "scores_": dict,  # in Logistic Regression CV,
    "_base_loss": {},  # BaseLoss in Logistic Regression,
    "loss_function_": {},  # LossFunction in SGD Classifier,
    "estimator_": {},  # LinearRegression model inside RANSAC
}

NUMPY_TYPE_DICT = {
    "numpy.intc": intc,
    "numpy.int32": int32,
    "numpy.int64": int64,
    "numpy.float64": float64,
    "numpy.infinity": lambda _: inf,
    "numpy.uint8": uint8,
    "numpy.uint64": uint64,
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
