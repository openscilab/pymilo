# -*- coding: utf-8 -*-
"""Parameters and constants."""
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from numpy import int32
from numpy import int64

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV

from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import MultiTaskLassoCV

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import MultiTaskElasticNetCV

from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV

from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import GammaRegressor

from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDOneClassSVM

from sklearn.linear_model import Perceptron

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import PassiveAggressiveRegressor

from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import HuberRegressor

from sklearn.linear_model import QuantileRegressor


PYMILO_VERSION = "3.6"

SKLEARN_MODEL_TABLE = {
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

    "TweedieRegressor": TweedieRegressor,
    "PoissonRegressor": PoissonRegressor,
    "GammaRegressor": GammaRegressor,

    "SGDRegressor": SGDRegressor,
    "SGDClassifier": SGDClassifier,
    "SGDOneClassSVM": SGDOneClassSVM,

    "Perceptron": Perceptron,

    "PassiveAggressiveRegressor": PassiveAggressiveRegressor,
    "PassiveAggressiveClassifier": PassiveAggressiveClassifier,

    "RANSACRegressor": RANSACRegressor,
    "TheilSenRegressor": TheilSenRegressor,
    "HuberRegressor": HuberRegressor,

    "QuantileRegressor": QuantileRegressor

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
    "numpy.int32": int32,
    "numpy.int64": int64,
    "numpy.infinity": lambda x: np.inf
}
