import os
import pytest

from pymilo.pymilo_param import SKLEARN_LINEAR_MODEL_TABLE, NOT_SUPPORTED

from linear_regression.linear_regression import linear_regression

from ridge.ridge_regression import ridge_regression
from ridge.ridge_regression_cv import ridge_regression_cv
from ridge.ridge_classifier import ridge_classifier
from ridge.ridge_classifier_cv import ridge_classifier_cv

from lasso_lars.lasso import lasso
from lasso_lars.lasso_cv import lasso_cv
from lasso_lars.lasso_lars import lasso_lars
from lasso_lars.lasso_lars_cv import lasso_lars_cv
from lasso_lars.lasso_lars_ic import lasso_lars_ic
from lasso_lars.multi_task_lasso import multi_task_lasso
from lasso_lars.multi_task_lasso_cv import multi_task_lasso_cv

from elasticnet.elastic_net import elastic_net
from elasticnet.elastic_net_cv import elastic_net_cv
from elasticnet.multi_task_elastic_net import multi_task_elastic_net
from elasticnet.multi_task_elastic_net_cv import multi_task_elastic_net_cv

from omp.omp import omp
from omp.omp_cv import omp_cv

from bayesian.bayesian_regression import bayesian_regression
from bayesian.ard_regression import ard_regression

from logistic.logistic_regression import logistic_regression
from logistic.logistic_regression_cv import logistic_regression_cv

from sgd.sgd_regression import sgd_regression
from sgd.sgd_classifier import sgd_classifier

from perceptron.perception import perceptron

from passive_aggressive.passive_aggressive_regressor import passive_agressive_regressor
from passive_aggressive.passive_aggressive_classifier import passive_aggressive_classifier

from robustness.ransac_regression import ransac_regression
from robustness.theil_sen_regression import theil_sen_regression
from robustness.huber_regression import huber_regression

if SKLEARN_LINEAR_MODEL_TABLE["TweedieRegressor"] != NOT_SUPPORTED:
    from glm.tweedie_regression import tweedie_regression
if SKLEARN_LINEAR_MODEL_TABLE["PoissonRegressor"] != NOT_SUPPORTED:
    from glm.poisson_regression import poisson_regression
if SKLEARN_LINEAR_MODEL_TABLE["GammaRegressor"] != NOT_SUPPORTED:
    from glm.gamma_regression import gamma_regression
if SKLEARN_LINEAR_MODEL_TABLE["SGDOneClassSVM"] != NOT_SUPPORTED:
    from sgd.sgd_oneclass_svm import sgd_oneclass_svm
if SKLEARN_LINEAR_MODEL_TABLE["QuantileRegressor"] != NOT_SUPPORTED:
    from quantile.quantile import quantile_regressor

LINEAR_MODELS = {
    "LINEAR_REGRESSION": [linear_regression],
    "RIDGE_REGRESSION_AND_CLASSIFICATION": [
        ridge_regression,
        ridge_regression_cv,
        ridge_classifier,
        ridge_classifier_cv],
    "LASSO_AND_LARS": [
        lasso,
        lasso_cv,
        lasso_lars,
        lasso_lars_cv,
        lasso_lars_ic,
        multi_task_lasso,
        multi_task_lasso_cv],
    "ELASTIC_NET": [
        elastic_net,
        elastic_net_cv],
    "MULTI_CLASS_ELASTIC_NET": [
        multi_task_elastic_net,
        multi_task_elastic_net_cv],
    "OMP": [
        omp,
        omp_cv],
    "BAYESIAN_REGRESSION": [
        bayesian_regression,
        ard_regression],
    "LOGISTIC_REGRESSION": [
        logistic_regression,
        logistic_regression_cv],
    "GLM": [
        tweedie_regression if SKLEARN_LINEAR_MODEL_TABLE["TweedieRegressor"] != NOT_SUPPORTED else (None,"TweedieRegressor"),
        poisson_regression if SKLEARN_LINEAR_MODEL_TABLE["PoissonRegressor"] != NOT_SUPPORTED else (None,"PoissonRegressor"),
        gamma_regression if SKLEARN_LINEAR_MODEL_TABLE["GammaRegressor"] != NOT_SUPPORTED else (None,"GammaRegressor")],
    "SGD": [
        sgd_regression,
        sgd_classifier,
        sgd_oneclass_svm if SKLEARN_LINEAR_MODEL_TABLE["SGDOneClassSVM"] != NOT_SUPPORTED else (None,"SGDOneClassSVM")],
    "PERCEPTRON": [perceptron],
    "PASSIVE_AGGRESSIVE_REGRESSION_AND_CLASSIFIER": [
        passive_agressive_regressor,
        passive_aggressive_classifier],
    "ROBUSTNESS_REGRESSION": [
        ransac_regression,
        theil_sen_regression,
        huber_regression],
    "QUANTILE_REGRESSION": [quantile_regressor if SKLEARN_LINEAR_MODEL_TABLE["QuantileRegressor"] != NOT_SUPPORTED else (None,"QuantileRegressor")]}

@pytest.fixture(scope="session", autouse=True)
def reset_exported_models_directory():
    exported_models_directory = os.path.join(
        os.getcwd(), "tests", "exported_linear_models")
    if not os.path.isdir(exported_models_directory):
        os.mkdir(exported_models_directory)
        return
    for file_name in os.listdir(exported_models_directory):
        # construct full file path
        json_file = os.path.join(exported_models_directory, file_name)
        if os.path.isfile(json_file):
            os.remove(json_file)

def test_full():
    for category in LINEAR_MODELS:
        for model in LINEAR_MODELS[category]:
            if isinstance(model, tuple):
                func, model_name = model
                if func == None:
                    print("Model: " + model_name + " is not supported in this python version.")
                    continue
            model()
