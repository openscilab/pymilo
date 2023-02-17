import os
import unittest

from linear_regression.test_linear_regression import test_linear_regression

from ridge.test_ridge_regression import test_ridge_regression
from ridge.test_ridge_regression_cv import test_ridge_regression_cv
from ridge.test_ridge_classifier import test_ridge_classifier
from ridge.test_ridge_classifier_cv import test_ridge_classifier_cv

from lasso_lars.test_lasso import test_lasso
from lasso_lars.test_lasso_cv import test_lasso_cv
from lasso_lars.test_lasso_lars import test_lasso_lars
from lasso_lars.test_lasso_lars_cv import test_lasso_lars_cv
from lasso_lars.test_lasso_lars_ic import test_lasso_lars_ic
from lasso_lars.test_multi_task_lasso import test_multi_task_lasso
from lasso_lars.test_multi_task_lasso_cv import test_multi_task_lasso_cv

from elasticnet.test_elastic_net import test_elastic_net
from elasticnet.test_elastic_net_cv import test_elastic_net_cv
from elasticnet.test_multi_task_elastic_net import test_multi_task_elastic_net
from elasticnet.test_multi_task_elastic_net_cv import test_multi_task_elastic_net_cv

from omp.test_omp import test_omp
from omp.test_omp_cv import test_omp_cv

from bayesian.test_bayesian_regression import test_bayesian_regression
from bayesian.test_ard_regression import test_ard_regression

from logistic.test_logistic_regression import test_logistic_regression
from logistic.test_logistic_regression_cv import test_logistic_regression_cv

try: 
    from glm.test_tweedie_regression import test_tweedie_regression
    from glm.test_poisson_regression import test_poisson_regression
    from glm.test_gamma_regression import test_gamma_regression
except: 
    ""

from sgd.test_sgd_regression import test_sgd_regression
from sgd.test_sgd_classifier import test_sgd_classifier

try:
    from sgd.test_sgd_oneclass_svm import test_sgd_oneclass_svm
except:
    ""
    
from perceptron.test_perception import test_perceptron

from passive_aggressive.test_passive_aggressive_regressor import test_passive_agressive_regressor
from passive_aggressive.test_passive_aggressive_classifier import test_passive_aggressive_classifier

from robustness.test_ransac_regression import test_ransac_regression
from robustness.test_theil_sen_regression import test_theil_sen_regression
from robustness.test_huber_regression import test_huber_regression

try:
    from quantile.test_quantile import test_quantile_regressor
except:
    ""
    
from pymilo.pymilo_param import SKLEARN_MODEL_TABLE, NOT_SUPPORTED
class TestStringMethods(unittest.TestCase):

    LINEAR_MODELS = {
        "LINEAR_REGRESSION": [test_linear_regression],
        "RIDGE_REGRESSION_AND_CLASSIFICATION": [
            test_ridge_regression,
            test_ridge_regression_cv,
            test_ridge_classifier,
            test_ridge_classifier_cv],
        "LASSO_AND_LARS": [
            test_lasso,
            test_lasso_cv,
            test_lasso_lars,
            test_lasso_lars_cv,
            test_lasso_lars_ic,
            test_multi_task_lasso,
            test_multi_task_lasso_cv],
        "ELASTIC_NET": [
            test_elastic_net,
            test_elastic_net_cv],
        "MULTI_CLASS_ELASTIC_NET": [
            test_multi_task_elastic_net,
            test_multi_task_elastic_net_cv],
        "OMP": [
            test_omp,
            test_omp_cv],
        "BAYESIAN_REGRESSION": [
            test_bayesian_regression,
            test_ard_regression],
        "LOGISTIC_REGRESSION": [
            test_logistic_regression,
            test_logistic_regression_cv],
        "GLM": [
            test_tweedie_regression if SKLEARN_MODEL_TABLE["TweedieRegressor"] != NOT_SUPPORTED else None,
            test_poisson_regression if SKLEARN_MODEL_TABLE["PoissonRegressor"] != NOT_SUPPORTED else None,
            test_gamma_regression if SKLEARN_MODEL_TABLE["GammaRegressor"] != NOT_SUPPORTED else None],
        "SGD": [
            test_sgd_regression,
            test_sgd_classifier,
            test_sgd_oneclass_svm if SKLEARN_MODEL_TABLE["SGDOneClassSVM"] != NOT_SUPPORTED else None],
        "PERCEPTRON": [test_perceptron],
        "PASSIVE_AGGRESSIVE_REGRESSION_AND_CLASSIFIER": [
            test_passive_agressive_regressor,
            test_passive_aggressive_classifier],
        "ROBUSTNESS_REGRESSION": [
            test_ransac_regression,
            test_theil_sen_regression,
            test_huber_regression],
        "QUANTILE_REGRESSION": [test_quantile_regressor if SKLEARN_MODEL_TABLE["QuantileRegressor"] != NOT_SUPPORTED else None]}

    def reset_exported_models_directory(self):
        exported_models_directory = os.path.join(
            os.getcwd(), "tests", "exported_models")
        if not os.path.isdir(exported_models_directory):
            os.mkdir(exported_models_directory)
            return
        for file_name in os.listdir(exported_models_directory):
            # construct full file path
            json_file = os.path.join(exported_models_directory, file_name)
            if os.path.isfile(json_file):
                os.remove(json_file)

    def test_full(self):
        self.reset_exported_models_directory()
        for category in self.LINEAR_MODELS.keys():
            category_all_test_pass = True
            for model in self.LINEAR_MODELS[category]:
                if model == None:
                    print("This particular model is not supported in this python version.")
                    continue
                try:
                    category_all_test_pass = category_all_test_pass and model()
                    self.assertTrue(category_all_test_pass)
                except:
                    print("model: " + model + " of category: " + category + " failed.")


if __name__ == '__main__':
    unittest.main()
