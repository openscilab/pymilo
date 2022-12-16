import os
import unittest

from test_linear_regression import test_linear_regression

from test_ridge_regression import test_ridge_regression
from test_ridge_regression_cv import test_ridge_regression_cv
from test_ridge_classifier import test_ridge_classifier
from test_ridge_classifier_cv import test_ridge_classifier_cv

from test_lasso import test_lasso
from test_lasso_cv import test_lasso_cv
from test_lasso_lars import test_lasso_lars
from test_lasso_lars_cv import test_lasso_lars_cv
from test_lasso_lars_ic import test_lasso_lars_ic

from test_multi_task_lasso import test_multi_task_lasso
from test_multi_task_lasso_cv import test_multi_task_lasso_cv

from test_elastic_net import test_elastic_net
from test_elastic_net_cv import test_elastic_net_cv
from test_multi_task_elastic_net import test_multi_task_elastic_net
from test_multi_task_elastic_net_cv import test_multi_task_elastic_net_cv

from test_omp import test_omp
from test_omp_cv import test_omp_cv

from test_bayesian_regression import test_bayesian_regression
from test_ard_regression import test_ard_regression

from test_logistic_regression import test_logistic_regression
from test_logistic_regression_cv import test_logistic_regression_cv

from test_tweedie_regression import test_tweedie_regression
from test_poisson_regression import test_poisson_regression
from test_gamma_regression import test_gamma_regression

from test_sgd_regression import test_sgd_regression
from test_sgd_classifier import test_sgd_classifier
from test_sgd_oneclass_svm import test_sgd_oneclass_svm

from test_perception import test_perceptron

from test_passive_aggressive_regressor import test_passive_agressive_regressor
from test_passive_aggressive_classifier import test_passive_aggressive_classifier

from test_ransac_regression import test_ransac_regression
from test_theil_sen_regression import test_theil_sen_regression
from test_huber_regression import test_huber_regression

from test_quantile import test_quantile_regressor

class TestStringMethods(unittest.TestCase):

    LINEAR_MODELS = {
        "LINEAR_REGRESSION": [test_linear_regression],
        "RIDGE_REGRESSION_AND_CLASSIFICATION": [test_ridge_regression,test_ridge_regression_cv,test_ridge_classifier,test_ridge_classifier_cv],
        "LASSO_AND_LARS": [test_lasso, test_lasso_cv, test_lasso_lars, test_lasso_lars_cv, test_lasso_lars_ic],
        "MULTI_CLASS_LASSO": [test_multi_task_lasso,test_multi_task_lasso_cv],
        "ELASTIC_NET":[test_elastic_net,test_elastic_net_cv],
        "MULTI_CLASS_ELASTIC_NET": [test_multi_task_elastic_net, test_multi_task_elastic_net_cv],
        "OMP": [test_omp, test_omp_cv],
        "BAYESIAN_REGRESSION": [test_bayesian_regression,test_ard_regression],
        "LOGISTIC_REGRESSION": [test_logistic_regression, test_logistic_regression_cv],
        "GLM": [test_tweedie_regression,test_poisson_regression,test_gamma_regression],
        "SGD": [test_sgd_regression, test_sgd_classifier, test_sgd_oneclass_svm],
        "PERCEPTRON": [test_perceptron],
        "PASSIVE_AGGRESSIVE_REGRESSION_AND_CLASSIFIER": [test_passive_agressive_regressor, test_passive_aggressive_classifier],
        "ROBUSTNESS_REGRESSION": [test_ransac_regression, test_theil_sen_regression, test_huber_regression],
        "QUANTILE_REGRESSION": [test_quantile_regressor] 
    }

    def reset_exported_models_directory(self):
        exported_models_directory = os.path.join(os.getcwd(),"tests","exported_models")
        if(not os.path.isdir(exported_models_directory)):
            os.mkdir(exported_models_directory)
            return 
        for file_name in os.listdir(exported_models_directory):
            print(file_name)
            # construct full file path
            json_file = os.path.join(exported_models_directory,file_name)
            if os.path.isfile(json_file):
                os.remove(json_file)

    def test_full(self):
        self.reset_exported_models_directory()
        for category in self.LINEAR_MODELS.keys():
            category_all_test_pass = True
            for model in self.LINEAR_MODELS[category]:
                category_all_test_pass = category_all_test_pass and model()
            self.assertTrue(category_all_test_pass)

if __name__ == '__main__':
    unittest.main()