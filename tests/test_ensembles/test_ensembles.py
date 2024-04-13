import os
import pytest

from adaboost.adaboost_regressor import adaboost_regressor
from adaboost.adaboost_classifier import adaboost_classifier

from bagging.bagging_regressor import bagging_regressor
from bagging.bagging_classifier import bagging_classifier

from extra_trees.extra_trees_regressor import extra_trees_regressor
from extra_trees.extra_trees_classifier import extra_trees_classifier

from gradient_booster.gradient_booster_regressor import gradient_booster_regressor
from gradient_booster.gradient_booster_classifier import gradient_booster_classifier

from random_forests.random_forest_regressor import random_forest_regressor
from random_forests.random_forest_classifier import random_forest_classifier
from isolation_forest import isolation_forest
from random_trees_embedding import random_trees_embedding


from pymilo.pymilo_param import SKLEARN_ENSEMBLE_TABLE, NOT_SUPPORTED

if SKLEARN_ENSEMBLE_TABLE["HistGradientBoostingRegressor"] != NOT_SUPPORTED:
    from hist_gradient_boosting.hist_gradient_boosting_regressor import hist_gradient_boosting_regressor
    from hist_gradient_boosting.hist_gradient_boosting_classifier import hist_gradient_boosting_classifier

ENSEMBLES = {
    "Adaboost": [adaboost_regressor, adaboost_classifier],
    "Bagging": [bagging_regressor, bagging_classifier], 
    "ExtaTrees": [extra_trees_regressor, extra_trees_classifier],
    "GradientBooster": [gradient_booster_regressor, gradient_booster_classifier],
    "HistGradientBooster": [
        hist_gradient_boosting_regressor if SKLEARN_ENSEMBLE_TABLE["HistGradientBoostingRegressor"] != NOT_SUPPORTED else (None, "HistGradientBoostingRegressor"),
        hist_gradient_boosting_classifier if SKLEARN_ENSEMBLE_TABLE["HistGradientBoostingClassifier"] != NOT_SUPPORTED else (None, "HistGradientBoostingClassifier")
        ],
    "Forests": [random_forest_regressor, random_forest_classifier, isolation_forest, random_trees_embedding],
}

@pytest.fixture(scope="session", autouse=True)
def reset_exported_models_directory():
    exported_models_directory = os.path.join(
        os.getcwd(), "tests", "exported_ensembles")
    if not os.path.isdir(exported_models_directory):
        os.mkdir(exported_models_directory)
        return
    for file_name in os.listdir(exported_models_directory):
        # construct full file path
        json_file = os.path.join(exported_models_directory, file_name)
        if os.path.isfile(json_file):
            os.remove(json_file)

def test_full():
    for category in ENSEMBLES:
        for model in ENSEMBLES[category]:
            if isinstance(model, tuple):
                func, model_name = model
                if func == None:
                    print("Model: " + model_name + " is not supported in this python version.")
                    continue
            model()
