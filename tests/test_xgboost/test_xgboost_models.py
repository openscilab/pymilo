import os
import pytest

xgboost = pytest.importorskip("xgboost")

from xgb_regressor import xgb_regressor_all_objectives
from xgb_classifier_binary import xgb_classifier_binary_all_objectives
from xgb_classifier_multiclass import xgb_classifier_multiclass_all_objectives
from xgb_ranker import xgb_ranker_all_objectives
from xgb_rf_regressor import xgb_rf_regressor
from xgb_rf_classifier import xgb_rf_classifier
from xgb_survival import xgb_survival_cox
from native_booster import native_booster_regression, native_booster_classification
from xgb_nested import (
    xgb_inside_pipeline,
    xgb_inside_voting_regressor,
    xgb_untrained_model,
    xgb_json_dump_round_trip,
)


XGBOOST_MODELS = {
    "REGRESSOR": [xgb_regressor_all_objectives],
    "CLASSIFIER_BINARY": [xgb_classifier_binary_all_objectives],
    "CLASSIFIER_MULTICLASS": [xgb_classifier_multiclass_all_objectives],
    "RANKER": [xgb_ranker_all_objectives],
    "RF_REGRESSOR": [xgb_rf_regressor],
    "RF_CLASSIFIER": [xgb_rf_classifier],
    "SURVIVAL": [xgb_survival_cox],
    "NATIVE_BOOSTER": [native_booster_regression, native_booster_classification],
    "NESTED": [
        xgb_inside_pipeline,
        xgb_inside_voting_regressor,
        xgb_untrained_model,
        xgb_json_dump_round_trip,
    ],
}


@pytest.fixture(scope="session", autouse=True)
def reset_exported_xgboost_directory():
    exported_dir = os.path.join(os.getcwd(), "tests", "exported_xgboost")
    if not os.path.isdir(exported_dir):
        os.makedirs(exported_dir, exist_ok=True)
        return
    for file_name in os.listdir(exported_dir):
        json_file = os.path.join(exported_dir, file_name)
        if os.path.isfile(json_file):
            os.remove(json_file)


def test_full():
    for category in XGBOOST_MODELS:
        for func in XGBOOST_MODELS[category]:
            func()
