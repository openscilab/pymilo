import os
import pytest

from pymilo.pymilo_param import XGBOOST_MODEL_TABLE, NOT_SUPPORTED
from xgb_classifier import xgb_classifier
from xgb_classifier_multiclass import xgb_classifier_multiclass
from xgb_regressor import xgb_regressor
from xgb_ranker import xgb_ranker
from xgb_rf_classifier import xgb_rf_classifier
from xgb_rf_regressor import xgb_rf_regressor
from xgb_model import xgb_model
from xgb_booster import xgb_booster
from xgb_dart import xgb_dart
from xgb_gblinear import xgb_gblinear
from xgb_early_stopping import xgb_early_stopping
from xgb_pipeline import xgb_pipeline
from xgb_feature_names import xgb_feature_names

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

XGBOOSTS = {
    "CLASSIFIER": [xgb_classifier, xgb_classifier_multiclass],
    "REGRESSOR": [xgb_regressor],
    "RANKER": [xgb_ranker],
    "RANDOM_FOREST": [xgb_rf_classifier, xgb_rf_regressor],
    "GENERIC": [xgb_model, xgb_booster],
    "BOOSTER_TYPES": [xgb_dart, xgb_gblinear],
    "FIT_VARIANTS": [xgb_early_stopping, xgb_feature_names],
    "COMPOSE": [xgb_pipeline],
}


@pytest.fixture(scope="session", autouse=True)
def reset_exported_models_directory():
    exported_models_directory = os.path.join(
        os.getcwd(), "tests", "exported_xgboosts")
    if not os.path.isdir(exported_models_directory):
        os.mkdir(exported_models_directory)
        return
    for file_name in os.listdir(exported_models_directory):
        json_file = os.path.join(exported_models_directory, file_name)
        if os.path.isfile(json_file):
            os.remove(json_file)


def test_full():
    if XGBOOST_MODEL_TABLE.get("XGBClassifier") == NOT_SUPPORTED:
        print("XGBoost is not installed; skipping XGBoost model tests.")
        return
    for category in XGBOOSTS:
        for model in XGBOOSTS[category]:
            print("Testing model: ", model.__name__)
            model()
