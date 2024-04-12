import os
import pytest

from adaboost.adaboost_regressor import adaboost_regressor
from adaboost.adaboost_classifier import adaboost_classifier

ENSEMBLES = {
    "Adaboost": [adaboost_regressor, adaboost_classifier],
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
            model()
