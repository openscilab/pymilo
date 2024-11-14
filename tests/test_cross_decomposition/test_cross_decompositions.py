import os
import pytest

from pls_regression import pls_regressor
from pls_canonical import pls_canonical
from cca import cca

CROSS_DECOMPOSITIONS = {
    "PLS_REGRESSION": [pls_regressor],
    "PLS_CANONICAL": [pls_canonical],
    "CCA": [cca],
}

@pytest.fixture(scope="session", autouse=True)
def reset_exported_models_directory():
    exported_models_directory = os.path.join(
        os.getcwd(), "tests", "exported_cross_decomposition")
    if not os.path.isdir(exported_models_directory):
        os.mkdir(exported_models_directory)
        return
    for file_name in os.listdir(exported_models_directory):
        # construct full file path
        json_file = os.path.join(exported_models_directory, file_name)
        if os.path.isfile(json_file):
            os.remove(json_file)

def test_full():
    for category in CROSS_DECOMPOSITIONS:
        for model in CROSS_DECOMPOSITIONS[category]:
            model()
