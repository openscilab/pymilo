import os
import pytest

from one_hot_encoder import one_hot_encoder
from label_binarizer import label_binarizer
from label_encoder import label_encoder
from standard_scaler import standard_scaler
from binarizer import binarizer
from function_transformer import function_transformer
from kbins_dicretizer import kbins_dicretizer

PREPROCESSINGS = [one_hot_encoder,
                  label_binarizer,
                  label_encoder,
                  standard_scaler,
                  binarizer,
                  function_transformer,
                  kbins_dicretizer
                  ]

@pytest.fixture(scope="session", autouse=True)
def reset_exported_models_directory():
    exported_models_directory = os.path.join(
        os.getcwd(), "tests", "exported_preprocessings")
    if not os.path.isdir(exported_models_directory):
        os.mkdir(exported_models_directory)
        return
    for file_name in os.listdir(exported_models_directory):
        # construct full file path
        json_file = os.path.join(exported_models_directory, file_name)
        if os.path.isfile(json_file):
            os.remove(json_file)

def test_full():
    for pre in PREPROCESSINGS:
        pre()
