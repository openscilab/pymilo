import os
import pytest
from pymilo.pymilo_param import SKLEARN_PREPROCESSING_TABLE, NOT_SUPPORTED

from one_hot_encoder import one_hot_encoder
from label_binarizer import label_binarizer
from label_encoder import label_encoder
from standard_scaler import standard_scaler
from binarizer import binarizer
from function_transformer import function_transformer
from kernel_centerer import kernel_centerer
from multilabel_binarizer import multilabel_binarizer
from max_abs_scaler import max_abs_scaler
from normalizer import normalizer
from ordinal_encoder import ordinal_encoder
from polynomial_features import polynomial_features
from robust_scaler import robust_scaler
from quantile_transformer import quantile_transformer
from kbins_discretizer import kbins_discretizer
from power_transformer import power_transformer

if SKLEARN_PREPROCESSING_TABLE["SplineTransformer"] != NOT_SUPPORTED:
    from spline_transformer import spline_transformer
if SKLEARN_PREPROCESSING_TABLE["TargetEncoder"] != NOT_SUPPORTED:
    from target_encoder import target_encoder

PREPROCESSINGS = [one_hot_encoder,
                  label_binarizer,
                  label_encoder,
                  standard_scaler,
                  binarizer,
                  function_transformer,
                  kernel_centerer,
                  multilabel_binarizer,
                  max_abs_scaler,
                  normalizer,
                  ordinal_encoder,
                  polynomial_features,
                  robust_scaler,
                  quantile_transformer,
                  kbins_discretizer,
                  power_transformer,
                  spline_transformer if SKLEARN_PREPROCESSING_TABLE["SplineTransformer"] != NOT_SUPPORTED else (None, "SplineTransformer"),
                  target_encoder if SKLEARN_PREPROCESSING_TABLE["TargetEncoder"] != NOT_SUPPORTED else (None, "TargetEncoder")
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
    for model in PREPROCESSINGS:
        if isinstance(model, tuple):
            func, model_name = model
            if func == None:
                print("Model: " + model_name + " is not supported in this python version.")
                continue
        model()
