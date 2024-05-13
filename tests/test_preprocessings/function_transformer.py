from numpy import array_equal, log1p
from sklearn.preprocessing import FunctionTransformer
from pymilo.transporters.preprocessing_transporter import PreprocessingTransporter
from pymilo.utils.test_pymilo import report_status
from pymilo.utils.util import import_function
from util import get_path, write_and_read

MODEL_NAME = "FunctionTransformer"

def function_transformer():
    f = log1p
    X = [[0, 1], [2, 3]]

    _function_transformer = FunctionTransformer(log1p).fit(X)
    pre_result = _function_transformer.transform(X)

    pt = PreprocessingTransporter()
    serialized_module = pt.serialize_pre_module(_function_transformer)
    file_addr = get_path(MODEL_NAME)
    post_pymilo_pre_model = pt.deserialize_pre_module(write_and_read(serialized_module, file_addr))

    post_result = post_pymilo_pre_model.transform(X)
    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
