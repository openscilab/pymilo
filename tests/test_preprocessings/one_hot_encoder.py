from numpy import array_equal
from sklearn.preprocessing import OneHotEncoder
from pymilo.utils.test_pymilo import report_status
from pymilo.transporters.preprocessing_transporter import PreprocessingTransporter
from util import get_path, write_and_read

MODEL_NAME = "OneHotEncoder"

def one_hot_encoder():
    X = [['Male', 1], ['Female', 3], ['Female', 2]]

    _one_hot_encoder = OneHotEncoder(handle_unknown='ignore').fit(X)
    pre_result = _one_hot_encoder.transform(X).toarray()

    pt = PreprocessingTransporter()
    serialized_module = pt.serialize_pre_module(_one_hot_encoder)
    file_addr = get_path(MODEL_NAME)
    post_pymilo_pre_model = pt.deserialize_pre_module(write_and_read(serialized_module, file_addr))

    post_result = post_pymilo_pre_model.transform(X).toarray()

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
