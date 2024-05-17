from numpy import array_equal
from sklearn.preprocessing import OrdinalEncoder
from pymilo.utils.test_pymilo import report_status
from pymilo.transporters.preprocessing_transporter import PreprocessingTransporter
from util import get_path, write_and_read

MODEL_NAME = "OrdinalEncoder"

def ordinal_encoder():
    X = [['Male', 1], ['Female', 3], ['Female,', 2]]
    _ordinal_encoder = OrdinalEncoder().fit(X)
    pre_result = _ordinal_encoder.transform(X)

    pt = PreprocessingTransporter()
    post_pymilo_pre_model = pt.deserialize_pre_module(
        write_and_read(
            pt.serialize_pre_module(_ordinal_encoder),
            get_path(MODEL_NAME)))
    post_result = post_pymilo_pre_model.transform(X)

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
