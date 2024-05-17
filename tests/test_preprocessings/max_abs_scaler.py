from numpy import array_equal
from sklearn.preprocessing import MaxAbsScaler
from pymilo.utils.test_pymilo import report_status
from pymilo.transporters.preprocessing_transporter import PreprocessingTransporter
from util import get_path, write_and_read

MODEL_NAME = "MaxAbsScaler"

def max_abs_scaler():
    X = [[ 1., -1.,  2.],
        [ 2.,  0.,  0.],
        [ 0.,  1., -1.]]

    _max_abs_scaler = MaxAbsScaler().fit(X)
    pre_result = _max_abs_scaler.transform(X)

    pt = PreprocessingTransporter()
    post_pymilo_pre_model = pt.deserialize_pre_module(
        write_and_read(
            pt.serialize_pre_module(_max_abs_scaler),
            get_path(MODEL_NAME)))
    post_result = post_pymilo_pre_model.transform(X)

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
