from numpy import array_equal, array
from sklearn.preprocessing import TargetEncoder
from pymilo.transporters.preprocessing_transporter import PreprocessingTransporter
from pymilo.utils.test_pymilo import report_status
from util import get_path, write_and_read

MODEL_NAME = "TargetEncoder"

def target_encoder():
    X = array([["dog"] * 20 + ["cat"] * 30 + ["snake"] * 38], dtype=object).T
    y = [90.3] * 5 + [80.1] * 15 + [20.4] * 5 + [20.1] * 25 + [21.2] * 8 + [49] * 30
    enc_auto = TargetEncoder(smooth="auto")
    pre_result = enc_auto.fit_transform(X, y)

    pt = PreprocessingTransporter()
    post_pymilo_pre_model = pt.deserialize_pre_module(
        write_and_read(
            pt.serialize_pre_module(enc_auto),
            get_path(MODEL_NAME)))
    post_result = post_pymilo_pre_model.transform(X)

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
