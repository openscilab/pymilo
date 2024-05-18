from numpy import array_equal, arange
from sklearn.preprocessing import SplineTransformer
from pymilo.utils.test_pymilo import report_status
from pymilo.transporters.preprocessing_transporter import PreprocessingTransporter
from util import get_path, write_and_read

MODEL_NAME = "SplineTransformer"

def spline_transformer():
    X = arange(6).reshape(6, 1)
    spline = SplineTransformer(degree=2, n_knots=3)
    pre_result = spline.fit_transform(X)

    pt = PreprocessingTransporter()
    post_pymilo_pre_model = pt.deserialize_pre_module(
        write_and_read(
            pt.serialize_pre_module(spline),
            get_path(MODEL_NAME)))
    post_result = post_pymilo_pre_model.fit_transform(X)

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
