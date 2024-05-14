from numpy import array_equal, array
from sklearn.preprocessing import KBinsDiscretizer
from pymilo.utils.test_pymilo import report_status
from pymilo.transporters.preprocessing_transporter import PreprocessingTransporter
from util import get_path, write_and_read

MODEL_NAME = "KBinsDiscretizer"

def kbins_dicretizer():
    X = array([[-2, 1, -4,   -1],
         [-1, 2, -3, -0.5],
         [ 0, 3, -2,  0.5],
         [ 1, 4, -1,    2]])

    est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform', subsample=None).fit(X)
    pre_result = est.transform(X)

    pt = PreprocessingTransporter()
    serialized_module = pt.serialize_pre_module(est)
    file_addr = get_path(MODEL_NAME)
    post_pymilo_pre_model = pt.deserialize_pre_module(write_and_read(serialized_module, file_addr))

    post_result = post_pymilo_pre_model.transform(X)

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
