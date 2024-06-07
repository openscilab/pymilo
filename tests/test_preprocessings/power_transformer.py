from numpy import array_equal
from sklearn.preprocessing import PowerTransformer
from pymilo.utils.test_pymilo import report_status
from pymilo.transporters.preprocessing_transporter import PreprocessingTransporter
from util import get_path, write_and_read

MODEL_NAME = "PowerTransformer"

def power_transformer():
    power_transformer = PowerTransformer()
    X = [[1, 2], [3, 2], [4, 5]]
    power_transformer = power_transformer.fit(X)
    pre_result = power_transformer.transform(X)

    pt = PreprocessingTransporter()
    post_pymilo_pre_model = pt.deserialize_pre_module(
        write_and_read(
            pt.serialize_pre_module(power_transformer),
            get_path(MODEL_NAME)))
    post_result = post_pymilo_pre_model.transform(X)

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
