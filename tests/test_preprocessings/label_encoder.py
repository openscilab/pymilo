from numpy import array_equal
from sklearn.preprocessing import LabelEncoder
from pymilo.transporters.preprocessing_transporter import PreprocessingTransporter
from pymilo.utils.test_pymilo import report_status
from util import get_path, write_and_read

MODEL_NAME = "LabelEncoder"

def label_encoder():
    X = ["paris", "paris", "tokyo", "amsterdam"]

    le = LabelEncoder().fit(X)
    pre_result = le.transform(X)

    pt = PreprocessingTransporter()
    post_pymilo_pre_model = pt.deserialize_pre_module(
        write_and_read(
            pt.serialize_pre_module(le),
            get_path(MODEL_NAME)))

    post_result = post_pymilo_pre_model.transform(X)

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
