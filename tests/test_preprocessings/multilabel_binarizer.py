from numpy import array_equal
from sklearn.preprocessing import MultiLabelBinarizer
from pymilo.utils.test_pymilo import report_status
from pymilo.transporters.preprocessing_transporter import PreprocessingTransporter
from util import get_path, write_and_read

MODEL_NAME = "MultiLabelBinarizer"

def multilabel_binarizer():
    X = [{'sci-fi', 'thriller'}, {'comedy'}]

    mlb = MultiLabelBinarizer().fit(X)
    pre_result = mlb.transform(X)

    pt = PreprocessingTransporter()
    post_pymilo_pre_model = pt.deserialize_pre_module(
        write_and_read(
            pt.serialize_pre_module(mlb),
            get_path(MODEL_NAME)))

    post_result = post_pymilo_pre_model.transform(X)

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
