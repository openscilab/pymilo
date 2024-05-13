from numpy import array_equal
from sklearn.preprocessing import LabelBinarizer
from pymilo.transporters.preprocessing_transporter import PreprocessingTransporter
from pymilo.utils.test_pymilo import report_status
from util import get_path, write_and_read

MODEL_NAME = "LabelBinarizer"

def label_binarizer():
    X = ['yes', 'no', 'no', 'yes']

    lb = LabelBinarizer().fit(X)
    pre_result = lb.transform(X)

    pt = PreprocessingTransporter()
    serialized_module = pt.serialize_pre_module(lb)
    file_addr = get_path(MODEL_NAME)
    post_pymilo_pre_model = pt.deserialize_pre_module(write_and_read(serialized_module, file_addr))

    post_result = post_pymilo_pre_model.transform(X)

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
