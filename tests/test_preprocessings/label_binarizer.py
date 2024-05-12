from numpy import array_equal
from sklearn.preprocessing import LabelBinarizer
from pymilo.transporters.preprocessing_transporter import PreprocessingTransporter
from pymilo.utils.test_pymilo import report_status

MODEL_NAME = "LabelBinarizer"

def label_binarizer():
    X = ['yes', 'no', 'no', 'yes']

    lb = LabelBinarizer().fit(X)
    pre_result = lb.transform(X)

    pt = PreprocessingTransporter()
    post_pymilo_pre_model = pt.deserialize_pre_module(
        pt.serialize_pre_module(lb)
    )
    post_result = post_pymilo_pre_model.transform(X)

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
