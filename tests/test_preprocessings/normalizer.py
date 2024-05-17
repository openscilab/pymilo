from numpy import array_equal
from sklearn.preprocessing import Normalizer
from pymilo.utils.test_pymilo import report_status
from pymilo.transporters.preprocessing_transporter import PreprocessingTransporter
from util import get_path, write_and_read

MODEL_NAME = "Normalizer"

def normalizer():
    X = [[4, 1, 2, 2],
         [1, 3, 9, 3],
         [5, 7, 5, 1]]
    
    _normalizer = Normalizer().fit(X)
    pre_result = _normalizer.transform(X)

    pt = PreprocessingTransporter()
    post_pymilo_pre_model = pt.deserialize_pre_module(
        write_and_read(
            pt.serialize_pre_module(_normalizer),
            get_path(MODEL_NAME)))    
    post_result = post_pymilo_pre_model.transform(X)

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
