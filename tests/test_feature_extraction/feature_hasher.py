from numpy import array_equal
from sklearn.feature_extraction import FeatureHasher
from pymilo.transporters.feature_extraction_transporter import FeatureExtractorTransporter
from pymilo.utils.test_pymilo import report_status
from util import get_path, write_and_read

MODEL_NAME = "FeatureHasher"

def feature_hasher():
    h = FeatureHasher(n_features=10)
    D = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]
    f = h.transform(D)

    pre_result = f.toarray()

    fe = FeatureExtractorTransporter()
    post_pymilo_pre_model = fe.deserialize_fe_module(
        write_and_read(
            fe.serialize_fe_module(h),
            get_path(MODEL_NAME)))
    post_result = post_pymilo_pre_model.transform(D).toarray()

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
