from numpy import array_equal
from sklearn.feature_extraction import DictVectorizer
from pymilo.transporters.feature_extraction_transporter import FeatureExtractorTransporter
from pymilo.utils.test_pymilo import report_status
from util import get_path, write_and_read

MODEL_NAME = "DictVectorizer"

def dict_vectorizer():
    v = DictVectorizer(sparse=False)
    D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
    _ = v.fit_transform(D)

    pre_result = v.transform({'foo': 4, 'unseen_feature': 3})
    
    fe = FeatureExtractorTransporter()
    post_pymilo_pre_model = fe.deserialize_fe_module(
        write_and_read(
            fe.serialize_fe_module(v),
            get_path(MODEL_NAME)))
    post_result = post_pymilo_pre_model.transform({'foo': 4, 'unseen_feature': 3})

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
