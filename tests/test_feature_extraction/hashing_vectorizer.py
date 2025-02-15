from numpy import array_equal
from util import get_path, write_and_read
from pymilo.utils.test_pymilo import report_status
from sklearn.feature_extraction.text import HashingVectorizer
from pymilo.transporters.feature_extraction_transporter import FeatureExtractorTransporter

MODEL_NAME = "HashingVectorizer"

def hashing_vectorizer():
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    hv = HashingVectorizer(n_features=2**4)
    X = hv.fit_transform(corpus)

    pre_result = X.toarray()
    fe = FeatureExtractorTransporter()
    post_pymilo_pre_model = fe.deserialize_fe_module(
        write_and_read(
            fe.serialize_fe_module(hv),
            get_path(MODEL_NAME)))
    post_result = post_pymilo_pre_model.fit_transform(corpus).toarray()

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
