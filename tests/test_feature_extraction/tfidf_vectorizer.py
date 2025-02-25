from numpy import array_equal
from util import get_path, write_and_read
from pymilo.utils.test_pymilo import report_status
from sklearn.feature_extraction.text import TfidfVectorizer
from pymilo.transporters.feature_extraction_transporter import FeatureExtractorTransporter

MODEL_NAME = "TfidfVectorizer"

def tfidf_vectorizer():
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(corpus)
    pre_result = X.toarray()

    fe = FeatureExtractorTransporter()

    post_pymilo_pre_model = fe.deserialize_fe_module(
        write_and_read(
            fe.serialize_fe_module(tfidf),
            get_path(MODEL_NAME)))

    post_result = post_pymilo_pre_model.fit_transform(corpus).toarray()

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
