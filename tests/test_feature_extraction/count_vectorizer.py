from numpy import array_equal, random
from sklearn.datasets import load_sample_images
from sklearn.feature_extraction.text import CountVectorizer
from pymilo.transporters.feature_extraction_transporter import FeatureExtractorTransporter
from pymilo.utils.test_pymilo import report_status
from util import get_path, write_and_read

MODEL_NAME = "CountVectorizer"

def count_vectorizer():
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    cv = CountVectorizer(analyzer='word', ngram_range=(2, 2))
    X = cv.fit_transform(corpus)
    pre_result = X.toarray()

    fe = FeatureExtractorTransporter()
    print("before: \n",cv.__dict__)
    post_pymilo_pre_model = fe.deserialize_fe_module(
        write_and_read(
            fe.serialize_fe_module(cv),
            get_path(MODEL_NAME)))
    print("before: \n",post_pymilo_pre_model.__dict__)
    post_result = post_pymilo_pre_model.fit_transform(corpus).toarray()

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
