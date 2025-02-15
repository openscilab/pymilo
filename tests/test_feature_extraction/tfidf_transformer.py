from numpy import array_equal
from util import get_path, write_and_read
from pymilo.utils.test_pymilo import report_status
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from pymilo.transporters.feature_extraction_transporter import FeatureExtractorTransporter

MODEL_NAME = "TfidfTransformer"

def tfidf_transformer():
    corpus = ['this is the first document',
            'this document is the second document',
            'and this is the third one',
            'is this the first document']
    vocabulary = ['this', 'document', 'first', 'is', 'second', 'the',
                'and', 'one']
    pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
                    ('tfid', TfidfTransformer())]).fit(corpus)

    _tfidf = pipe['tfid']
    pre_result = _tfidf.idf_

    fe = FeatureExtractorTransporter()

    post_pymilo_pre_model = fe.deserialize_fe_module(
        write_and_read(
            fe.serialize_fe_module(_tfidf),
            get_path(MODEL_NAME)))

    post_result = post_pymilo_pre_model.idf_

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
