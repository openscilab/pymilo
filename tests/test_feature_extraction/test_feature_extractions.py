import os
import pytest
from count_vectorizer import count_vectorizer
from dict_vectorizer import dict_vectorizer
from feature_hasher import feature_hasher
from hashing_vectorizer import hashing_vectorizer
from patch_extractor import patch_extractor
from tfidf_transformer import tfidf_transformer
from tfidf_vectorizer import tfidf_vectorizer

FEATURE_EXTRACTIONS = [
    count_vectorizer,
    dict_vectorizer,
    feature_hasher,
    hashing_vectorizer,
    patch_extractor,
    tfidf_transformer,
    tfidf_vectorizer,
]

@pytest.fixture(scope="session", autouse=True)
def reset_exported_models_directory():
    exported_models_directory = os.path.join(
        os.getcwd(), "tests", "exported_feature_extraction")
    if not os.path.isdir(exported_models_directory):
        os.mkdir(exported_models_directory)
        return
    for file_name in os.listdir(exported_models_directory):
        json_file = os.path.join(exported_models_directory, file_name)
        if os.path.isfile(json_file):
            os.remove(json_file)

def test_full():
    for model in FEATURE_EXTRACTIONS:
        if isinstance(model, tuple):
            func, model_name = model
            if func == None:
                print("Model: " + model_name + " is not supported in this python version.")
                continue
        model()
