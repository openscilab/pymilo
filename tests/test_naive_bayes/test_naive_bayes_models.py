import os
import pytest

from gaussian import gaussian_naive_bayes
from multinomial import multinomial_naive_bayes
from complement import complement_naive_bayes
from bernoulli import bernoulli_naive_bayes
from categorical import categorical_naive_bayes

NAIVE_BAYES_MODELS = [
    gaussian_naive_bayes,
    multinomial_naive_bayes,
    complement_naive_bayes,
    bernoulli_naive_bayes,
    categorical_naive_bayes
]

@pytest.fixture(scope="session", autouse=True)
def reset_exported_models_directory():
    exported_models_directory = os.path.join(
        os.getcwd(), "tests", "exported_naive_bayes")
    if not os.path.isdir(exported_models_directory):
        os.mkdir(exported_models_directory)
        return
    for file_name in os.listdir(exported_models_directory):
        # construct full file path
        json_file = os.path.join(exported_models_directory, file_name)
        if os.path.isfile(json_file):
            os.remove(json_file)

def test_full():
    for model in NAIVE_BAYES_MODELS:
        model()
