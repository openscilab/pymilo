import os
import pytest

from pymilo.pymilo_param import SKLEARN_SVM_TABLE

from linear_svc import linear_svc
from linear_svr import linear_svr
from nu_svc import nu_svc
from nu_svr import nu_svr
from one_class_svm import one_class_svm
from svc import svc
from svr import svr 

SVMS = {
    "LINEAR": [linear_svc, linear_svr],
    "Nu": [nu_svc, nu_svr],
    "ONE_CLASS": [one_class_svm],
    "SVC": [svc],
    "SVR": [svr],
}

@pytest.fixture(scope="session", autouse=True)
def reset_exported_models_directory():
    exported_models_directory = os.path.join(
        os.getcwd(), "tests", "exported_svms")
    if not os.path.isdir(exported_models_directory):
        os.mkdir(exported_models_directory)
        return
    for file_name in os.listdir(exported_models_directory):
        # construct full file path
        json_file = os.path.join(exported_models_directory, file_name)
        if os.path.isfile(json_file):
            os.remove(json_file)

def test_full():
    for category in SVMS:
        for model in SVMS[category]:
            return # model()
