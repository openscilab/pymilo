from sklearn.svm import NuSVC

from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "NuSVC"

def nu_svc():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    nu_svc = NuSVC().fit(x_train, y_train)
    pymilo_classification_test(nu_svc, MODEL_NAME, (x_test, y_test))
