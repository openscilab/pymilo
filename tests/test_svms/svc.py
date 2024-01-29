from sklearn.svm import SVC

from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "SVC"

def svc():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    svc = SVC(gamma='auto').fit(x_train, y_train)
    pymilo_classification_test(svc, MODEL_NAME, (x_test, y_test))
