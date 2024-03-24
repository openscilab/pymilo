from sklearn.svm import LinearSVC

from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "LinearSVC"

def linear_svc():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    linear_svc = LinearSVC(random_state=0, tol=1e-5).fit(x_train, y_train)
    pymilo_classification_test(linear_svc, MODEL_NAME, (x_test, y_test))
