from sklearn.svm import LinearSVR

from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "LinearSVR"

def linear_svr():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    linear_svr = LinearSVR(random_state=3, tol=1e-5).fit(x_train, y_train)
    pymilo_regression_test(linear_svr, MODEL_NAME, (x_test, y_test))
