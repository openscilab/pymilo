from sklearn.svm import SVR

from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "SVR"

def svr():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    svr = SVR(C=1.0, epsilon=0.2).fit(x_train, y_train)
    pymilo_regression_test(svr, MODEL_NAME, (x_test, y_test))
