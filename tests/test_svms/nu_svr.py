from sklearn.svm import NuSVR

from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "NuSVC"

def nu_svr():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    nu_svr = NuSVR(C=1.0, nu=0.1).fit(x_train, y_train)
    pymilo_regression_test(nu_svr, MODEL_NAME, (x_test, y_test))
