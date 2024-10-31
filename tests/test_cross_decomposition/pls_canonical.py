from sklearn.cross_decomposition import PLSCanonical
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "PLSCanonical"

def pls_canonical():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    pls_canonical = PLSCanonical(n_components=1).fit(x_train, y_train)
    pymilo_regression_test(pls_canonical, MODEL_NAME, (x_test, y_test))
