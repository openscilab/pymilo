from sklearn.cross_decomposition import CCA
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "CCA"

def cca():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    cca = CCA(n_components=1).fit(x_train, y_train)
    pymilo_regression_test(cca, MODEL_NAME, (x_test, y_test))
