from sklearn.cross_decomposition import PLSRegression
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "PLSRegression"

def pls_regressor():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    pls_regressor = PLSRegression(n_components=2).fit(x_train, y_train)
    pymilo_regression_test(pls_regressor, MODEL_NAME, (x_test, y_test))
