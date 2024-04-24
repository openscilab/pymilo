from sklearn.ensemble import IsolationForest
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "IsolationForest"

def isolation_forest():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    isolation_forest = IsolationForest(random_state=0).fit(x_train, y_train)
    pymilo_regression_test(isolation_forest, MODEL_NAME, (x_test, y_test))
