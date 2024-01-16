from sklearn.tree import ExtraTreeRegressor

from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Extra Tree Regressor"

def extra_tree_regression():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Decision Tree Regressor
    extra_tree_regressor = ExtraTreeRegressor(random_state=0)
    extra_tree_regressor = extra_tree_regressor.fit(x_train, y_train)
    assert pymilo_regression_test(
        extra_tree_regressor, MODEL_NAME, (x_test, y_test)) == True
