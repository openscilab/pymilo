from sklearn.tree import DecisionTreeRegressor

from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

import numpy as np

MODEL_NAME = "Decision Tree Regressor"

def decision_tree_regression():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Decision Tree Regressor
    decision_tree_regressor = DecisionTreeRegressor(random_state=1)
    decision_tree_regressor = decision_tree_regressor.fit(x_train, y_train)
    assert pymilo_regression_test(
        decision_tree_regressor, MODEL_NAME, (x_test, y_test)) == True 
