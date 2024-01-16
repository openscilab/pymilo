from sklearn.linear_model import RANSACRegressor
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "RANSAC-Regressor"


def ransac_regression():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create ransac regression object
    ransac_random_state = 3
    ransac_regression = RANSACRegressor(random_state=ransac_random_state)
    # Train the model using the training sets
    ransac_regression.fit(x_train, y_train)
    assert pymilo_regression_test(
        ransac_regression, MODEL_NAME, (x_test, y_test)) == True
