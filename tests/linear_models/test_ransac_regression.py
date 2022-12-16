from  sklearn.linear_model import RANSACRegressor
from test_pymilo import test_pymilo_regression
from data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "RANSAC-Regressor"

def test_ransac_regression():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create ransac regression object
    ransac_random_state = 3
    ransac_regression = RANSACRegressor(random_state= ransac_random_state)
    # Train the model using the training sets
    ransac_regression.fit(x_train, y_train)
    return test_pymilo_regression(ransac_regression, MODEL_NAME, (x_test,y_test))
    