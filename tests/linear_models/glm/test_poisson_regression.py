from  sklearn.linear_model import PoissonRegressor
from pymilo.utils.test_pymilo import test_pymilo_regression
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Poisson-Regression"

def test_poisson_regression():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Poisson regression object
    poisson_alpha = 0.5
    poisson_regression = PoissonRegressor(alpha= poisson_alpha)
    # Train the model using the training sets
    poisson_regression.fit(x_train, y_train)
    return test_pymilo_regression(poisson_regression,MODEL_NAME,(x_test,y_test))