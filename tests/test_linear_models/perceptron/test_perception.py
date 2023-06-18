from sklearn.linear_model import Perceptron
from pymilo.utils.test_pymilo import test_pymilo_regression
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Perceptron"


def test_perceptron():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create perceptron regression object

    perceptron_random_state = 0
    perceptron_tol = 1e-3
    perceptron = Perceptron(
        random_state=perceptron_random_state,
        tol=perceptron_tol)
    # Train the model using the training sets
    perceptron.fit(x_train, y_train)
    assert test_pymilo_regression(
        perceptron, MODEL_NAME, (x_test, y_test)) == True 
