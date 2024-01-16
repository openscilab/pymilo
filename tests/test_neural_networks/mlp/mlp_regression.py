from sklearn.neural_network import MLPRegressor
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Multi Layer Perceptron Regression"


def multi_layer_perceptron_regression():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Passive Aggressive Regression object
    par_random_state = 2
    par_max_iter = 100
    multi_layer_perceptron_regression = MLPRegressor(random_state=1, max_iter=500).fit(x_train, y_train)
    # Train the model using the training sets
    multi_layer_perceptron_regression.fit(x_train, y_train)
    assert pymilo_regression_test(
        multi_layer_perceptron_regression, MODEL_NAME, (x_test, y_test)) == True 

