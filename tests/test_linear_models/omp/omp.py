from sklearn.linear_model import OrthogonalMatchingPursuit
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Orthogonal-Matching-Pursuit-Regression"


def omp():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Orthogonal Matching Pursuit regression object
    omp_n_nonzero_coefs = 10
    omp_regression = OrthogonalMatchingPursuit(
        n_nonzero_coefs=omp_n_nonzero_coefs)
    # Train the model using the training sets
    omp_regression.fit(x_train, y_train)
    assert pymilo_regression_test(
        omp_regression, MODEL_NAME, (x_test, y_test)) == True
