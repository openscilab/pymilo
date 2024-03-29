from sklearn.linear_model import RidgeCV
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Ridge-Regression-CV"


def ridge_regression_cv():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create ridgeCV regression object
    ridge_cv_alphas = [1e-3, 1e-2, 1e-1, 1]
    ridge_regression_cv = RidgeCV(alphas=ridge_cv_alphas)
    # Train the model using the training sets
    ridge_regression_cv.fit(x_train, y_train)
    assert pymilo_regression_test(
        ridge_regression_cv, MODEL_NAME, (x_test, y_test)) == True
