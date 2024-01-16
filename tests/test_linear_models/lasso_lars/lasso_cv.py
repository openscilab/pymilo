from sklearn.linear_model import LassoCV
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Lasso-Regression-CV"


def lasso_cv():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Lasso CV regression object
    lasso_alphas = [1e-3, 1e-2, 1e-1, 1]
    lasso_cv = 5
    lasso_random_state = 0
    lasso_cv_regression = LassoCV(
        alphas=lasso_alphas,
        cv=lasso_cv,
        random_state=lasso_random_state)
    # Train the model using the training sets
    lasso_cv_regression.fit(x_train, y_train)
    assert pymilo_regression_test(
        lasso_cv_regression, MODEL_NAME, (x_test, y_test)) == True
