from sklearn.linear_model import LassoLarsCV
from pymilo.utils.test_pymilo import test_pymilo_regression
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Lasso-Lars-CV-Regression"


def test_lasso_lars_cv():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Lasso Lars CV regression object
    lasso_cv = 5
    lasso_lars_cv_regression = LassoLarsCV(cv=lasso_cv)
    # Train the model using the training sets
    lasso_lars_cv_regression.fit(x_train, y_train)
    return test_pymilo_regression(
        lasso_lars_cv_regression, MODEL_NAME, (x_test, y_test))
