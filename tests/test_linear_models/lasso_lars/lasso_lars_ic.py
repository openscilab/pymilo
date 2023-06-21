from sklearn.linear_model import LassoLarsIC
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Lasso-Lars-IC-Regression"


def lasso_lars_ic():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Lasso Lars IC regression object
    lasso_criterian = "bic"
    lass_lars_ic_regression = LassoLarsIC(criterion=lasso_criterian)
    # Train the model using the training sets
    lass_lars_ic_regression.fit(x_train, y_train)
    assert pymilo_regression_test(
        lass_lars_ic_regression, MODEL_NAME, (x_test, y_test)) == True 
