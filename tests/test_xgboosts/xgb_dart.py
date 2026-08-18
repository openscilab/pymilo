from pymilo.utils.data_exporter import prepare_simple_regression_datasets
from pymilo.utils.test_pymilo import pymilo_regression_test
from xgboost_test_helpers import cpu_estimator_kwargs, xgboost_available

MODEL_NAME = "XGBRegressorDART"


def xgb_dart():
    if not xgboost_available:
        print("Model: " + MODEL_NAME + " is not supported in this python version.")
        return
    from xgboost import XGBRegressor
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    model = XGBRegressor(
        **cpu_estimator_kwargs(XGBRegressor, booster="dart", rate_drop=0.1)
    ).fit(x_train, y_train)
    assert pymilo_regression_test(model, MODEL_NAME, (x_test, y_test))
