import pytest

xgboost = pytest.importorskip("xgboost")
from xgboost import XGBRFRegressor

from pymilo.utils.data_exporter import prepare_simple_regression_datasets
from pymilo.utils.test_pymilo import pymilo_regression_test


def xgb_rf_regressor():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    model = XGBRFRegressor(
        n_estimators=10,
        max_depth=4,
        random_state=0,
    )
    model.fit(x_train, y_train)
    assert pymilo_regression_test(model, "XGBRFRegressor", (x_test, y_test)) == True
