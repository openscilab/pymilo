import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost")
from xgboost import XGBRegressor

from pymilo.utils.data_exporter import prepare_simple_regression_datasets
from pymilo.utils.test_pymilo import pymilo_regression_test


REGRESSION_OBJECTIVES = [
    ("reg:squarederror", "XGBRegressor-SquaredError"),
    ("reg:absoluteerror", "XGBRegressor-AbsoluteError"),
    ("reg:gamma", "XGBRegressor-Gamma"),
    ("reg:tweedie", "XGBRegressor-Tweedie"),
]


def _positive_targets(y):
    """Gamma and Tweedie require strictly positive targets."""
    y = np.asarray(y, dtype=float)
    return np.where(y > 0, y, 1e-3)


def xgb_regressor_all_objectives():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    for objective, name in REGRESSION_OBJECTIVES:
        if objective in ("reg:gamma", "reg:tweedie"):
            y_train_use = _positive_targets(y_train)
            y_test_use = _positive_targets(y_test)
        else:
            y_train_use = y_train
            y_test_use = y_test
        model = XGBRegressor(
            objective=objective,
            n_estimators=8,
            max_depth=3,
            random_state=0,
        )
        model.fit(x_train, y_train_use)
        assert pymilo_regression_test(model, name, (x_test, y_test_use)) == True
