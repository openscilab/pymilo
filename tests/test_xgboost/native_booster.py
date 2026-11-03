import pytest

xgboost = pytest.importorskip("xgboost")

from pymilo.utils.data_exporter import prepare_simple_regression_datasets, prepare_simple_classification_datasets
from pymilo.utils.test_pymilo import pymilo_xgb_booster_test


def native_booster_regression():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    dtrain = xgboost.DMatrix(x_train, label=y_train)
    params = {
        "objective": "reg:squarederror",
        "max_depth": 3,
        "eta": 0.1,
        "verbosity": 0,
    }
    booster = xgboost.train(params, dtrain, num_boost_round=8)
    assert pymilo_xgb_booster_test(booster, "NativeBooster-Regression", (x_test, y_test)) == True


def native_booster_classification():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    dtrain = xgboost.DMatrix(x_train, label=y_train)
    params = {
        "objective": "binary:logistic",
        "max_depth": 3,
        "eta": 0.1,
        "verbosity": 0,
    }
    booster = xgboost.train(params, dtrain, num_boost_round=8)
    assert pymilo_xgb_booster_test(booster, "NativeBooster-Binary", (x_test, y_test)) == True
