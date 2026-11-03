import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost")
from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression

from pymilo import Export, Import
from pymilo.utils.data_exporter import prepare_simple_regression_datasets
from pymilo.utils.test_pymilo import pymilo_regression_test, report_status


def xgb_inside_pipeline():
    """An XGBoost model nested in a sklearn Pipeline is routed through the compose chain."""
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBRegressor(n_estimators=8, max_depth=3, random_state=0)),
    ])
    model.fit(x_train, y_train)
    assert pymilo_regression_test(model, "Pipeline-XGBRegressor", (x_test, y_test)) == True


def xgb_inside_voting_regressor():
    """An XGBoost model nested in a sklearn ensemble is routed through the ensemble chain."""
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    model = VotingRegressor([
        ("xgb", XGBRegressor(n_estimators=8, max_depth=3, random_state=0)),
        ("linear", LinearRegression()),
    ])
    model.fit(x_train, y_train)
    assert pymilo_regression_test(model, "VotingRegressor-XGBRegressor", (x_test, y_test)) == True


def _same_param(left, right):
    """XGBoost's default `missing` hyperparameter is NaN, which never equals itself."""
    if isinstance(left, float) and isinstance(right, float) and np.isnan(left) and np.isnan(right):
        return True
    return left == right


def xgb_untrained_model():
    """An unfitted wrapper carries no inner Booster yet, its hyperparameters must still survive."""
    model = XGBRegressor(n_estimators=7, max_depth=4, random_state=0)
    retrieved = Import(json_dump=Export(model).to_json()).to_model()
    expected, retrieved_params = model.get_params(), retrieved.get_params()
    result = set(expected) == set(retrieved_params) and all(
        _same_param(expected[key], retrieved_params[key]) for key in expected)
    # NaN survives the JSON round-trip rather than degrading to None
    result = result and np.isnan(retrieved.missing)
    report_status(result, "XGBRegressor-Untrained")
    assert result == True


def xgb_json_dump_round_trip():
    """Importing from an in-memory JSON dump must be as lossless as importing from a file."""
    x_train, y_train, x_test, _ = prepare_simple_regression_datasets()
    model = XGBRegressor(n_estimators=8, max_depth=3, random_state=0)
    model.fit(x_train, y_train)
    retrieved = Import(json_dump=Export(model).to_json()).to_model()
    result = np.array_equal(model.predict(x_test), retrieved.predict(x_test))
    report_status(result, "XGBRegressor-JsonDump")
    assert result == True
