import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost")
from xgboost import XGBRegressor

from pymilo.utils.test_pymilo import pymilo_regression_test


def _synthetic_survival_dataset(n=200, n_features=5, censor_fraction=0.3, seed=0):
    """
    Build a small Cox-style dataset.

    XGBoost's `survival:cox` objective encodes labels as positive (observed event)
    or negative (right-censored at |y|) survival times.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    linpred = X[:, 0] - 0.5 * X[:, 1]
    times = rng.exponential(scale=np.exp(-linpred))
    censored_mask = rng.uniform(size=n) < censor_fraction
    y = np.where(censored_mask, -times, times)
    return X, y


def xgb_survival_cox():
    try:
        X, y = _synthetic_survival_dataset()
        model = XGBRegressor(
            objective="survival:cox",
            n_estimators=8,
            max_depth=3,
            random_state=0,
        )
        model.fit(X[:-30], y[:-30])
    except Exception as exc:
        pytest.skip(f"survival:cox is unavailable in this xgboost build: {exc}")
        return

    x_test = X[-30:]
    y_test = y[-30:]
    assert pymilo_regression_test(
        model, "XGBRegressor-SurvivalCox", (x_test, y_test)) == True
