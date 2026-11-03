import pytest

xgboost = pytest.importorskip("xgboost")
from xgboost import XGBRFClassifier

from pymilo.utils.data_exporter import prepare_simple_classification_datasets
from pymilo.utils.test_pymilo import pymilo_classification_test


def xgb_rf_classifier():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    model = XGBRFClassifier(
        n_estimators=10,
        max_depth=4,
        random_state=0,
    )
    model.fit(x_train, y_train)
    assert pymilo_classification_test(model, "XGBRFClassifier", (x_test, y_test)) == True
