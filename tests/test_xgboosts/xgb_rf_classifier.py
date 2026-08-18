from pymilo.utils.data_exporter import prepare_simple_classification_datasets
from pymilo.utils.test_pymilo import pymilo_classification_test
from xgboost_test_helpers import cpu_estimator_kwargs, xgboost_available

MODEL_NAME = "XGBRFClassifier"


def xgb_rf_classifier():
    if not xgboost_available:
        print("Model: " + MODEL_NAME + " is not supported in this python version.")
        return
    from xgboost import XGBRFClassifier
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    model = XGBRFClassifier(**cpu_estimator_kwargs(XGBRFClassifier)).fit(x_train, y_train)
    assert pymilo_classification_test(model, MODEL_NAME, (x_test, y_test))
