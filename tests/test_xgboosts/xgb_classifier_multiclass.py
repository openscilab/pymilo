from sklearn.datasets import make_classification
from pymilo.utils.test_pymilo import pymilo_prediction_test
from xgboost_test_helpers import cpu_estimator_kwargs, xgboost_available

MODEL_NAME = "XGBClassifierMultiClass"


def xgb_classifier_multiclass():
    if not xgboost_available:
        print("Model: " + MODEL_NAME + " is not supported in this python version.")
        return
    from xgboost import XGBClassifier
    x, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        random_state=0,
    )
    x_train, y_train, x_test = x[:90], y[:90], x[90:]
    model = XGBClassifier(
        **cpu_estimator_kwargs(XGBClassifier, objective="multi:softprob")
    ).fit(x_train, y_train)
    assert pymilo_prediction_test(model, MODEL_NAME, x_test)
    imported = __import__("pymilo.utils.test_pymilo", fromlist=["pymilo_test"]).pymilo_test(model, MODEL_NAME + "_proba_check")
    assert (imported.predict_proba(x_test).shape == model.predict_proba(x_test).shape)
    from numpy import allclose
    assert allclose(imported.predict_proba(x_test), model.predict_proba(x_test), rtol=1e-5, atol=1e-6)
