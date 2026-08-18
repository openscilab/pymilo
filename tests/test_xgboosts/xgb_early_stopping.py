from numpy import allclose
from pymilo.utils.data_exporter import prepare_simple_classification_datasets
from pymilo.utils.test_pymilo import pymilo_test, pymilo_classification_test
from xgboost_test_helpers import cpu_estimator_kwargs, xgboost_available

MODEL_NAME = "XGBClassifierEarlyStopping"


def xgb_early_stopping():
    if not xgboost_available:
        print("Model: " + MODEL_NAME + " is not supported in this python version.")
        return
    from xgboost import XGBClassifier
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    model = XGBClassifier(
        **cpu_estimator_kwargs(
            XGBClassifier,
            n_estimators=20,
            early_stopping_rounds=3,
            eval_metric="logloss",
        )
    )
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
    assert pymilo_classification_test(model, MODEL_NAME, (x_test, y_test))
    imported = pymilo_test(model, MODEL_NAME + "_attrs")
    assert imported.best_iteration == model.best_iteration
    assert imported.evals_result()["validation_0"]["logloss"]
    assert allclose(imported.predict_proba(x_test), model.predict_proba(x_test), rtol=1e-5, atol=1e-6)
