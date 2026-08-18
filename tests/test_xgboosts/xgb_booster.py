import xgboost as xgb
from numpy import allclose
from pymilo.utils.data_exporter import prepare_simple_classification_datasets
from pymilo.utils.test_pymilo import pymilo_prediction_test
from xgboost_test_helpers import xgboost_available

MODEL_NAME = "Booster"


def xgb_booster():
    if not xgboost_available:
        print("Model: " + MODEL_NAME + " is not supported in this python version.")
        return
    x_train, y_train, x_test, _ = prepare_simple_classification_datasets()
    dtrain = xgb.DMatrix(x_train, label=y_train)
    params = {
        "max_depth": 2,
        "eta": 0.3,
        "objective": "binary:logistic",
        "verbosity": 0,
        "nthread": 1,
        "device": "cpu",
    }
    booster = xgb.train(params, dtrain, num_boost_round=6)

    def _predict(current, data):
        return current.predict(xgb.DMatrix(data))

    assert pymilo_prediction_test(booster, MODEL_NAME, x_test, predict_fn=_predict)
    # Direct equality check kept for extra safety around DMatrix reconstruction.
    from pymilo.utils.test_pymilo import pymilo_test
    imported = pymilo_test(booster, MODEL_NAME + "_direct")
    assert allclose(_predict(booster, x_test), _predict(imported, x_test), rtol=1e-5, atol=1e-6)
