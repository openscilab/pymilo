from pymilo.utils.data_exporter import prepare_simple_ranking_datasets
from pymilo.utils.test_pymilo import pymilo_prediction_test
from xgboost_test_helpers import cpu_estimator_kwargs, xgboost_available

MODEL_NAME = "XGBRanker"


def xgb_ranker():
    if not xgboost_available:
        print("Model: " + MODEL_NAME + " is not supported in this python version.")
        return
    from xgboost import XGBRanker
    x_train, y_train, qid_train, x_test, _, _ = prepare_simple_ranking_datasets()
    model = XGBRanker(**cpu_estimator_kwargs(XGBRanker)).fit(x_train, y_train, qid=qid_train)
    assert pymilo_prediction_test(model, MODEL_NAME, x_test)
