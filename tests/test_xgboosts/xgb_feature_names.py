from numpy import allclose, array_equal
from pymilo.utils.data_exporter import prepare_simple_classification_datasets
from pymilo.utils.test_pymilo import pymilo_test
from xgboost_test_helpers import cpu_estimator_kwargs, xgboost_available

MODEL_NAME = "XGBClassifierFeatureNames"


def xgb_feature_names():
    if not xgboost_available:
        print("Model: " + MODEL_NAME + " is not supported in this python version.")
        return
    from xgboost import XGBClassifier
    x_train, y_train, x_test, _ = prepare_simple_classification_datasets()
    model = XGBClassifier(**cpu_estimator_kwargs(XGBClassifier)).fit(x_train, y_train)
    names = ["feat_{0}".format(idx) for idx in range(x_train.shape[1])]
    model.get_booster().feature_names = names
    imported = pymilo_test(model, MODEL_NAME)
    assert imported.get_booster().feature_names == names
    assert array_equal(imported.feature_names_in_, model.feature_names_in_)
    assert allclose(imported.predict_proba(x_test), model.predict_proba(x_test), rtol=1e-5, atol=1e-6)
