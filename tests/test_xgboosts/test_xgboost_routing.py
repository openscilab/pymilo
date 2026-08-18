import json
import warnings

import pytest

from pymilo import Export, Import
from pymilo.chains.util import get_transporter
from pymilo.chains.xgboost_chain import (
    deserialize_xgboost_booster,
    is_xgboost_model,
    serialize_xgboost_booster,
    xgboost_chain,
)
from pymilo.pymilo_func import get_xgboost_version, print_supported_ml_models
from pymilo.pymilo_param import UNEQUAL_XGBOOST_VERSIONS, XGBOOST_MODEL_TABLE, NOT_SUPPORTED
from pymilo.utils.data_exporter import prepare_simple_classification_datasets
from pymilo.utils.util import get_sklearn_class, get_sklearn_type
from xgboost_test_helpers import cpu_estimator_kwargs, xgboost_available

pytestmark = pytest.mark.skipif(
    not xgboost_available or XGBOOST_MODEL_TABLE.get("XGBClassifier") == NOT_SUPPORTED,
    reason="xgboost is not installed",
)


def test_get_transporter_routes_wrappers_and_category_string():
    from xgboost import XGBClassifier, XGBRegressor, Booster
    assert get_transporter("XGBOOST")[0] == "XGBOOST"
    assert get_transporter("XGBClassifier")[0] == "XGBOOST"
    assert get_transporter(XGBClassifier())[0] == "XGBOOST"
    assert get_transporter(XGBRegressor())[0] == "XGBOOST"
    assert get_transporter(Booster())[0] == "XGBOOST"
    assert is_xgboost_model("XGBRanker") is True
    assert is_xgboost_model(XGBClassifier()) is True


def test_get_sklearn_class_and_type():
    from xgboost import XGBClassifier
    assert get_sklearn_class("XGBClassifier") is XGBClassifier
    assert get_sklearn_class("Booster") is not None
    assert get_sklearn_type(XGBClassifier()) == "XGBClassifier"


def test_print_supported_ml_models_includes_xgboost(capsys):
    print_supported_ml_models()
    captured = capsys.readouterr().out
    assert "XGBOOST" in captured
    assert "XGBClassifier" in captured
    assert "Booster" in captured


def test_get_xgboost_version_matches_package():
    import xgboost
    assert get_xgboost_version() == xgboost.__version__


def test_unequal_xgboost_version_warns():
    from xgboost import XGBClassifier
    x_train, y_train, _, _ = prepare_simple_classification_datasets()
    model = XGBClassifier(**cpu_estimator_kwargs(XGBClassifier)).fit(x_train, y_train)
    payload = json.loads(Export(model).to_json())
    payload["xgboost_version"] = "0.0.0-fake"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Import(json_dump=json.dumps(payload))
    messages = [str(item.message) for item in caught]
    assert any(UNEQUAL_XGBOOST_VERSIONS in message for message in messages)


def test_serialize_helpers_roundtrip_booster():
    import xgboost as xgb
    from numpy import allclose
    x_train, y_train, x_test, _ = prepare_simple_classification_datasets()
    booster = xgb.train(
        {"max_depth": 2, "objective": "binary:logistic", "verbosity": 0, "device": "cpu", "nthread": 1},
        xgb.DMatrix(x_train, label=y_train),
        num_boost_round=3,
    )
    wrapped = serialize_xgboost_booster(booster)
    restored = deserialize_xgboost_booster(wrapped)
    assert allclose(
        booster.predict(xgb.DMatrix(x_test)),
        restored.predict(xgb.DMatrix(x_test)),
        rtol=1e-5,
        atol=1e-6,
    )
    assert xgboost_chain.is_supported(booster) is True


def test_callbacks_and_callable_eval_metric_are_serialized():
    from xgboost import XGBClassifier, callback
    x_train, y_train, _, _ = prepare_simple_classification_datasets()

    def custom_metric(y_true, y_pred):
        return 0.0

    kwargs = cpu_estimator_kwargs(XGBClassifier, callbacks=[callback.EvaluationMonitor(period=10)])
    model = XGBClassifier(**kwargs)
    # eval_metric callables are constructor-accepted; keep training simple.
    model.fit(x_train, y_train)
    model.eval_metric = custom_metric
    dumped = json.loads(Export(model).to_json())
    assert "pymilo-xgboost-callbacks" in dumped["data"]["callbacks"]
    imported = Import(json_dump=json.dumps(dumped)).to_model()
    assert imported.n_estimators == model.n_estimators
