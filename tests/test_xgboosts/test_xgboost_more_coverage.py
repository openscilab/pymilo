import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pymilo.chains.xgboost_chain import (
    deserialize_xgboost_booster,
    xgboost_chain,
)
from pymilo.pymilo_func import get_xgboost_version
from pymilo.pymilo_param import XGBOOST_MODEL_TABLE, NOT_SUPPORTED, XGBOOST_NOT_INSTALLED
from pymilo.transporters.xgboost_transporter import (
    PYMILO_XGBOOST_BOOSTER,
    XGBoostTransporter,
    _looks_like_json,
    _restore_optional_field,
    _safe_call,
    _safe_getattr,
    booster_from_transparent_dict,
    booster_to_transparent_dict,
    extract_requested_device,
    is_gpu_device,
    is_xgboost_booster,
    is_xgboost_gpu_available,
    rewrite_gpu_fields_to_cpu,
)
from pymilo.utils.data_exporter import prepare_simple_classification_datasets
from xgboost_test_helpers import cpu_estimator_kwargs, xgboost_available

pytestmark = pytest.mark.skipif(
    not xgboost_available or XGBOOST_MODEL_TABLE.get("XGBClassifier") == NOT_SUPPORTED,
    reason="xgboost is not installed",
)


def test_helpers_cover_error_and_edge_paths():
    assert _looks_like_json(b"") is False
    assert _looks_like_json(b"  {\"a\": 1}") is True
    assert _looks_like_json(b"[]") is True
    assert _looks_like_json(b"\x00binary") is False
    assert _safe_call(lambda: (_ for _ in ()).throw(RuntimeError("x"))) is None
    assert _safe_getattr(object(), "missing") is None
    assert _restore_optional_field(lambda: (_ for _ in ()).throw(RuntimeError("x"))) is False
    assert _restore_optional_field(lambda: None) is True
    assert extract_requested_device(None, None, fallback="cpu") == "cpu"
    assert extract_requested_device({"learner": {"generic_param": {"device": "cuda"}}}, None) == "cuda"
    rewrite_gpu_fields_to_cpu("scalar")
    assert is_gpu_device("CUDA:1") is True


def test_instantiate_unknown_type_raises():
    with pytest.raises(ValueError):
        xgboost_chain._instantiate("NotAnXGBModel")


def test_deserialize_xgboost_booster_accepts_bare_payload():
    import xgboost as xgb
    x_train, y_train, x_test, _ = prepare_simple_classification_datasets()
    booster = xgb.train(
        {"max_depth": 1, "objective": "binary:logistic", "verbosity": 0, "device": "cpu", "nthread": 1},
        xgb.DMatrix(x_train, label=y_train),
        num_boost_round=2,
    )
    bare = booster_to_transparent_dict(booster)
    restored = deserialize_xgboost_booster(bare)
    from numpy import allclose
    assert allclose(booster.predict(xgb.DMatrix(x_test)), restored.predict(xgb.DMatrix(x_test)), rtol=1e-5, atol=1e-6)


def test_inner_model_deserialize_roundtrip():
    from xgboost import XGBClassifier
    from pymilo import Export
    x_train, y_train, x_test, _ = prepare_simple_classification_datasets()
    model = XGBClassifier(**cpu_estimator_kwargs(XGBClassifier)).fit(x_train, y_train)
    payload = json.loads(Export(model).to_json())
    restored = xgboost_chain.deserialize(
        {"data": payload["data"], "type": payload["model_type"]},
        is_inner_model=True,
    )
    from numpy import allclose
    assert allclose(restored.predict_proba(x_test), model.predict_proba(x_test), rtol=1e-5, atol=1e-6)


def test_setattr_attribute_error_is_skipped():
    from xgboost import XGBClassifier
    from pymilo import Export
    x_train, y_train, _, _ = prepare_simple_classification_datasets()
    model = XGBClassifier(**cpu_estimator_kwargs(XGBClassifier)).fit(x_train, y_train)
    payload = json.loads(Export(model).to_json())
    payload["data"]["classes_"] = [0, 1]
    restored = xgboost_chain.deserialize(
        {"data": payload["data"], "type": "XGBClassifier"},
        is_inner_model=True,
    )
    assert restored.n_classes_ == 2


def test_transporter_serializes_missing_nan_and_eval_metric():
    transporter = XGBoostTransporter()
    data = {"missing": float("nan"), "eval_metric": (lambda y, p: 0.0), "other": 1}
    assert transporter.serialize(data, "missing", "XGBClassifier")["np-type"] == "numpy.nan"
    serialized_metric = transporter.serialize(data, "eval_metric", "XGBClassifier")
    assert "pymilo-xgboost-callable" in serialized_metric
    assert transporter.serialize(data, "other", "XGBClassifier") == 1
    assert transporter.deserialize({"eval_metric": serialized_metric}, "eval_metric", "XGBClassifier")
    assert transporter.deserialize(
        {"callbacks": {"pymilo-xgboost-callbacks": ["EvaluationMonitor"]}},
        "callbacks",
        "XGBClassifier",
    ) is None
    assert transporter.deserialize({"plain": 3}, "plain", "XGBClassifier") == 3


def test_booster_from_dict_without_gpu_remap():
    import xgboost as xgb
    x_train, y_train, _, _ = prepare_simple_classification_datasets()
    booster = xgb.train(
        {"max_depth": 1, "objective": "binary:logistic", "verbosity": 0, "device": "cpu", "nthread": 1},
        xgb.DMatrix(x_train, label=y_train),
        num_boost_round=1,
    )
    payload = booster_to_transparent_dict(booster)
    restored = booster_from_transparent_dict(payload, map_gpu_to_cpu=False)
    assert restored.num_features() == booster.num_features()


def test_support_guards_when_xgboost_flag_is_false():
    with patch("pymilo.chains.xgboost_chain.xgboost_support", False):
        with pytest.raises(ImportError) as exc_info:
            xgboost_chain.serialize(SimpleNamespace())
        assert XGBOOST_NOT_INSTALLED in str(exc_info.value)
        with pytest.raises(ImportError):
            xgboost_chain.deserialize(SimpleNamespace(data={}, type="XGBClassifier"))
    with patch("pymilo.transporters.xgboost_transporter.xgboost_support", False):
        assert is_xgboost_booster(object()) is False
        assert is_xgboost_gpu_available() is False
    with patch("pymilo.pymilo_func.xgboost_support", False):
        assert get_xgboost_version() is None


def test_get_xgboost_version_handles_import_failure():
    import pymilo.pymilo_func as func

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("simulated missing xgboost")
        return real_import(name, *args, **kwargs)

    with patch.object(func, "xgboost_support", True):
        with patch("builtins.__import__", side_effect=fake_import):
            assert func.get_xgboost_version() is None


def test_gpu_availability_build_info_exception():
    import xgboost
    with patch.object(xgboost, "build_info", side_effect=RuntimeError("no build info")):
        assert is_xgboost_gpu_available() is False
