import json
import os

import pytest
from numpy import allclose

from pymilo import Export, Import
from pymilo.pymilo_param import PYMILO_VERSION, XGBOOST_MODEL_TABLE, NOT_SUPPORTED
from pymilo.utils.data_exporter import prepare_simple_classification_datasets
from xgboost_test_helpers import cpu_estimator_kwargs, xgboost_available

pytestmark = pytest.mark.skipif(
    not xgboost_available or XGBOOST_MODEL_TABLE.get("XGBClassifier") == NOT_SUPPORTED,
    reason="xgboost is not installed",
)


def _fitted_classifier():
    from xgboost import XGBClassifier
    x_train, y_train, x_test, _ = prepare_simple_classification_datasets()
    model = XGBClassifier(**cpu_estimator_kwargs(XGBClassifier)).fit(x_train, y_train)
    return model, x_test


def test_to_json_is_human_readable_and_not_binary():
    model, x_test = _fitted_classifier()
    exported = Export(model)
    dumped = exported.to_json()
    payload = json.loads(dumped)

    assert payload["model_type"] == "XGBClassifier"
    assert payload["pymilo_version"] == PYMILO_VERSION
    assert "sklearn_version" in payload
    assert "xgboost_version" in payload
    assert isinstance(payload["xgboost_version"], str)

    booster_body = payload["data"]["_Booster"]["pymilo-xgboost-booster"]
    assert "model-json" in booster_body
    assert isinstance(booster_body["model-json"], dict)
    learner = booster_body["model-json"]["learner"]
    trees = learner["gradient_booster"]["model"]["trees"]
    assert isinstance(trees, list) and len(trees) > 0
    first_tree = trees[0]
    assert "split_conditions" in first_tree
    assert "left_children" in first_tree
    assert "base_weights" in first_tree
    # The official XGBoost JSON must be present as structured data, not a blob.
    assert not isinstance(booster_body["model-json"], (bytes, bytearray))
    assert "ubj" not in dumped.lower()
    assert booster_body["trees-dump"] is None or isinstance(booster_body["trees-dump"], list)
    assert "n_estimators" in payload["data"]
    assert payload["data"]["device"] == "cpu"

    imported = Import(json_dump=dumped).to_model()
    assert allclose(imported.predict_proba(x_test), model.predict_proba(x_test), rtol=1e-5, atol=1e-6)


def test_save_and_reload_json_file(tmp_path):
    model, x_test = _fitted_classifier()
    path = os.path.join(str(tmp_path), "xgb_classifier.json")
    Export(model).save(path)
    with open(path, "r") as handle:
        raw = handle.read()
    payload = json.loads(raw)
    assert payload["model_type"] == "XGBClassifier"
    imported = Import(file_adr=path).to_model()
    assert allclose(imported.predict(x_test), model.predict(x_test))


def test_unfitted_wrapper_roundtrip():
    from xgboost import XGBClassifier
    model = XGBClassifier(**cpu_estimator_kwargs(XGBClassifier))
    dumped = Export(model).to_json()
    payload = json.loads(dumped)
    assert payload["model_type"] == "XGBClassifier"
    assert "_Booster" not in payload["data"]
    imported = Import(json_dump=dumped).to_model()
    assert imported.n_estimators == model.n_estimators
    assert not imported.__sklearn_is_fitted__()


def test_batch_export_import_json(tmp_path):
    model, x_test = _fitted_classifier()
    directory = os.path.join(str(tmp_path), "batch")
    count = Export.batch_export([model, model], directory)
    assert count == 2
    imported_count, models = Import.batch_import(directory)
    assert imported_count == 2
    assert allclose(models[0].predict_proba(x_test), model.predict_proba(x_test), rtol=1e-5, atol=1e-6)


def test_booster_json_contains_version_and_objective():
    import xgboost as xgb
    from pymilo.utils.data_exporter import prepare_simple_classification_datasets
    x_train, y_train, _, _ = prepare_simple_classification_datasets()
    booster = xgb.train(
        {"max_depth": 2, "objective": "binary:logistic", "verbosity": 0, "device": "cpu", "nthread": 1},
        xgb.DMatrix(x_train, label=y_train),
        num_boost_round=3,
    )
    payload = json.loads(Export(booster).to_json())
    assert payload["model_type"] == "Booster"
    body = payload["data"]["pymilo-xgboost-booster"]
    assert body["model-json"]["version"]
    assert body["model-json"]["learner"]["objective"]["name"] == "binary:logistic"
    assert body["config"]["learner"]["generic_param"]["device"] == "cpu"
